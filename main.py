import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


DEFAULT_DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-3B-Instruct"

DEFAULT_PROMPTS = [
    "Explain how speculative decoding speeds up text generation.",
    "Write a short paragraph about the benefits of unit testing in production systems.",
    "Summarize the causes of the French Revolution in simple language.",
    "Give three practical tips for debugging CUDA out-of-memory errors.",
    "Describe how HTTP caching improves web application performance.",
    "Write a concise overview of gradient descent for a beginner.",
    "Explain the difference between latency and throughput with an everyday analogy.",
    "List the tradeoffs between Python multiprocessing and multithreading.",
    "Describe why reproducibility matters in machine learning experiments.",
    "Summarize the CAP theorem and its practical implications.",
]


@dataclass
class BenchmarkResult:
    method: str
    total_runtime_s: float
    tokens_per_second: float
    avg_latency_ms: float
    generated_tokens: int
    prompts_run: int
    speedup_vs_greedy: float = 1.0
    avg_acceptance_rate: float | None = None
    avg_accepted_tokens_per_step: float | None = None
    extra_metrics: Dict[str, float] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark greedy and speculative decoding.")
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--quantize-4bit", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_prompts(path: str | None) -> List[str]:
    if not path:
        return DEFAULT_PROMPTS
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".json"):
            payload = json.load(handle)
            if not isinstance(payload, list) or not all(isinstance(x, str) for x in payload):
                raise ValueError("JSON prompts file must contain a list of strings.")
            return payload
        prompts = [line.strip() for line in handle if line.strip()]
        if not prompts:
            raise ValueError("Prompt file is empty.")
        return prompts


def build_quant_config(enabled: bool) -> BitsAndBytesConfig | None:
    if not enabled:
        return None
    if BitsAndBytesConfig is None:
        raise ImportError("bitsandbytes support is unavailable in this transformers installation.")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model(
    model_name: str,
    device: str,
    dtype: torch.dtype | str,
    quantize_4bit: bool,
    trust_remote_code: bool,
):
    quant_config = build_quant_config(quantize_4bit)
    kwargs = {
        "trust_remote_code": trust_remote_code,
    }
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
        kwargs["device_map"] = {"": device}
        if dtype != "auto":
            kwargs["torch_dtype"] = dtype
    else:
        if dtype != "auto":
            kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if quant_config is None:
        model.to(device)
    model.eval()
    return model


def load_tokenizer(model_name: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def choose_next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if logits.ndim == 3 and logits.shape[1] == 1:
        logits = logits[:, -1, :]

    if temperature <= 0:
        if logits.ndim == 2:
            return torch.argmax(logits, dim=-1, keepdim=True)
        return torch.argmax(logits, dim=-1)

    probs = torch.softmax(logits / temperature, dim=-1)
    vocab_size = probs.shape[-1]
    probs_2d = probs.reshape(-1, vocab_size)
    samples = torch.multinomial(probs_2d, num_samples=1)
    if logits.ndim == 2:
        return samples.reshape(probs.shape[0], 1)
    return samples.reshape(*probs.shape[:-1])


def truncate_kv_cache(past_key_values, keep_length: int):
    if past_key_values is None:
        return None

    if hasattr(past_key_values, "crop") and callable(past_key_values.crop):
        past_key_values.crop(keep_length)
        return past_key_values

    truncated_layers = []
    for layer in past_key_values:
        truncated_tensors = []
        for tensor in layer:
            if tensor is None:
                truncated_tensors.append(None)
            elif tensor.ndim >= 3:
                truncated_tensors.append(tensor[:, :, :keep_length, :])
            else:
                truncated_tensors.append(tensor)
        truncated_layers.append(tuple(truncated_tensors))
    return tuple(truncated_layers)


def warmup_model_state(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    return outputs.past_key_values, outputs.logits[:, -1, :]


def greedy_generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
) -> Dict[str, object]:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    generated_tokens: List[torch.Tensor] = []
    model_calls = 0

    start = time.perf_counter()
    with torch.inference_mode():
        past_key_values, next_logits = warmup_model_state(model, input_ids=input_ids, attention_mask=attention_mask)
        model_calls += 1
        for _ in range(max_new_tokens):
            next_token = choose_next_token(next_logits, temperature)
            generated_tokens.append(next_token.cpu())
            outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            model_calls += 1
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    runtime = time.perf_counter() - start
    new_tokens = torch.cat(generated_tokens, dim=-1) if generated_tokens else torch.empty((1, 0), dtype=input_ids.dtype)
    return {
        "generated_text": tokenizer.decode(new_tokens[0], skip_special_tokens=True),
        "generated_tokens": max_new_tokens,
        "runtime_s": runtime,
        "model_calls": model_calls,
    }


def speculative_generate(
    draft_model,
    target_model,
    tokenizer,
    prompt: str,
    draft_device: str,
    target_device: str,
    max_new_tokens: int,
    k: int,
    temperature: float,
) -> Dict[str, object]:
    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_ids_draft = encoded["input_ids"].to(draft_device)
    prompt_mask_draft = encoded["attention_mask"].to(draft_device)
    prompt_ids_target = encoded["input_ids"].to(target_device)
    prompt_mask_target = encoded["attention_mask"].to(target_device)

    steps = 0
    total_accepted = 0
    generated_tokens: List[torch.Tensor] = []
    draft_model_calls = 0
    target_model_calls = 0
    transferred_proposal_tokens = 0
    transferred_replacement_tokens = 0
    total_proposed = 0

    start = time.perf_counter()
    with torch.inference_mode():
        draft_outputs = draft_model(input_ids=prompt_ids_draft, attention_mask=prompt_mask_draft, use_cache=True)
        draft_kv = draft_outputs.past_key_values
        draft_model_calls += 1

        target_outputs = target_model(input_ids=prompt_ids_target, attention_mask=prompt_mask_target, use_cache=True)
        target_kv = target_outputs.past_key_values
        target_logits = target_outputs.logits[:, -1, :]
        target_model_calls += 1

        next_token_target = choose_next_token(target_logits, temperature)
        next_token_draft = next_token_target.to(draft_device)

        generated_tokens.append(next_token_target.cpu())
        committed_tokens = 1
        seq_len = prompt_ids_target.shape[1]

        while committed_tokens < max_new_tokens:
            steps += 1
            proposal_len = min(k, max_new_tokens - committed_tokens)
            total_proposed += proposal_len
            proposals: List[torch.Tensor] = []
            curr_draft_kv = draft_kv
            curr_token = next_token_draft

            for _ in range(proposal_len):
                out = draft_model(input_ids=curr_token, past_key_values=curr_draft_kv, use_cache=True)
                curr_draft_kv = out.past_key_values
                curr_token = choose_next_token(out.logits[:, -1, :], temperature)
                proposals.append(curr_token)
                draft_model_calls += 1

            proposal_tensor_draft = torch.cat(proposals, dim=-1)
            proposal_tensor_target = proposal_tensor_draft.to(target_device)
            transferred_proposal_tokens += proposal_len

            target_input = torch.cat([next_token_target, proposal_tensor_target], dim=-1)
            out = target_model(input_ids=target_input, past_key_values=target_kv, use_cache=True)
            curr_target_kv = out.past_key_values
            target_preds = choose_next_token(out.logits, temperature)
            target_model_calls += 1

            mismatch_index = proposal_len
            for idx in range(proposal_len):
                if proposal_tensor_target[0, idx] != target_preds[0, idx]:
                    mismatch_index = idx
                    break

            accepted_proposals = proposal_tensor_target[:, :mismatch_index]
            replacement_token = target_preds[:, mismatch_index : mismatch_index + 1]

            if mismatch_index > 0:
                generated_tokens.append(accepted_proposals.cpu())
            if committed_tokens + mismatch_index < max_new_tokens:
                generated_tokens.append(replacement_token.cpu())
                transferred_replacement_tokens += 1

            total_accepted += mismatch_index
            committed_tokens += mismatch_index + 1

            keep_length = seq_len + 1 + mismatch_index
            target_kv = truncate_kv_cache(curr_target_kv, keep_length)

            if mismatch_index == proposal_len:
                out_draft_extra = draft_model(input_ids=proposal_tensor_draft[:, -1:], past_key_values=curr_draft_kv, use_cache=True)
                draft_kv = out_draft_extra.past_key_values
                draft_model_calls += 1
            else:
                draft_kv = truncate_kv_cache(curr_draft_kv, keep_length)

            seq_len = keep_length
            next_token_target = replacement_token
            next_token_draft = next_token_target.to(draft_device)
    if draft_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(draft_device)
    if target_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(target_device)
    runtime = time.perf_counter() - start

    new_tokens = torch.cat(generated_tokens, dim=-1)[:, :max_new_tokens]
    acceptance_rate = total_accepted / total_proposed if total_proposed else math.nan
    draft_fraction_of_output = total_accepted / max_new_tokens if max_new_tokens else math.nan
    accepted_per_step = total_accepted / steps if steps else math.nan

    return {
        "generated_text": tokenizer.decode(new_tokens[0], skip_special_tokens=True),
        "generated_tokens": max_new_tokens,
        "runtime_s": runtime,
        "acceptance_rate": acceptance_rate,
        "draft_fraction_of_output": draft_fraction_of_output,
        "accepted_tokens_per_step": accepted_per_step,
        "draft_model_calls": draft_model_calls,
        "target_model_calls": target_model_calls,
        "proposal_tokens_transferred": transferred_proposal_tokens,
        "replacement_tokens_transferred": transferred_replacement_tokens,
        "speculative_steps": steps,
        "total_proposed": total_proposed,
    }


def aggregate_results(
    method: str,
    runs: Sequence[Dict[str, object]],
    speedup_vs_greedy: float = 1.0,
) -> BenchmarkResult:
    total_runtime = sum(float(run["runtime_s"]) for run in runs)
    total_tokens = sum(int(run["generated_tokens"]) for run in runs)
    tokens_per_second = total_tokens / total_runtime if total_runtime > 0 else math.inf
    avg_latency_ms = (total_runtime / total_tokens) * 1000 if total_tokens > 0 else math.inf

    acceptance_rate = None
    accepted_tokens_per_step = None
    extra_metrics = None
    if "acceptance_rate" in runs[0]:
        acceptance_rate = statistics.mean(float(run["acceptance_rate"]) for run in runs)
        accepted_tokens_per_step = statistics.mean(float(run["accepted_tokens_per_step"]) for run in runs)
        extra_metrics = {
            "draft_fraction_of_output": statistics.mean(float(run["draft_fraction_of_output"]) for run in runs),
            "draft_model_calls": statistics.mean(float(run["draft_model_calls"]) for run in runs),
            "target_model_calls": statistics.mean(float(run["target_model_calls"]) for run in runs),
            "proposal_tokens_transferred": statistics.mean(float(run["proposal_tokens_transferred"]) for run in runs),
            "replacement_tokens_transferred": statistics.mean(float(run["replacement_tokens_transferred"]) for run in runs),
            "speculative_steps": statistics.mean(float(run["speculative_steps"]) for run in runs),
            "total_proposed": statistics.mean(float(run["total_proposed"]) for run in runs),
        }
    elif "model_calls" in runs[0]:
        extra_metrics = {
            "model_calls": statistics.mean(float(run["model_calls"]) for run in runs),
        }

    return BenchmarkResult(
        method=method,
        total_runtime_s=total_runtime,
        tokens_per_second=tokens_per_second,
        avg_latency_ms=avg_latency_ms,
        generated_tokens=total_tokens,
        prompts_run=len(runs),
        speedup_vs_greedy=speedup_vs_greedy,
        avg_acceptance_rate=acceptance_rate,
        avg_accepted_tokens_per_step=accepted_tokens_per_step,
        extra_metrics=extra_metrics,
    )


def print_results_table(results: Sequence[BenchmarkResult]) -> None:
    headers = [
        "method",
        "tok/s",
        "avg_latency_ms",
        "speedup",
        "runtime_s",
        "accept_rate",
        "accepted/step",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result.method,
                f"{result.tokens_per_second:.2f}",
                f"{result.avg_latency_ms:.2f}",
                f"{result.speedup_vs_greedy:.2f}x",
                f"{result.total_runtime_s:.2f}",
                "-" if result.avg_acceptance_rate is None else f"{result.avg_acceptance_rate:.3f}",
                "-" if result.avg_accepted_tokens_per_step is None else f"{result.avg_accepted_tokens_per_step:.2f}",
            ]
        )

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    separator = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(line)
    print(separator)
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))

    print()
    for result in results:
        if not result.extra_metrics:
            continue
        print(f"[{result.method}]")
        for key, value in result.extra_metrics.items():
            print(f"  {key}: {value:.2f}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    prompts = load_prompts(args.prompts_file)
    dtype = resolve_dtype(args.dtype)

    print(f"Loading tokenizer from {args.target_model}")
    tokenizer = load_tokenizer(args.target_model, trust_remote_code=args.trust_remote_code)

    print(f"Loading target model on {args.target_device}: {args.target_model}")
    target_model = load_model(
        model_name=args.target_model,
        device=args.target_device,
        dtype=dtype,
        quantize_4bit=args.quantize_4bit,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Loading draft model on {args.draft_device}: {args.draft_model}")
    draft_model = load_model(
        model_name=args.draft_model,
        device=args.draft_device,
        dtype=dtype,
        quantize_4bit=args.quantize_4bit,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Running greedy baseline for {len(prompts)} prompts, {args.max_new_tokens} new tokens each")
    greedy_runs = [
        greedy_generate(
            model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=args.target_device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        for prompt in prompts
    ]
    greedy_result = aggregate_results("greedy", greedy_runs)
    results = [greedy_result]

    for k in args.ks:
        print(f"Running speculative decoding with k={k}")
        speculative_runs = [
            speculative_generate(
                draft_model=draft_model,
                target_model=target_model,
                tokenizer=tokenizer,
                prompt=prompt,
                draft_device=args.draft_device,
                target_device=args.target_device,
                max_new_tokens=args.max_new_tokens,
                k=k,
                temperature=args.temperature,
            )
            for prompt in prompts
        ]
        speculative_summary = aggregate_results(method=f"speculative_k={k}", runs=speculative_runs)
        speculative_result = aggregate_results(
            method=speculative_summary.method,
            runs=speculative_runs,
            speedup_vs_greedy=speculative_summary.tokens_per_second / greedy_result.tokens_per_second,
        )
        results.append(speculative_result)

    print()
    print_results_table(results)


if __name__ == "__main__":
    main()
