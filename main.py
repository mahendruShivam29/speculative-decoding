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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark greedy and speculative decoding.")
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--quantize-4bit", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
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
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


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
    generated = input_ids
    mask = attention_mask

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=generated, attention_mask=mask, use_cache=False)
            next_token = choose_next_token(outputs.logits[:, -1, :], temperature)
            generated = torch.cat([generated, next_token], dim=-1)
            next_mask = torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=device)
            mask = torch.cat([mask, next_mask], dim=-1)
    runtime = time.perf_counter() - start
    new_tokens = generated[:, input_ids.shape[1] :]
    return {
        "generated_text": tokenizer.decode(new_tokens[0], skip_special_tokens=True),
        "generated_tokens": max_new_tokens,
        "runtime_s": runtime,
    }


def draft_propose(
    model,
    context_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int,
    temperature: float,
) -> torch.Tensor:
    generated = context_ids
    mask = attention_mask
    proposals: List[torch.Tensor] = []

    with torch.inference_mode():
        for _ in range(k):
            outputs = model(input_ids=generated, attention_mask=mask, use_cache=False)
            next_token = choose_next_token(outputs.logits[:, -1, :], temperature)
            proposals.append(next_token)
            generated = torch.cat([generated, next_token], dim=-1)
            next_mask = torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)
            mask = torch.cat([mask, next_mask], dim=-1)

    return torch.cat(proposals, dim=-1)


def verify_proposals(
    model,
    context_ids: torch.Tensor,
    proposal_ids: torch.Tensor,
    device: str,
    temperature: float,
) -> tuple[torch.Tensor, int, bool]:
    context_on_target = context_ids.to(device)
    proposal_on_target = proposal_ids.to(device)
    combined = torch.cat([context_on_target, proposal_on_target], dim=-1)
    attention_mask = torch.ones_like(combined, device=device)

    with torch.inference_mode():
        outputs = model(input_ids=combined, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits

    context_len = context_on_target.shape[1]
    proposal_len = proposal_on_target.shape[1]
    verify_logits = logits[:, context_len - 1 : context_len - 1 + proposal_len, :]

    accepted = 0
    replacement_token = None
    all_accepted = True

    for offset in range(proposal_len):
        target_token = choose_next_token(verify_logits[:, offset, :], temperature)
        draft_token = proposal_on_target[:, offset : offset + 1]
        if torch.equal(target_token, draft_token):
            accepted += 1
            continue
        replacement_token = target_token
        all_accepted = False
        break

    if replacement_token is None:
        replacement_token = torch.empty((proposal_on_target.shape[0], 0), dtype=proposal_on_target.dtype, device=device)

    return replacement_token, accepted, all_accepted


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
    generated = encoded["input_ids"].to(draft_device)
    mask = encoded["attention_mask"].to(draft_device)
    original_length = generated.shape[1]
    steps = 0
    total_accepted = 0

    start = time.perf_counter()
    while generated.shape[1] - original_length < max_new_tokens:
        steps += 1
        remaining = max_new_tokens - (generated.shape[1] - original_length)
        proposal_len = min(k, remaining)
        proposal_ids = draft_propose(
            model=draft_model,
            context_ids=generated,
            attention_mask=mask,
            k=proposal_len,
            temperature=temperature,
        )

        replacement_token, accepted, all_accepted = verify_proposals(
            model=target_model,
            context_ids=generated,
            proposal_ids=proposal_ids,
            device=target_device,
            temperature=temperature,
        )

        accepted_tokens = proposal_ids[:, :accepted]
        if accepted > 0:
            generated = torch.cat([generated, accepted_tokens], dim=-1)
            accepted_mask = torch.ones((mask.shape[0], accepted), dtype=mask.dtype, device=draft_device)
            mask = torch.cat([mask, accepted_mask], dim=-1)
            total_accepted += accepted

        if not all_accepted and generated.shape[1] - original_length < max_new_tokens:
            replacement_on_draft = replacement_token.to(draft_device)
            generated = torch.cat([generated, replacement_on_draft], dim=-1)
            replacement_mask = torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=draft_device)
            mask = torch.cat([mask, replacement_mask], dim=-1)

    runtime = time.perf_counter() - start
    new_tokens = generated[:, original_length : original_length + max_new_tokens]
    acceptance_rate = total_accepted / max_new_tokens if max_new_tokens else math.nan
    accepted_per_step = total_accepted / steps if steps else math.nan

    return {
        "generated_text": tokenizer.decode(new_tokens[0], skip_special_tokens=True),
        "generated_tokens": max_new_tokens,
        "runtime_s": runtime,
        "acceptance_rate": acceptance_rate,
        "accepted_tokens_per_step": accepted_per_step,
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
    if "acceptance_rate" in runs[0]:
        acceptance_rate = statistics.mean(float(run["acceptance_rate"]) for run in runs)
        accepted_tokens_per_step = statistics.mean(float(run["accepted_tokens_per_step"]) for run in runs)

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
