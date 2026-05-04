# Speculative Decoding Benchmark

This repository implements a simplified speculative decoding benchmark using Hugging Face Transformers and PyTorch. It is designed for a Colab Plus development workflow first, with a later option to rerun the same script on a true 2-GPU machine for the final assignment benchmark.

## What this script does

`main.py` benchmarks:

- greedy decoding with the target model only,
- speculative decoding with a small draft model and a larger target model,
- speculative chunk sizes `k=2`, `k=4`, and `k=8`.

For each method it reports:

- total runtime,
- tokens per second,
- average latency per token,
- speedup over greedy decoding.

For speculative decoding it also reports:

- average acceptance rate,
- average number of accepted draft tokens per speculative step.

It also prints extra instrumentation to help analyze why a run sped up or slowed down:

- average draft-model calls,
- average target-model calls,
- average draft, verification, and commit time,
- average proposal-token transfers,
- average replacement-token transfers,
- average speculative steps.

## Recommended model pair for Colab Plus

This project is set up for the following practical pair:

- draft model: `Qwen/Qwen2.5-0.5B-Instruct`
- target model: `Qwen/Qwen2.5-3B-Instruct`

This is intentionally smaller than the assignment's suggested `Qwen3-235b` / `Qwen3-4b` pair. The reason is hardware realism: a Colab T4 with 12 GB VRAM cannot run the originally suggested target model.

The design keeps the draft and target device assignments configurable so the same script can be rerun later on a real 2-GPU machine.

## Setup

Install dependencies:

```bash
pip install torch transformers accelerate bitsandbytes sentencepiece
```

If you are running in Colab, make sure the runtime has GPU enabled before launching the script.

## Usage

### Colab Plus or single-GPU development run

Run both models on the same GPU with 4-bit quantization:

```bash
python main.py \
  --draft-device cuda:0 \
  --target-device cuda:0 \
  --quantize-4bit \
  --dtype float16
```

### Later 2-GPU validation run

When you have access to a true dual-GPU machine, split the models across devices:

```bash
python main.py \
  --draft-device cuda:0 \
  --target-device cuda:1 \
  --quantize-4bit \
  --dtype float16
```

### Custom prompts

You can pass a text file with one prompt per line or a JSON file containing a list of strings:

```bash
python main.py --prompts-file prompts.txt
```

## Design decisions

- The implementation uses one tokenizer, loaded from the target model family, because speculative decoding works best when draft and target share the same tokenization scheme.
- Greedy decoding is implemented with KV cache enabled so that, after prompt prefill, only one new token is processed per step.
- Speculative decoding keeps separate draft and target prefix state with KV cache enabled on both models.
  The draft model proposes `k` tokens incrementally from cached state, the target model verifies them in one cached forward pass per speculative step, and the accepted prefix is committed by truncating speculative KV state instead of replaying accepted tokens.
- The script keeps `draft_device` and `target_device` as explicit parameters so the same code path can be used for both single-GPU development and real two-GPU validation.
- The implementation transfers only proposal tokens to the target device and only the replacement token back to the draft device. It does not resend the full context each step.

## Expected bottlenecks

On Colab Plus with a single T4, the likely bottlenecks are:

- target-model forward pass latency,
- repeated draft proposal work when acceptance is low,
- remaining verification overhead on the target model,
- limited VRAM forcing quantization,
- lack of true cross-GPU parallelism.

On a real 2-GPU run, additional bottlenecks become relevant:

- transfer of proposed tokens between devices,
- synchronization between draft and target verification,
- reduced speedup when draft-token acceptance is low,
- wasted draft work when `k` is too large.

## What to analyze in your writeup

Your final assignment writeup should explain:

- why speculative decoding can be faster than greedy decoding,
- when it fails to help,
- how acceptance rate changes with `k`,
- why `k=2`, `k=4`, and `k=8` show different tradeoffs,
- how much of the runtime is dominated by the target model.

## Honest positioning for the assignment

For a Colab Plus development run, this should be described as:

- a correct small-scale implementation of the speculative decoding algorithm,
- a practical benchmark under student hardware constraints,
- preparation for a final same-script rerun on a proper 2-GPU environment.

It should not be described as equivalent to the exact assignment architecture unless the script is actually rerun with separate visible devices such as `cuda:0` and `cuda:1` in the same runtime.
