# Speculative Decoding Benchmark

This repository implements a simplified speculative decoding benchmark using Hugging Face Transformers and PyTorch. It is set up for a final Kaggle dual-GPU benchmark run, while still remaining usable on a single GPU for development.

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

- average fraction of proposed draft tokens accepted,
- average fraction of final output contributed by accepted draft tokens,
- average draft-model calls,
- average target-model calls,
- average proposal-token transfers,
- average replacement-token transfers,
- average speculative steps.

## Recommended model pair for Colab Plus

This project is set up for the following practical pair:

- draft model: `Qwen/Qwen2.5-0.5B-Instruct`
- target model: `Qwen/Qwen2.5-3B-Instruct`

This is intentionally smaller than the assignment's suggested `Qwen3-235b` / `Qwen3-4b` pair. The reason is hardware realism: a Colab T4 with 12 GB VRAM cannot run the originally suggested target model.

The design keeps the draft and target device assignments configurable, but the default benchmark configuration now targets a real 2-GPU machine:

- draft model on `cuda:0`
- target model on `cuda:1`

## Setup

Install dependencies:

```bash
pip install torch transformers accelerate sentencepiece
```

If you plan to use 4-bit quantization for development-only experiments, also install:

```bash
pip install bitsandbytes
```

If you are running in Kaggle, make sure the notebook has GPU enabled before launching the script.

## Usage

### Kaggle dual-GPU benchmark run

Run the draft and target models on separate GPUs with native `float16`:

```bash
python main.py \
  --draft-device cuda:0 \
  --target-device cuda:1 \
  --dtype float16
```

### Single-GPU development run

If you need a smaller local or Colab-style debugging run:

```bash
python main.py \
  --draft-device cuda:0 \
  --target-device cuda:0 \
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
- The final benchmark path avoids blocking CUDA synchronizations inside the hot loop so the runtime is closer to true end-to-end throughput.
- Host transfers are deferred until the end of each prompt, and mismatch detection is vectorized so speculative verification incurs only one host sync per step instead of one per token.

## Expected bottlenecks

On a single GPU, the likely bottlenecks are:

- target-model forward pass latency,
- repeated draft proposal work when acceptance is low,
- target and draft models competing for the same GPU,
- limited VRAM forcing quantization,
- lack of true cross-GPU parallelism.

On a real 2-GPU run such as Kaggle dual T4, the main remaining bottlenecks become:

- transfer of proposed tokens between devices,
- draft-model token-by-token generation cost,
- reduced speedup when draft-token acceptance is low,
- wasted draft work when `k` is too large.

## What to analyze in your writeup

Your final assignment writeup should explain:

- why speculative decoding can be faster than greedy decoding,
- when it fails to help,
- how acceptance rate changes with `k`,
- why `k=2`, `k=4`, and `k=8` show different tradeoffs,
- how much of the runtime is dominated by the target model.

For the current Qwen `0.5B -> 3B` dual-T4 setup, a good explanation is:

- speculative decoding reduces target-model calls substantially, which is the intended effect,
- but a batch-size-1 draft model is not proportionally as fast as its parameter count suggests,
- as `k` increases, draft errors compound and the proposed-token acceptance rate drops,
- once the cost of generating speculative draft tokens exceeds the target-side savings, larger `k` values become slower than greedy decoding.

## Final benchmark results

The final benchmark was run on a Kaggle dual-T4 environment with:

- draft model on `cuda:0`
- target model on `cuda:1`
- `float16`
- 10 prompts
- 100 generated tokens per prompt

Results:

| Method | tok/s | avg latency (ms/token) | speedup vs greedy |
| --- | ---: | ---: | ---: |
| Greedy | 22.34 | 44.76 | 1.00x |
| Speculative `k=2` | 18.99 | 52.66 | 0.85x |
| Speculative `k=4` | 16.66 | 60.01 | 0.75x |
| Speculative `k=8` | 11.74 | 85.16 | 0.53x |

Additional observations:

- Greedy target-model calls: `101` per prompt on average
- Speculative target-model calls:
  - `47.4` for `k=2`
  - `38.1` for `k=4`
  - `33.4` for `k=8`
- Proposed-token acceptance rate:
  - `0.583` for `k=2`
  - `0.440` for `k=4`
  - `0.284` for `k=8`

These numbers show that speculative decoding did reduce target-model calls substantially, which means the implementation is functioning correctly. However, the reduction in target work was not enough to offset the draft-model cost and inter-GPU overhead for this specific model pair and hardware setup.

## Latency math

The benchmark generated `1000` total tokens (`10 prompts * 100 tokens`).

### Target model latency

The greedy baseline took `44.76s` to generate `1000` tokens.

- Target latency per token:
  - `44.76 / 1000 = 0.04476s`
  - about `45 ms/token`

### Draft model latency estimate from `k=2`

The speculative `k=2` run took `52.66s`.

Average speculative steps per prompt:

- `46.4`

Across `10` prompts, that is:

- `464` total speculative steps

Average latency per speculative step:

- `52.66 / 464 = 0.1135s`
- about `113 ms/step`

In each `k=2` speculative step, the system performs:

- `1` target verification pass
- `2` draft passes
- token transfer between `cuda:0` and `cuda:1`

Using the greedy target latency as a rough estimate for the target pass:

- target contribution per step: about `45 ms`
- remaining draft + transfer overhead:
  - `113 ms - 45 ms = 68 ms`

Since `k=2` uses two draft passes per step, the rough draft-side cost is:

- `(68 ms) / 2 = 34 ms` per draft token

### Fatal ratio

This gives the following rough latency ratio:

- target model: `45 ms/token`
- draft model + transfer overhead: `34 ms/token`

So the target model is only about:

- `45 / 34 = 1.32x`

slower than the draft side.

This is the core reason speculative decoding failed to produce a speedup here. For speculative decoding to help substantially, the target model generally needs to be much slower than the draft model. In this run, the draft side was too expensive relative to the target side.

## Why the draft model is not much faster

Although the draft model has far fewer parameters (`0.5B` vs `3B`), batch-size-1 inference is not determined purely by parameter count.

At this scale, latency is dominated by:

- memory bandwidth,
- CUDA kernel launch overhead,
- framework overhead,
- and inter-GPU transfer/synchronization costs.

The `3B` model does move more weight data than the `0.5B` model, but the smaller model still pays much of the same fixed overhead for each forward pass. As a result, the draft model is not remotely `6x` faster in wall-clock latency, even though it has about `1/6` the parameter count.

This explains the final behavior:

- `k=2` is the best tradeoff for this setup,
- larger `k` values make acceptance rate worse,
- and the draft model spends too much time proposing tokens that are later rejected.

## Honest positioning for the assignment

For a Kaggle dual-GPU run, this can be described as:

- a true two-GPU speculative decoding benchmark with `cuda:0` and `cuda:1`,
- a hardware-constrained but assignment-compliant model pair,
- an honest evaluation of speedup versus greedy decoding on dual T4 GPUs.

For a single-GPU development run, this should still be described as:

- a correct small-scale implementation of the speculative decoding algorithm,
- a practical benchmark under student hardware constraints,
- preparation for a final same-script rerun on a proper 2-GPU environment.

It should not be described as equivalent to the exact assignment architecture unless the script is actually rerun with separate visible devices such as `cuda:0` and `cuda:1` in the same runtime.
