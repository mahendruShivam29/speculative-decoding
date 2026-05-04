# Requirements for "Implement Speculative Decoding"

## Purpose of the assignment

This assignment is asking for a working, reproducible implementation of speculative decoding across two GPUs.

The goal is not only to write code that runs, but also to show that you can:

- understand how speculative decoding works,
- use two GPUs at the same time,
- measure performance carefully,
- explain bottlenecks and engineering tradeoffs clearly.

The expected level is strong. The assignment is meant for someone with machine learning, systems, or high-performance computing experience.

## Main idea in simple English

The task compares two ways of generating text with a large language model:

1. Normal greedy decoding
2. Speculative decoding

In normal greedy decoding, the large target model generates one token at a time.

In speculative decoding, a smaller draft model first guesses a few future tokens quickly. Then the larger target model checks those guesses. If the guesses are correct, several tokens can be accepted at once, which can make generation faster.

The assignment wants this split across two GPUs:

- GPU 0 runs the draft model
- GPU 1 runs the target model

## Tools and framework requirements

The implementation must use:

- Hugging Face Transformers
- PyTorch

This means the solution should rely on those libraries for model loading, tokenization, inference, tensor movement, and generation logic.

## Model requirements

The assignment suggests these models:

- Target model: `Qwen3-235b`
- Draft model: `Qwen3-4b`

But it also allows using another similar pair if the suggested models do not fit on the available GPUs.

Important detail:

- The instructions say to ask `algos@runara.ai` to confirm the model choice.

So the practical requirement is:

- use the suggested Qwen pair if possible,
- otherwise use a comparable large target model and smaller draft model that fit the hardware,
- and treat model confirmation as part of the expected setup process.

## Hardware requirements

The implementation must use two GPUs explicitly:

- `cuda:0` for the draft model
- `cuda:1` for the target model

This is not optional. The assignment specifically wants multi-GPU engineering, not a single-GPU or CPU-only simulation.

The code must correctly move models and tensors between devices. It is not enough to load both models and hope PyTorch handles placement automatically.

The assignment even gives the expected placement style:

```python
draft.to("cuda:0")
target.to("cuda:1")
```

So the implementation must deliberately manage:

- model placement,
- input tensor placement,
- movement of proposed tokens or sequences from GPU 0 to GPU 1,
- any needed synchronization or transfer logic.

## Part 1: Baseline greedy decoding

The first required feature is a baseline implementation of standard autoregressive decoding using only the target model.

That means:

- the draft model is not used here,
- the target model generates tokens one by one,
- generation should be greedy, meaning each next token is chosen directly rather than sampled.

The assignment requires running this baseline for:

- 10 prompts
- 100 generated tokens per prompt

For this baseline, the code must measure:

- tokens per second,
- latency per token,
- total runtime.

This baseline is important because the speculative method will later be compared against it.

## Part 2: Speculative decoding

The second required feature is a simplified speculative decoding implementation.

The requested algorithm is:

1. The draft model on GPU 0 generates `k=4` proposed tokens.
2. The proposed sequence is sent to GPU 1.
3. The target model verifies the proposed tokens one by one.
4. The matching prefix is accepted.
5. The first mismatched token is rejected and replaced with the target model’s token.
6. The process repeats until 100 output tokens have been generated.

In simpler terms:

- the small model guesses several next tokens,
- the big model checks whether those guesses match what it would have produced,
- all correct guesses in a row are kept,
- the first wrong guess is replaced by the target model’s answer,
- then generation continues from the updated sequence.

This is called a simplified version, so the assignment does not ask for every optimization or every detail from the research paper. It asks for the core mechanism to be implemented correctly and clearly.

## Output length requirement

Both decoding methods must continue until:

- 100 output tokens are generated.

This should be treated as a hard stopping condition for the benchmark.

The implementation should make sure the counting logic is correct, especially in speculative decoding where multiple tokens may be accepted in one iteration.

## Part 3: Multi-GPU engineering expectations

This part focuses on correct engineering, not just algorithm logic.

The code must show that:

- the draft model really runs on GPU 0,
- the target model really runs on GPU 1,
- tensors are moved to the correct device before computation,
- proposed tokens or token sequences are transferred correctly from one GPU to the other,
- the implementation does not accidentally fall back to one device for everything.

This likely means the code should be written carefully enough that another engineer can inspect it and clearly see where:

- prompt tokens start,
- draft generation happens,
- verification happens,
- cross-device data transfer happens.

## Part 4: Benchmarking requirements

The assignment requires performance comparison between:

- Greedy decoding
- Speculative decoding

The results should be reported in a comparison table with at least:

- method,
- tokens per second,
- average latency,
- speedup.

The greedy baseline should be treated as the reference with:

- speedup = `1.0x`

The speculative implementation should report its relative speedup compared with greedy decoding.

The benchmark must be run for multiple speculative chunk sizes:

- `k=2`
- `k=4`
- `k=8`

This means the code should not hardcode speculative decoding only for `k=4`. Even though the algorithm description uses `k=4` as the example flow, the benchmarking requirement makes it clear that the implementation should support at least these three values of `k`.

## What should be measured and reported

Across the assignment, the expected measurable outputs are:

- total runtime for greedy decoding,
- total runtime for speculative decoding,
- tokens per second for each method,
- average latency per token for each method,
- speedup of speculative decoding relative to greedy decoding.

A good interpretation is that results should be stable, clearly computed, and easy to reproduce.

## Deliverables

The assignment explicitly requires two files.

### 1. `main.py`

This must be a runnable script.

The instructions specifically say it should run as:

```bash
python main.py
```

So `main.py` should:

- load the models,
- run baseline greedy decoding,
- run speculative decoding,
- collect benchmark metrics,
- produce output that shows the results.

### 2. `README.md`

This should be a short explanation covering:

- design decisions,
- performance results,
- bottlenecks found,
- what would be optimized next.

This means the README is not just setup text. It should explain the engineering choices and what was learned from the benchmark.

## What a complete solution should contain

A complete solution should likely include all of the following:

- code to load tokenizer and models,
- explicit device placement for draft and target models,
- baseline greedy decoding logic,
- speculative decoding logic,
- prompt handling for 10 prompts,
- generation stopping at 100 tokens,
- timing and metric collection,
- benchmark comparison for `k=2`, `k=4`, and `k=8`,
- readable output from `main.py`,
- a short but informative `README.md`.

## Hidden expectations implied by the assignment

Even though not everything is written as a strict rule, the assignment strongly implies these expectations:

- the code should be clean and readable,
- the results should be reproducible,
- the implementation should be correct before it is optimized,
- performance discussion should include bottlenecks, not just raw numbers,
- the candidate should understand why speculative decoding may or may not speed things up in practice,
- the candidate should be able to explain cross-GPU overheads such as data transfer and synchronization.

## Likely evaluation criteria

Based on the assignment text, the work will probably be judged on:

- correctness of the greedy baseline,
- correctness of the speculative decoding logic,
- proper use of two GPUs,
- quality of metric collection,
- clarity of performance analysis,
- code cleanliness and reproducibility,
- quality of explanation in the README.

## Final plain-English summary

In simple terms, the assignment wants you to build a small experiment that proves you can do four things well:

1. Run normal text generation with a large model.
2. Build speculative decoding with a smaller helper model plus a larger verifier model.
3. Split the work correctly across two GPUs.
4. Measure whether the speculative method is actually faster, and explain why.

The final result should be a runnable `main.py` script and a short `README.md` that explains the implementation, the benchmark numbers, the bottlenecks, and the next optimization ideas.
