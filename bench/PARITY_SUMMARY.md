# Parity Summary — gollmgo vs vLLM (TinyLlama-1.1B-Chat-v1.0)

## Environment
- Hardware: DGX Spark GB10 (NVIDIA GB10 Grace Blackwell, 121.6 GB HBM)
- Git SHA: `f1d3c89`
- Date: 2026-04-14 (UTC)
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, dtype bfloat16
- Server config: `bench/baseline_config.json`
  (max_batch_size=64, token_budget=4096, prefill_chunk=512, FCFS, recompute preempt)
- Workload: prompt_len=128, output_len=128, concurrency=4, closed-loop, warmup=10, temperature=0
- Prompt generation: `tokenizer_exact`; token count: `tokenizer_completion`
- Sample counts: **gollmgo = 3 reps × 1000 prompts (3000 measurements)**, **vLLM = 1 rep × 400 prompts**
  (asymmetric — vLLM run truncated to meet time budget; see Notes)
- vLLM container: `vllm/vllm-openai:latest`, `--gpu-memory-utilization 0.9`, `--max-model-len 2048`

## Results

| Metric | gollmgo | vLLM | gollmgo / vLLM | Epic 1 (≥0.6x TPS) | Epic 2 target | Status vs Epic 2 |
|---|---|---|---|---|---|---|
| tokens_per_second | 81.83 | 234.24 | **0.35x** | FAIL | ≥ 1.0x vLLM | **FAIL** |
| requests_per_second | 1.770 | 1.856 | 0.95x | — | — | — |
| TTFT p50 (ms) | 113.748 | 40.772 | 2.79x (worse) | — | ≤ 15 ms | **FAIL** (7.6x over) |
| TTFT p95 (ms) | 141.975 | 48.043 | 2.96x (worse) | — | — | — |
| TTFT p99 (ms) | 148.886 | 51.276 | 2.90x (worse) | — | — | — |
| ITL p50 (ms) | 22.847 | 16.312 | 1.40x (worse) | — | ≤ 10 ms | **FAIL** (2.3x over) |
| ITL p95 (ms) | 113.663 | 25.191 | 4.51x (worse) | — | stable tail | **FAIL** |
| ITL p99 (ms) | 678.209 | 29.917 | **22.7x (worse)** | — | stable tail | **FAIL — severe** |
| E2E p50 (ms) | 2262.079 | 2193.044 | 1.03x | — | — | — |
| E2E p95 (ms) | 2314.629 | 2382.652 | 0.97x | — | — | — |
| E2E p99 (ms) | 2333.041 | 2465.031 | 0.95x | — | — | — |
| Error rate | 0 / 3000 | 0 / 400 | — | — | 0 errors | **PASS** |

## Exit Gate Status

**Epic 1 (Foundation)**
- [ ] Throughput ≥ 0.6x vLLM — **FAIL** (measured 0.35x on TinyLlama TPS)
- [x] End-to-end serving is stable (no errors in 3000 requests)
- [x] Reproducible benchmark output with config + metadata

**Epic 2 (Parity+)**
- [ ] ≥ 1.0x vLLM throughput — **FAIL** (0.35x)
- [ ] TTFT p50 ≤ 15 ms — **FAIL** (113.7 ms measured; 7.6x over target)
- [ ] ITL p50 ≤ 10 ms — **FAIL** (22.8 ms; 2.3x over target)
- [ ] Stable tail under sustained load — **FAIL** (ITL p99 = 678 ms = 22.7x vLLM)
- [x] Regression automation in place (`bench/compare.sh`, `check_regression.sh`, CI thresholds)

**Net:** Neither Epic 1 nor Epic 2 exit gates are met. The scheduler/kernels
orchestration is correct (no errors, tokenizer-exact accounting, reproducible
harness), but throughput and tail latency are materially behind vLLM on
identical workload.

## Largest Measured Gaps (priority order)

1. **ITL p99 tail (22.7x vLLM)** — 678 ms vs 30 ms. Decode-path stalls,
   likely scheduler/KV contention or per-step launch overhead without CUDA
   graph replay. This is the single worst number and the most user-visible
   regression on interactive streaming.
2. **Steady-state throughput (0.35x vLLM)** — 82 vs 234 tok/s at
   concurrency=4. Consistent with ITL gap: we are paying per-token
   overhead vLLM avoids.
3. **TTFT p50 (2.79x vLLM)** — 114 ms vs 41 ms. 128-token prompt should
   land in single-digit milliseconds on GB10; current number implies
   prefill is unbatched or not graph-captured, or first-token emission is
   synchronous with a long critical path.
4. **ITL p95 (4.5x vLLM)** — 114 ms vs 25 ms. Same root cause as p99 but
   hit earlier in the distribution, suggesting the stall is not a rare
   outlier.
5. **Quality parity** — not measured in this lane. No task/eval loop
   today; throughput wins are meaningless without an output-quality
   guardrail.

## Notes / Caveats

- Asymmetric sample sizes: gollmgo was driven with the canonical
  `bench/baseline_config.json` (3 reps × 1000 prompts). vLLM was run at
  1 rep × 400 prompts to fit the wall-clock budget. This is enough
  signal for the 3–4x gap seen here (noise floor is well below the gap),
  but before treating this summary as a release-gate number, re-run vLLM
  at the full 3×1000 canonical config (`bench/compare.sh --against vllm
  --model …`).
- E2E percentiles are near parity only because `output_len=128` caps
  per-request work; under open-ended lengths the TPS gap translates into
  a proportional E2E gap. Do not read E2E alone as parity.
- vLLM rep 1 from the aborted 3-rep run (255 tok/s) was ~9% higher than
  the 400-prompt number (234 tok/s), consistent with amortising startup
  jitter over more samples. The gollmgo/vLLM ratio does not shift
  materially with more vLLM samples.

## Reproduction

```bash
# Canonical (slow, ~25 min wall clock):
bash bench/compare.sh --against vllm \
  --model /path/to/TinyLlama-1.1B-Chat-v1.0 \
  --config bench/baseline_config.json

# Time-boxed (what produced this summary, ~4 min vLLM side):
# gollmgo: 3-rep 1000-prompt canonical run
# vLLM: 1-rep 400-prompt run against the same warm container
```

Fresh artifacts:
- `bench/results/gollmgo_tinyllama.json`
- `bench/results/vllm_tinyllama.json`
