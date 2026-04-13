# gollmgo - Benchmark Methodology

## Why This Exists
Benchmarking is a release gate, not marketing text. Every performance claim must be reproducible and comparable against strong baselines.

Primary competitor baseline: `vLLM`.
Secondary references: `TensorRT-LLM`, `llama.cpp` (when scenario-appropriate).

---

## Standard Benchmark Environment

### Primary hardware profile
- Machine: `NVIDIA DGX Spark`
- Chip: `GB10 Grace Blackwell superchip`
- Mode: single GPU for Phase 1-3 scorecards

### Software profile
- Pinned driver + CUDA toolkit versions
- gollmgo binary built from exact git SHA
- Baseline servers pinned to exact versions
- Dedicated host during runs (no co-tenancy)

Always include full environment metadata in outputs.

---

## Benchmark Types

### 1. Offline Throughput
All requests arrive immediately. Measures aggregate token throughput.

```bash
make bench BENCH=throughput MODEL=/models/Meta-Llama-3-8B-Instruct DATASET=sharegpt NUM_PROMPTS=1000
```

Report:
- output tokens/sec
- total tokens/sec
- completed requests/sec
- GPU memory use and utilization

### 2. Online Serving
Poisson arrivals at fixed QPS. Measures user-facing latency behavior.

```bash
make bench BENCH=serving MODEL=/models/Meta-Llama-3-8B-Instruct DATASET=sharegpt QPS=10 DURATION=300s
```

Report:
- TTFT P50/P95/P99
- ITL P50/P95/P99
- E2E latency P50/P95/P99
- timeout/error rate
- queue depth and KV utilization

### 3. Single-Request Latency
Focuses on cold/warm TTFT and decode speed for interactive UX.

```bash
make bench BENCH=latency MODEL=/models/Meta-Llama-3-8B-Instruct INPUT_TOKENS=128 OUTPUT_TOKENS=128
```

### 4. Memory Efficiency
Measures sequence capacity and fragmentation under realistic lengths.

```bash
make bench BENCH=memory MODEL=/models/Meta-Llama-3-8B-Instruct
```

---

## Baseline Comparison Protocol

Use identical model weights, tokenizer, sampling params, and request mix.

```bash
./bench/compare.sh --against vllm --model /models/Meta-Llama-3-8B-Instruct --dataset sharegpt
./bench/compare.sh --against trtllm --model /models/Meta-Llama-3-8B-Instruct --dataset sharegpt
```

Rules:
- Fresh server process per run group.
- Warmup phase excluded from measured window.
- At least 3 repetitions; report median and spread.

---

## Exit Targets

### Phase 1
- Throughput >= 60% of vLLM baseline
- Stable serving without request-loss under sustained load

### Phase 2
- Throughput >= 100% of vLLM on at least one primary workload
- TTFT P50 <= 15ms (128-token prompt)
- ITL P50 <= 10ms on interactive profile

### Phase 3
- No performance regressions over release branch baselines
- P99 latency behavior remains stable at target load

---

## Regression Gate Policy
A change is a regression if any of the following hold against baseline:
- Throughput drops by > 5%
- TTFT P99 increases by > 20%
- Error rate increases materially

Regressions block merge unless explicitly waived with rationale and rollback plan.

---

## Result Storage Format
Store run outputs in `bench/results/` as JSON:

```json
{
  "timestamp": "2026-04-13T00:00:00Z",
  "git_sha": "<sha>",
  "hardware": "DGX Spark GB10",
  "model": "Meta-Llama-3-8B-Instruct",
  "benchmark": "serving",
  "config": {
    "max_batch_size": 128,
    "block_size": 16
  },
  "results": {
    "throughput_tok_s": 0,
    "ttft_p50_ms": 0,
    "ttft_p99_ms": 0,
    "itl_p50_ms": 0,
    "error_rate": 0
  }
}
```

---

## Reading Metrics Quickly
- Throughput: cluster and batch-job quality indicator.
- TTFT: interactive responsiveness indicator.
- ITL: streaming smoothness indicator.
- KV utilization: memory-efficiency indicator.
- Queue depth trend: admission/scheduler health indicator.
