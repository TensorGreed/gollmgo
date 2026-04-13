# Epic 02 - Performance Parity+ on Single GPU

**Phase:** 2  
**Target:** Q2-Q3 2026  
**Prerequisite:** Epic 01 complete  
**Status:** Planned

See also:
- `epics/epic-02-execution-plan.md` for the fast-track, capability-gated implementation order used by coding agents.

---

## Goal
Reach or exceed vLLM performance on target single-GPU workloads while preserving correctness, stability, and operator control.

---

## Stories

### E02-S01: PagedAttention v2
**Points:** 8

Deliverables:
- long-context optimized kernel path
- two-pass reduction for stable softmax accumulation
- correctness and profiling artifacts

Acceptance:
- Throughput improvement on long-context workloads vs v1.

### E02-S02: Chunked prefill scheduler path
**Points:** 5

Deliverables:
- per-tick prefill budgeting
- mixed prefill/decode batch assembly
- starvation-avoidance tests

Acceptance:
- Tail latency improves under mixed short/long prompts.

### E02-S03: CUDA graph capture and replay
**Points:** 13

Deliverables:
- graph warmup for selected batch shapes
- replay policy with fallback to eager path
- instrumentation for graph hit/miss rates

Acceptance:
- Reduced launch overhead and measurable TTFT improvement.

### E02-S04: Prefix cache
**Points:** 8

Deliverables:
- block hash strategy
- LRU eviction policy
- prefix hit-rate metrics

Acceptance:
- System-prompt-heavy traffic shows material prefill savings.

### E02-S05: Quantized serving paths
**Points:** 8

Deliverables:
- backend-gated FP8 serving path
- backend-gated INT8 serving path
- quality regression checks by task profile

Acceptance:
- Memory reduction and throughput gains without unacceptable quality drift.

### E02-S06: Speculative decoding
**Points:** 8

Deliverables:
- n-gram mode
- optional draft-model mode
- acceptance accounting and rollback-safe toggles

Acceptance:
- Positive throughput gains on supported workloads.

### E02-S07: Preemption and policy upgrades
**Points:** 5

Deliverables:
- policy framework (FCFS/SJF/priority)
- recompute and optional swap preemption
- fairness and starvation tests

Acceptance:
- Improved P95/P99 behavior under overload scenarios.

### E02-S08: Benchmark and regression automation
**Points:** 5

Deliverables:
- CI benchmark job definitions
- perf-baseline files and regression thresholds
- release summary template with deltas

Acceptance:
- Hot-path regressions are detected before release.

---

## Exit Targets
- Throughput: `>= 1.0x` vLLM on primary serving benchmark.
- TTFT P50: `<= 15ms` at target prompt length.
- ITL P50: `<= 10ms` on interactive profile.
- Prefix cache hit rate: high enough to justify feature cost on targeted workloads.

---

## Risks
- Overfitting to synthetic benchmarks instead of real traffic distributions.
- Quality regressions from aggressive quantization/speculative paths.
- Complexity creep in scheduler policy interactions.
