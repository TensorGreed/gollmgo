# Epic 02 - Fast-Track Execution Plan

## Purpose
This is the agent-friendly execution board for Phase 2. It is not week-based.

Use it to sequence performance work without losing the plot:
- establish a stable baseline first,
- improve one bottleneck at a time,
- keep quality and regression checks attached to every hot-path change.

---

## Board
- [ ] M0: Baseline freeze and instrumentation
- [ ] M1: Scheduler fairness and chunked prefill
- [ ] M2: Long-context attention path
- [ ] M3: CUDA graph capture and replay
- [ ] M4: Prefix cache
- [ ] M5: Quantized serving paths
- [ ] M6: Speculative decoding
- [ ] M7: Preemption and policy upgrades
- [ ] M8: Parity gate and regression automation

---

## Milestones

### M0: Baseline freeze and instrumentation
Maps to:
- E02-S08 (foundation)

Outputs:
- pinned Phase 1 baseline numbers
- benchmark config freeze for parity runs
- metric coverage for TTFT, ITL, queue depth, cache hit rate, graph hit rate

Verification:
- repeatable benchmark runs with stable metadata

Unblocks:
- all later milestones

### M1: Scheduler fairness and chunked prefill
Maps to:
- E02-S02

Outputs:
- per-tick prefill budgeting
- mixed prefill/decode assembly improvements
- starvation tests on mixed short/long prompts

Verification:
- serving benchmark shows improved tail latency under mixed load

Unblocks:
- M7
- parity-quality reads for later optimizations

### M2: Long-context attention path
Maps to:
- E02-S01

Outputs:
- paged attention v2 or equivalent long-context kernel path
- correctness tests against v1/naive reference
- Nsight profile for long-context behavior

Verification:
- long-context throughput improves without correctness drift

Unblocks:
- stronger parity numbers on long prompts

### M3: CUDA graph capture and replay
Maps to:
- E02-S03

Outputs:
- graph warmup for selected shapes
- replay path with eager fallback
- graph hit/miss instrumentation

Verification:
- measurable TTFT reduction on warm-path requests

Unblocks:
- parity gate on interactive latency

### M4: Prefix cache
Maps to:
- E02-S04

Outputs:
- block hashing and cache lookup path
- LRU eviction policy
- prefix hit/miss metrics

Verification:
- system-prompt-heavy workload shows meaningful prefill savings

Unblocks:
- production-ready high-reuse workloads

### M5: Quantized serving paths
Maps to:
- E02-S05

Outputs:
- backend-gated FP8 path
- backend-gated INT8 path
- quality checks on representative prompts/tasks

Verification:
- improved memory headroom and throughput with acceptable output quality

Depends on:
- M0

### M6: Speculative decoding
Maps to:
- E02-S06

Outputs:
- n-gram path first
- draft-model mode second if warranted
- acceptance-rate accounting and kill switch config

Verification:
- throughput uplift on supported workloads with stable outputs

Depends on:
- M0

### M7: Preemption and policy upgrades
Maps to:
- E02-S07

Outputs:
- policy framework for FCFS, SJF, and priority
- recompute and optional swap preemption
- fairness and starvation test coverage

Verification:
- overload benchmark shows improved P95/P99 latency behavior

Depends on:
- M1

### M8: Parity gate and regression automation
Maps to:
- E02-S08

Outputs:
- CI benchmark job definitions
- baseline files and thresholds
- release-ready parity summary format

Verification:
- automated runs catch regressions before merge

Depends on:
- M0
- M1
- M2
- M3

Exit gate:
- at least parity with vLLM on primary workload
- TTFT and ITL targets met on pinned interactive profile
- regression automation in place

---

## Parallel Work Guidance
After M0:
- M1 and M2 can move in parallel.
- M3 and M4 can move in parallel once scheduler behavior is stable enough to interpret results.
- M5 and M6 should be isolated behind config flags until parity data is clean.

Do not stack multiple speculative optimizations into one benchmark comparison. One performance story per change is easier to trust.

---

## Agent Rules
- Always attach before/after benchmark notes for hot-path work.
- Prefer feature flags for risky optimizations.
- Treat output-quality regressions as blockers, not footnotes.
