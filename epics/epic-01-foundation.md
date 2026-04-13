# Epic 01 - Foundation: Single-GPU Serving Core

**Phase:** 1  
**Target:** Q2 2026  
**Platform:** NVIDIA DGX Spark (GB10 Grace Blackwell superchip)  
**Status:** Planned

See also:
- `epics/epic-01-execution-plan.md` for the fast-track, capability-gated implementation order used by coding agents.

---

## Goal
Ship a reliable single-GPU inference server that can serve LLaMA-family models through an OpenAI-compatible API with correct continuous batching and paged KV cache.

---

## Stories

### E01-S01: Project bootstrap and CI
**Points:** 3

Deliverables:
- Go module layout + Make targets
- CI for lint + unit test
- baseline docs and contributor workflow

Acceptance:
- Fresh checkout builds and runs unit tests without GPU.

### E01-S02: CUDA backend scaffold
**Points:** 5

Deliverables:
- CGo bridge + minimal C API
- backend runner lifecycle (create/warmup/step/close)
- mock runner interface for tests

Acceptance:
- Backend health check passes and warmup executes on DGX Spark.

### E01-S03: Model loading layer
**Points:** 8

Deliverables:
- safetensors loading path
- GGUF loading path
- model metadata normalization for engine use

Acceptance:
- Supported model loads and reports consistent config.

### E01-S04: Correct eager forward pass
**Points:** 13

Deliverables:
- baseline FP16/BF16 forward path
- correctness tests against known references
- deterministic sampling controls

Acceptance:
- Stable top-k agreement on fixed prompts and seeds.

### E01-S05: KV cache manager v1
**Points:** 8

Deliverables:
- block pool, block table, allocate/free
- sequence ownership/refcount tracking
- cache utilization metrics

Acceptance:
- No fragmentation pathologies in long-running stress tests.

### E01-S06: PagedAttention v1 integration
**Points:** 13

Deliverables:
- kernel implementation + tests
- integration into forward path
- baseline Nsight profile

Acceptance:
- Correctness validated and kernel path active in serving loop.

### E01-S07: Scheduler v1
**Points:** 8

Deliverables:
- FCFS admission
- continuous batching loop
- decode progression and EOS completion logic

Acceptance:
- Concurrent request set completes without deadlocks or starvation.

### E01-S08: OpenAI-compatible API v1
**Points:** 8

Deliverables:
- `/v1/chat/completions` (streaming and non-streaming)
- `/v1/models`
- request validation + structured errors

Acceptance:
- OpenAI SDK can call gollmgo via `base_url` override.

### E01-S09: Config, health, metrics
**Points:** 5

Deliverables:
- config loading + validation
- health/readiness endpoints
- Prometheus metrics for queue, TTFT, throughput, KV usage

Acceptance:
- Operators can validate health and observe core performance signals.

### E01-S10: Benchmark harness baseline
**Points:** 5

Deliverables:
- offline and serving benchmark modes
- pinned benchmark config for DGX Spark
- baseline comparison script for vLLM

Acceptance:
- Benchmark output is reproducible and stored with metadata.

---

## Exit Criteria
- End-to-end serving is stable for 24h continuous run.
- Throughput reaches at least 60% of vLLM baseline on identical workload.
- All stories include tests and minimal operator docs.

---

## Dependencies
- E01-S02 blocks backend-dependent engine work.
- E01-S03 + E01-S04 block scheduler integration tests.
- E01-S05 + E01-S06 block performant serving paths.
- E01-S07 + E01-S08 + E01-S09 block production-like validation.

---

## Risks
- CGo/toolchain setup complexity on first pass.
- Incorrect KV ownership under concurrent cancellation.
- Benchmark noise without strict environment pinning.
