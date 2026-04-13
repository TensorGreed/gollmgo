# Epic 01 - Fast-Track Execution Plan

## Purpose
This is the agent-friendly execution board for Phase 1. It is intentionally not week-based.

Use it as a capability-gated plan:
- pick the highest unblocked milestone,
- keep the patch scoped to that milestone or one sub-slice,
- finish with tests, docs, and a runnable verification step.

This plan is meant to compress well when you want to fast-track work.

---

## Board
- [x] M0: Repo spine
- [x] M1: Interface spine and mocks
- [x] M2: Operator surface thin slice
- [x] M3: CUDA backend bring-up
- [x] M4: Model and tokenizer layer
- [ ] M5: Correct eager inference
- [ ] M6: KV cache and paged attention
- [ ] M7: Continuous batching and Phase 1 exit gate

---

## Milestones

### M0: Repo spine
Maps to:
- E01-S01
- E01-S09 (skeleton only)

Outputs:
- module layout
- `Makefile` targets
- CI for lint and CPU-only tests
- config skeleton and example file
- logging, health, and metrics package skeletons

Verification:
- `make lint`
- `make test`

Unblocks:
- all later milestones

### M1: Interface spine and mocks
Maps to:
- E01-S02 (interface portion)
- E01-S07 (state model shell)
- E01-S08 (handler contracts)

Outputs:
- `Runner`, `Engine`, `Scheduler`, `Tokenizer`, and model-loader interfaces
- mock runner and mock engine for tests
- sequence state model and internal request/response contracts

Verification:
- unit tests for state transitions and mocked request flow

Unblocks:
- M2
- M3
- M4

### M2: Operator surface thin slice
Maps to:
- E01-S08
- E01-S09

Outputs:
- bootable HTTP server
- `/v1/chat/completions` request validation
- SSE and non-streaming response shells
- `/v1/models`, `/health/live`, `/health/ready`, `/metrics`
- config loading and startup validation

Recommended implementation note:
- use mocks first so the user-facing surface exists before full GPU inference lands

Verification:
- integration test with mocked token stream
- `curl` smoke checks for health and models endpoints

Unblocks:
- operator testing while backend work continues

### M3: CUDA backend bring-up
Maps to:
- E01-S02

Outputs:
- CGo bridge
- minimal backend handle lifecycle
- warmup/smoke path on DGX Spark
- GPU-tagged integration test

Verification:
- `make build`
- GPU smoke test for create, warmup, and destroy

Unblocks:
- M5

### M4: Model and tokenizer layer
Maps to:
- E01-S03
- E01-S09

Outputs:
- safetensors loader
- GGUF loader path
- tokenizer selection and round-trip tests
- normalized model metadata for engine use

Verification:
- fixture-based model metadata load tests
- tokenization parity tests on known strings

Unblocks:
- M5

### M5: Correct eager inference
Maps to:
- E01-S04

Outputs:
- baseline FP16/BF16 eager forward pass
- deterministic sampling controls
- correctness tests against trusted references

Verification:
- fixed-seed prompt tests with stable top-k agreement
- single-sequence end-to-end local inference

Depends on:
- M3
- M4

Unblocks:
- M6

### M6: KV cache and paged attention
Maps to:
- E01-S05
- E01-S06

Outputs:
- block pool and block tables
- allocate/free/refcount rules
- paged attention kernel integration
- cache metrics and correctness tests

Verification:
- block allocation tests
- paged-vs-naive attention correctness tests
- baseline Nsight profile captured

Depends on:
- M5

Unblocks:
- M7

### M7: Continuous batching and Phase 1 exit gate
Maps to:
- E01-S07
- E01-S08
- E01-S10

Outputs:
- FCFS continuous batching scheduler
- end-to-end API path on real inference
- offline and serving benchmarks
- pinned baseline comparison against vLLM

Verification:
- concurrent integration tests with mixed request sizes
- benchmark artifacts stored with config and hardware metadata

Depends on:
- M2
- M6

Exit gate:
- stable end-to-end serving
- reproducible benchmark output
- at least 60 percent of vLLM baseline on identical workload

---

## Parallel Work Guidance
After M1, these can move in parallel:
- M2 operator surface
- M3 backend bring-up
- M4 model and tokenizer layer

After M5 lands, keep M6 tightly scoped to cache correctness first, then kernel performance.

M7 should only absorb pieces that are already correct in isolation. Do not use scheduler integration to discover basic kernel or model-loader bugs.

---

## Agent Rules
- Update this board as milestones move.
- Prefer vertical slices that leave something runnable behind.
- When touching hot paths, attach benchmark notes in the same change.
- When blocked, move to another unblocked lane instead of widening scope.
