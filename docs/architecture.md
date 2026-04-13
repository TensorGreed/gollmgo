# gollmgo - System Architecture

## Objectives
- Match or beat vLLM on real serving workloads.
- Keep operations simple: one binary, one config, clear health/metrics surfaces.
- Optimize first for `NVIDIA DGX Spark (GB10 Grace Blackwell superchip)`.
- Preserve clean backend abstraction for future ROCm support.

---

## Runtime Topology (Phase 1-3)

```text
Client
  -> OpenAI API Server
    -> Scheduler (single owner goroutine)
      -> Engine Orchestrator
        -> KV Cache Manager
        -> Backend Runner (CUDA now, ROCm later)
          -> Attention + GEMM + Sampling kernels
```

### Ownership rules
- API owns request lifecycle and cancellation.
- Scheduler owns sequence state transitions.
- KV cache owns GPU memory for keys/values.
- Backend runner owns kernel dispatch and stream/graph execution.

No component bypasses these ownership boundaries.

---

## Component Responsibilities

### API layer (`internal/api/`)
- Implements OpenAI-compatible endpoints.
- Converts request payloads into internal sequences.
- Streams tokens with per-request cancellation support.
- Exposes health, readiness, metrics, and admin control endpoints.

### Scheduler (`internal/scheduler/`)
- Performs iteration-level continuous batching.
- Admits requests based on queue policy and KV pressure.
- Balances prefill vs decode using configurable token budgets.
- Handles preemption and re-admission deterministically.

### KV cache (`internal/kvcache/`)
- Paged block pool with block tables per sequence.
- Prefix cache with block-level hash and LRU eviction.
- Alloc/free and refcount semantics that are safe under preemption.

### Engine (`internal/engine/`)
- Builds flattened batches for backend execution.
- Manages graph replay/eager fallback policy.
- Runs logits post-processing and sampling.

### Backend (`internal/backend/`)
Interface target:
```go
type Runner interface {
    Warmup(ctx context.Context, profile WarmupProfile) error
    Step(ctx context.Context, batch *Batch) (*StepOutput, error)
    Capabilities() Capabilities
    Close() error
}
```

Initial implementation: `cuda` backend via CGo.
Future implementation: `rocm` backend using same interface contract.

---

## Scheduler Model

Sequence states:
`WAITING -> PREFILLING -> DECODING -> FINISHED`
With preemption path:
`DECODING -> PREEMPTED -> WAITING`

Per tick:
1. Collect active decode set.
2. Admit new sequences if KV and batch budgets allow.
3. Reserve chunked prefill budget.
4. Build one mixed batch (prefill + decode).
5. Run one backend step.
6. Emit tokens, update state, and release resources.

Invariants:
- Tick path is non-blocking and free of unbounded I/O.
- Block-table correctness is validated before kernel launch.
- Batch size and token-budget limits are enforced every tick.

---

## Memory Model

### Key idea
Inference is mostly memory-bandwidth bound; cache design is product-critical.

### KV layout
- Fixed-size blocks (configurable `block_size`).
- Logical sequence blocks map to physical blocks via block table.
- Prefix sharing is block-level and refcounted.

### Budgeting strategy
- Reserve runtime headroom for model weights, activations, and CUDA context.
- Allocate KV pool from remaining memory by policy (`max_memory_fraction`).
- Expose utilization/fragmentation metrics continuously.

---

## Kernel Strategy

Backends by capability and workload:
- PagedAttention v1: correctness baseline and fallback.
- PagedAttention v2: long-context and higher throughput path.
- Vendor-optimized attention path (when available and validated).

Kernel requirements:
- Verified against reference outputs.
- Profiled with Nsight Compute.
- Integrated through minimal C API surface.

---

## Reliability and Operations

### Required production surfaces
- `/health/live`, `/health/ready`, `/metrics`
- Structured logs with request ID and sequence ID
- Request cancellation propagation to scheduler/runner
- Graceful shutdown with drain timeout

### Failure handling
- Backpressure when queue depth exceeds policy.
- Explicit error classes for OOM, invalid request, backend failure.
- Optional admission throttling when TTFT SLO degrades.

---

## Portability Plan: CUDA to ROCm

Architecture rule: scheduler/API/model-loading code must be backend-agnostic.

Portability steps:
1. Lock backend interface and shared tensor/launch contracts.
2. Keep CUDA-specific logic isolated under `internal/backend/cuda` + `kernels/`.
3. Introduce ROCm backend under `internal/backend/rocm` with parity tests.
4. Run vendor-comparison benchmarks before feature graduation.

---

## Scaling Plan
- Phase 1-3: single GPU excellence.
- Phase 4: single-node multi-GPU (tensor parallel + admission control).
- Phase 5: cluster mode (disaggregated prefill/decode + KV handoff).

Architecture must evolve without breaking API compatibility or operator workflows.
