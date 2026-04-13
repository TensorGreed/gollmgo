# Architecture Decision Records

## Status Legend
- `Proposed`: not final, pending validation.
- `Accepted`: agreed and active.
- `Rejected`: evaluated and not adopted.
- `Superseded`: replaced by a newer ADR.

## Working Rule
This project is starting from scratch. Every ADR begins as `Proposed` and moves to `Accepted` only after:
1. technical spike or prototype,
2. benchmark or reliability evidence, and
3. explicit team sign-off.

---

## ADR-001: Go for control plane, backend runtime for GPU execution

**Status:** Proposed  
**Proposed Date:** 2026-04-13  
**Review Trigger:** End of Epic 01

### Context
We need high-concurrency serving behavior with predictable operational characteristics and explicit control over scheduler and lifecycle logic.

### Proposed Decision
Use Go for API/scheduler/orchestration and a backend runtime boundary for GPU execution (CUDA first).

### Consequences if accepted
- Strong concurrency model and operational simplicity.
- Requires careful CGo/backend boundary management.

---

## ADR-002: Paged KV cache as default memory strategy

**Status:** Proposed  
**Proposed Date:** 2026-04-13  
**Review Trigger:** After PagedAttention v1 benchmark validation

### Context
Static per-sequence KV allocation wastes memory and caps batch size.

### Proposed Decision
Adopt block-based paged KV cache with sequence block tables and prefix-sharing support.

### Consequences if accepted
- Better memory efficiency and higher concurrency.
- More complex kernel indexing and cache-management logic.

---

## ADR-003: OpenAI-compatible API as primary serving interface

**Status:** Proposed  
**Proposed Date:** 2026-04-13  
**Review Trigger:** After API MVP integration testing

### Context
Adoption speed depends on ecosystem compatibility.

### Proposed Decision
Use OpenAI-compatible endpoints as the primary external API.

### Consequences if accepted
- Easy client/tool integration.
- Protocol overhead and shape constraints vs custom binary APIs.

---

## ADR-004: Single-GPU-first execution on DGX Spark

**Status:** Proposed  
**Proposed Date:** 2026-04-13  
**Review Trigger:** End of Epic 01 and Epic 02 planning checkpoint

### Context
The available dev platform is NVIDIA DGX Spark (GB10). Early complexity must be controlled.

### Proposed Decision
Prioritize single-GPU correctness, throughput, and operational maturity before multi-GPU or cluster features.

### Consequences if accepted
- Faster path to a stable and benchmarkable core.
- Multi-node features are intentionally deferred.

---

## ADR-005: Performance claims require pinned baseline comparisons

**Status:** Proposed  
**Proposed Date:** 2026-04-13  
**Review Trigger:** First benchmark automation rollout

### Context
Optimization work is high risk without strict measurement discipline.

### Proposed Decision
Every hot-path change must reference benchmark deltas against pinned baselines, with vLLM as primary comparator.

### Consequences if accepted
- Slower but safer optimization cadence.
- Higher confidence in release quality and claims.

---

## ADR-006: Backend portability contract (CUDA now, ROCm later)

**Status:** Proposed  
**Proposed Date:** 2026-04-13  
**Review Trigger:** Before multi-GPU implementation hardens backend interfaces

### Context
Product roadmap includes AMD expansion.

### Proposed Decision
Define and protect a backend interface so scheduler/API/model layers remain vendor-agnostic.

### Consequences if accepted
- Requires stricter abstractions from day one.
- Avoids expensive rewrites when ROCm backend work begins.

---

## ADR-007: No Python runtime dependency at serve time

**Status:** Proposed  
**Proposed Date:** 2026-04-13  
**Review Trigger:** End of model loading/tokenizer MVP

### Context
Runtime Python dependencies increase cold-start complexity, memory footprint, and operational fragility.

### Proposed Decision
Keep serve path Python-free; use native loaders/runtime components.

### Consequences if accepted
- More implementation work in Go/C++.
- Cleaner deployment and lower runtime overhead.
