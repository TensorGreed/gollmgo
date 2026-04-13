# Epic 04 - Single-Node Multi-GPU Scale-Up

**Phase:** 4  
**Target:** Q4 2026  
**Prerequisite:** Epic 03  
**Status:** Planned

See also:
- `epics/epic-04-execution-plan.md` for the fast-track, capability-gated implementation order used by coding agents.

---

## Goal
Scale from single GPU to multi-GPU on one node while preserving API behavior and production reliability.

---

## Stories

### E04-S01: Parallelism strategy baseline
**Points:** 8

Deliverables:
- tensor parallel execution path (2/4/8 GPUs)
- communication topology selection
- model partition metadata handling

Acceptance:
- Functional multi-GPU inference on target model with deterministic outputs.

### E04-S02: Distributed runtime plumbing
**Points:** 8

Deliverables:
- NCCL communication management
- fault-aware process/group lifecycle
- startup validation for GPU topology

Acceptance:
- Stable startup and shutdown across supported GPU counts.

### E04-S03: Scheduler and KV awareness per device
**Points:** 8

Deliverables:
- per-device memory pressure visibility
- admission decisions aware of shard state
- metrics for cross-device bottlenecks

Acceptance:
- Scheduler avoids pathological imbalance across GPUs.

### E04-S04: Multi-LoRA serving at scale
**Points:** 8

Deliverables:
- adapter management API
- adapter-aware batching strategy
- cache policy for adapter weights

Acceptance:
- Multi-adapter serving overhead remains within SLO budget.

### E04-S05: Multi-GPU benchmark suite
**Points:** 5

Deliverables:
- standardized scale-up benchmark matrix
- comparison runs vs single-GPU baseline
- release threshold definitions for scale efficiency

Acceptance:
- Near-linear scaling demonstrated at least through 4 GPUs on target workloads.

---

## Exit Criteria
- Reliable multi-GPU serving with production controls intact.
- Scale efficiency and latency meet published thresholds.

---

## Risks
- Communication overhead limiting scaling.
- Scheduler complexity increases with device-level state.
