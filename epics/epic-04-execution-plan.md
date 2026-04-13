# Epic 04 - Fast-Track Execution Plan

## Purpose
This is the agent-friendly execution board for Phase 4. It is intentionally lighter than the Phase 1 and 2 plans because scale-up work depends on what the single-GPU system actually becomes.

---

## Board
- [ ] M0: Topology and partitioning spike
- [ ] M1: Multi-GPU runtime foundation
- [ ] M2: Tensor parallel inference path
- [ ] M3: Scheduler and KV device awareness
- [ ] M4: Multi-LoRA scale path
- [ ] M5: Scale-up gate

---

## Milestones

### M0: Topology and partitioning spike
Maps to:
- E04-S01

Outputs:
- chosen first topology target
- model partition metadata shape
- communication assumptions and constraints

Verification:
- design spike or prototype validates basic feasibility

### M1: Multi-GPU runtime foundation
Maps to:
- E04-S02

Outputs:
- process group lifecycle
- NCCL setup and teardown
- topology validation at startup

Verification:
- stable bring-up and shutdown across supported GPU counts

### M2: Tensor parallel inference path
Maps to:
- E04-S01

Outputs:
- distributed forward path
- deterministic output checks against single-GPU baseline

Verification:
- target model runs correctly on 2 GPUs first

Depends on:
- M1

### M3: Scheduler and KV device awareness
Maps to:
- E04-S03

Outputs:
- per-device pressure metrics
- admission logic aware of distributed state
- imbalance detection signals

Verification:
- scheduler avoids repeated shard imbalance during load tests

Depends on:
- M2

### M4: Multi-LoRA scale path
Maps to:
- E04-S04

Outputs:
- adapter lifecycle API
- adapter-aware batching
- adapter cache policy

Verification:
- throughput overhead stays inside defined budget

Depends on:
- M2

### M5: Scale-up gate
Maps to:
- E04-S05

Outputs:
- scale-up benchmark suite
- published scale-efficiency targets
- release note template for multi-GPU runs

Verification:
- near-linear scaling demonstrated through first supported topology

Depends on:
- M2
- M3

---

## Parallel Work Guidance
M3 and M4 can overlap once M2 is stable on the first topology. Keep the first target narrow; expanding supported topologies too early will slow everything down.

---

## Agent Rules
- Lock the first supported topology before widening support.
- Compare every multi-GPU result back to a single-GPU baseline.
