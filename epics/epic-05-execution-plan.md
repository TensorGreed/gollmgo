# Epic 05 - Fast-Track Execution Plan

## Purpose
This is the agent-friendly execution board for Phase 5. It stays intentionally compact because cluster mode should only harden after multi-GPU assumptions are real, not imagined.

---

## Board
- [ ] M0: Cluster contract and control plane
- [ ] M1: Remote execution protocol
- [ ] M2: Disaggregated prefill/decode slice
- [ ] M3: Distributed KV and prefix strategy
- [ ] M4: Failure handling and recovery
- [ ] M5: Cluster gate

---

## Milestones

### M0: Cluster contract and control plane
Maps to:
- E05-S01

Outputs:
- node identity and registration model
- fleet health and capability schema
- control-plane API outline

Verification:
- simulated node registration and state tracking works

### M1: Remote execution protocol
Maps to:
- E05-S02

Outputs:
- request handoff format
- remote decode-readiness contract
- timeout and retry semantics

Verification:
- request can cross process or node boundary in controlled test

Depends on:
- M0

### M2: Disaggregated prefill/decode slice
Maps to:
- E05-S02

Outputs:
- first end-to-end split execution path
- scheduler routing for prefill vs decode roles

Verification:
- controlled demo workload completes across split roles

Depends on:
- M1

### M3: Distributed KV and prefix strategy
Maps to:
- E05-S03

Outputs:
- KV ownership and transfer rules
- distributed prefix lookup behavior
- eviction and consistency policy

Verification:
- correctness preserved under node churn simulation

Depends on:
- M2

### M4: Failure handling and recovery
Maps to:
- E05-S04

Outputs:
- node-loss handling path
- degraded-mode routing behavior
- recovery or retry policy

Verification:
- single-node failure does not cascade into full outage

Depends on:
- M2

### M5: Cluster gate
Maps to:
- E05-S05

Outputs:
- multi-node benchmark harness
- failure-mode benchmarks
- cost-per-token reporting

Verification:
- horizontal scale shows useful throughput gain with controlled latency growth

Depends on:
- M3
- M4

---

## Parallel Work Guidance
M3 and M4 can overlap after the first split execution path exists. Keep simulated environments first; do not drag in full production orchestration before the execution model is proven.

---

## Agent Rules
- Favor explicit contracts over hidden distributed behavior.
- Treat recovery semantics as part of the product surface, not an afterthought.
