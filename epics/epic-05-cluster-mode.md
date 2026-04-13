# Epic 05 - Cluster Mode and Disaggregated Serving

**Phase:** 5  
**Target:** Future  
**Prerequisite:** Epic 04  
**Status:** Planned

See also:
- `epics/epic-05-execution-plan.md` for the fast-track, capability-gated implementation order used by coding agents.

---

## Goal
Enable resilient multi-node serving with disaggregated prefill/decode and cluster-aware routing.

---

## Stories

### E05-S01: Cluster control and node registration
**Points:** 8

Deliverables:
- node discovery/registration model
- capability and health metadata exchange
- control-plane API for fleet state

Acceptance:
- Cluster can discover and track serving nodes reliably.

### E05-S02: Disaggregated prefill/decode pipeline
**Points:** 13

Deliverables:
- prefill pool and decode pool roles
- remote handoff protocol for decode readiness
- scheduler integration for remote execution

Acceptance:
- End-to-end request path works across prefill/decode node split.

### E05-S03: Distributed KV and prefix cache strategy
**Points:** 13

Deliverables:
- remote KV handoff and ownership rules
- distributed prefix cache lookup policy
- eviction and consistency model

Acceptance:
- Cache and KV correctness preserved during node churn.

### E05-S04: Fault tolerance and recovery
**Points:** 8

Deliverables:
- request recovery strategy on node failure
- admission/routing fallback behavior
- SLO-aware degradation mode

Acceptance:
- Single-node failures do not cause full service outage.

### E05-S05: Cluster benchmarking and cost model
**Points:** 8

Deliverables:
- multi-node throughput/latency benchmark harness
- saturation and failure-mode benchmark scenarios
- cost-per-token reporting

Acceptance:
- Horizontal scale shows controlled latency growth and cost visibility.

---

## Exit Criteria
- Cluster remains available under single-node failures.
- Throughput scales with added nodes under validated workloads.

---

## Risks
- Network overhead and remote KV complexity.
- Operational complexity growing faster than performance gains.
