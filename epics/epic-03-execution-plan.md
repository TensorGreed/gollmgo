# Epic 03 - Fast-Track Execution Plan

## Purpose
This is the agent-friendly execution board for Phase 3. It focuses on making the single-GPU server production-usable without turning the codebase into a pile of operational special cases.

---

## Board
- [ ] M0: Failure model and SLO baseline
- [ ] M1: Graceful lifecycle
- [ ] M2: Protection and overload controls
- [ ] M3: Observability depth
- [ ] M4: Model lifecycle operations
- [ ] M5: Release gates and runbooks

---

## Milestones

### M0: Failure model and SLO baseline
Maps to:
- E03-S05 (foundation)

Outputs:
- explicit failure catalog
- production SLO draft
- soak/stress benchmark definitions

Verification:
- repeatable soak and overload test harness exists

Unblocks:
- all later milestones

### M1: Graceful lifecycle
Maps to:
- E03-S01

Outputs:
- startup readiness gating
- graceful shutdown and drain path
- request cancellation propagation

Verification:
- planned shutdown test completes without dropping active work inside timeout budget

### M2: Protection and overload controls
Maps to:
- E03-S02

Outputs:
- API key auth
- rate limits and token quotas
- overload responses and admission backpressure

Verification:
- overload tests show controlled degradation instead of queue collapse

### M3: Observability depth
Maps to:
- E03-S03

Outputs:
- structured logs with stable IDs
- expanded metric coverage
- tracing hooks for critical request path

Verification:
- operators can reconstruct a latency spike from telemetry alone

### M4: Model lifecycle operations
Maps to:
- E03-S04

Outputs:
- model load, unload, and reload controls
- admin endpoints with access control
- rollback-safe model switch flow

Verification:
- controlled reload succeeds without downtime in staged test

### M5: Release gates and runbooks
Maps to:
- E03-S05

Outputs:
- benchmark and reliability CI gates
- release checklist
- deployment and incident runbooks
- config migration notes

Verification:
- release process blocks on defined perf and reliability regressions

Depends on:
- M0
- M1
- M2
- M3
- M4

Exit gate:
- stable soak behavior
- production telemetry sufficient for debugging
- release process can be followed without tribal knowledge

---

## Parallel Work Guidance
After M0:
- M1 and M3 can move in parallel.
- M2 can begin once request identity and config wiring are stable.
- M4 should not start before lifecycle and auth basics are clear.

---

## Agent Rules
- Bias toward explicit operational behavior over clever magic.
- Document every new operator-facing control in the same patch.
- Reliability fixes that add latency cost need benchmark notes.
