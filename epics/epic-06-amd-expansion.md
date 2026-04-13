# Epic 06 - AMD ROCm Expansion

**Phase:** 6  
**Target:** Future  
**Prerequisite:** Epic 02 for baseline parity, Epic 03 for production controls  
**Status:** Planned

See also:
- `epics/epic-06-execution-plan.md` for the fast-track, capability-gated implementation order used by coding agents.

---

## Goal
Add AMD backend support with feature parity for core serving behavior and clear cross-vendor performance visibility.

---

## Stories

### E06-S01: Backend abstraction hardening
**Points:** 8

Deliverables:
- finalized runner/backend interface contracts
- shared tensor/launch abstraction used by CUDA and ROCm backends
- parity tests at interface level

Acceptance:
- Core engine/scheduler code runs unchanged across backend providers.

### E06-S02: ROCm runner and kernel bridge
**Points:** 13

Deliverables:
- ROCm runtime integration
- HIP kernel bridge for attention/cache ops
- robust error mapping and diagnostics

Acceptance:
- End-to-end serving works on supported AMD hardware.

### E06-S03: Feature parity for scheduling and cache semantics
**Points:** 8

Deliverables:
- parity for preemption/chunked prefill/prefix caching behavior
- backend capability negotiation logic
- config validation by provider

Acceptance:
- Request semantics match CUDA path for supported features.

### E06-S04: Quantization path parity and quality checks
**Points:** 8

Deliverables:
- supported quantization modes for ROCm path
- quality and performance comparison suite
- fallback policy for unsupported precision modes

Acceptance:
- Quantized serving works with documented quality/perf characteristics.

### E06-S05: Cross-vendor benchmark matrix
**Points:** 5

Deliverables:
- benchmark matrix across NVIDIA and AMD targets
- release report format with vendor-specific caveats
- CI hooks for periodic parity checks

Acceptance:
- Published and reproducible cross-vendor performance reports.

---

## Exit Criteria
- Core API and scheduling semantics are consistent across vendors.
- ROCm backend is production-usable for defined model/profile set.

---

## Risks
- Feature lag due to backend kernel maturity differences.
- Additional maintenance burden from dual-backend support.
