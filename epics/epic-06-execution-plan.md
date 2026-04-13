# Epic 06 - Fast-Track Execution Plan

## Purpose
This is the agent-friendly execution board for Phase 6. It keeps the ROCm expansion grounded in backend parity instead of drifting into a rewrite.

---

## Board
- [ ] M0: Backend contract freeze
- [ ] M1: ROCm runtime bring-up
- [ ] M2: HIP kernel bridge for core ops
- [ ] M3: Scheduler and cache semantic parity
- [ ] M4: Quantization and capability parity
- [ ] M5: Cross-vendor gate

---

## Milestones

### M0: Backend contract freeze
Maps to:
- E06-S01

Outputs:
- finalized backend interface contracts
- provider capability model
- interface-level parity tests

Verification:
- core engine code runs unchanged against provider stubs

### M1: ROCm runtime bring-up
Maps to:
- E06-S02

Outputs:
- ROCm runner lifecycle
- startup validation and diagnostics
- first successful device initialization

Verification:
- build and warmup path works on supported AMD target

Depends on:
- M0

### M2: HIP kernel bridge for core ops
Maps to:
- E06-S02

Outputs:
- HIP path for attention and cache-critical ops
- error mapping and debug visibility

Verification:
- end-to-end inference path works on supported model/profile

Depends on:
- M1

### M3: Scheduler and cache semantic parity
Maps to:
- E06-S03

Outputs:
- matching preemption, prefix cache, and chunked prefill behavior
- provider-aware config validation

Verification:
- request semantics match CUDA path for supported features

Depends on:
- M2

### M4: Quantization and capability parity
Maps to:
- E06-S04

Outputs:
- ROCm quantization support matrix
- fallback rules for unsupported precision modes
- quality and performance comparison suite

Verification:
- quantized modes have documented quality and performance behavior

Depends on:
- M2

### M5: Cross-vendor gate
Maps to:
- E06-S05

Outputs:
- NVIDIA vs AMD benchmark matrix
- release report format
- periodic parity checks

Verification:
- published, reproducible cross-vendor comparison exists

Depends on:
- M3
- M4

---

## Parallel Work Guidance
M3 and M4 can overlap after M2 is functional. Keep the first AMD target narrow and well-supported rather than pretending broad parity too early.

---

## Agent Rules
- Never let backend-specific conditionals leak back into scheduler or API logic.
- Publish capability gaps plainly instead of hiding them behind partial support.
