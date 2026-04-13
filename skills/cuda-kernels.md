# Skill: CUDA Kernel Development

## Use This Skill When
- Editing files in `kernels/`
- Adding or tuning attention/cache kernels
- Debugging kernel correctness or performance issues

---

## Kernel Workflow
1. Build a reference implementation (CPU or trusted backend).
2. Add correctness tests before optimization.
3. Implement kernel with explicit memory/indexing assumptions.
4. Profile with Nsight Compute.
5. Integrate through the backend bridge with error-safe handling.

---

## Core Rules
- Keep kernel API stable and minimal.
- Validate every CUDA API return code.
- Use FP32 accumulation where numerically required.
- Never skip bounds checks on block-table indexing.
- Document kernel launch assumptions (block size, grid strategy).

---

## Paged KV Access Reminder
- Block table maps logical sequence blocks to physical cache blocks.
- Ensure physical block ID validation before pointer arithmetic.
- Preserve coalesced access where possible; poor access patterns will dominate runtime.

---

## Profiling Checklist
Use Nsight Compute for each significant kernel change:
- achieved memory bandwidth
- occupancy
- warp divergence
- shared-memory bank conflicts
- register pressure and spills

Keep before/after profiles with benchmark notes in PRs.

---

## Integration Checklist
- [ ] Kernel correctness tests pass
- [ ] Error paths tested (invalid shape, OOM, bad indices)
- [ ] Backend bridge updated safely
- [ ] Benchmark delta reported
- [ ] Architecture docs updated if dispatch behavior changes
