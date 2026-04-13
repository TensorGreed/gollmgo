# gollmgo - Engineering Standards

## Engineering Values
- Correctness before optimization.
- Measured optimization before speculation.
- Operational simplicity as a first-class requirement.
- Consistent architecture boundaries across all features.

---

## Go Standards

### Package boundaries
- `cmd/` wires dependencies only.
- `internal/` holds implementation.
- Keep scheduler, cache, backend, and API concerns separated.

### Error handling
- Wrap errors with component context.
- Use typed sentinel errors for control flow.
- No string matching for error semantics.

### Concurrency
- Scheduler tick path must remain lock-light and non-blocking.
- Every goroutine must have explicit shutdown behavior.
- Use context cancellation and wait groups for lifecycle control.

### Logging
- Use structured logs (`slog` or equivalent).
- Include request/sequence IDs in all request-scoped logs.
- Avoid noisy per-token logs outside debug mode.

---

## Backend and CGo Rules

### Golden boundary
All GPU runtime calls stay behind backend runners and kernel bridge layers.

### C API design
- Expose only C-compatible types across CGo.
- Return explicit status codes and fetch error details via API.
- Keep header surface small and versioned.

### Memory ownership
- GPU buffers are owned by backend runtime.
- Go passes only CPU-side inputs and receives outputs/handles.
- KV memory lifecycle is managed by the cache/runner boundary.

---

## Testing Pyramid

### Unit tests
- No GPU dependency.
- Mock runner/cache/tokenizer interfaces.

### GPU integration tests
- Tagged and isolated.
- Validate correctness on real hardware.

### Benchmarks
- Required for scheduler/cache/kernel/hot-path changes.
- Compared against stored baselines in CI.

---

## Performance Workflow
1. Capture baseline metrics.
2. Form one optimization hypothesis.
3. Implement minimal change.
4. Re-run benchmark and profile.
5. Keep only measurable wins.

Hot-path checklist:
- [ ] No avoidable allocations in decode loop
- [ ] No blocking I/O in scheduler tick
- [ ] No hidden CPU fallback
- [ ] Kernel path selected as expected

---

## Code Review Requirements
- [ ] Tests pass (`make test`, plus GPU tests when relevant)
- [ ] Lint and static checks pass
- [ ] Config/docs updated for user-visible behavior changes
- [ ] Benchmark delta included for hot-path modifications
- [ ] Rollback strategy described for risky changes

---

## Dependency Policy
Before adding a dependency:
1. Can stdlib solve it?
2. Is maintenance and security posture acceptable?
3. Does it introduce hard runtime dependencies we do not want?

Prefer fewer dependencies and clearer operational footprints.

---

## Agent Collaboration Rules (Claude + Codex)
- Always read `docs/roadmap.md` and the relevant epic before implementation.
- Keep commits and patches scoped to one objective.
- Update docs with code changes in the same PR.
- If assumptions are made (hardware, model shape, benchmark mode), state them explicitly.
