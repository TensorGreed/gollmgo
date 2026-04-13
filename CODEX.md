# gollmgo - Codex Project Memory

## What Codex Should Assume
- Product goal: build a world-class LLM inference engine.
- Hardware-first target: `NVIDIA DGX Spark (GB10 Grace Blackwell superchip)`.
- Delivery order: single GPU -> multi-GPU -> cluster -> AMD.

## Immediate Priorities
1. Single-GPU correctness and stability.
2. Continuous batching + paged KV performance.
3. OpenAI-compatible API and production controls.
4. Best-in-class benchmark performance on pinned workloads.

## Engineering Guardrails
- Keep scheduler tick path fast and non-blocking.
- Keep CGo/backend boundary narrow.
- Treat benchmark regressions as release blockers.
- Avoid runtime Python dependencies in serving path.
- Use `Go 1.22.2+` locally; CI is pinned to `Go 1.22.2`.

## Read Order For Any Task
1. `docs/roadmap.md`
2. `docs/architecture.md`
3. Relevant `epics/*.md`
4. `docs/engineering.md`
5. `docs/configuration.md` (if behavior/config changes)

## Definition of Good Change
- Correct behavior with tests.
- Clear operator-facing docs.
- Measurable performance impact for hot-path changes.
- No architecture boundary violations.

## Suggested Task Split
- API and config work: `internal/api`, `internal/config`, docs.
- Scheduler work: `internal/scheduler`, `internal/kvcache`, benchmarks.
- Kernel/backend work: `kernels`, `internal/backend`, profiling notes.

## Non-Goals (for now)
- Training workflows.
- Cluster features before single-GPU parity is real.
- Broad hardware abstraction before CUDA path is mature.
