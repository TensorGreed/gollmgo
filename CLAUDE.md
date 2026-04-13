# gollmgo - Claude Code Project Memory

## Mission
Build a production-grade inference engine with best-in-class throughput and latency, easier to operate than Python-heavy stacks, and extensible to multi-node and multi-vendor GPUs.

## Product Bar (Non-Negotiable)
- Performance: competitive tokens/sec, TTFT, and ITL on real workloads.
- Reliability: safe under load, predictable latency, graceful degradation.
- Ease of use: single-binary workflow, OpenAI-compatible API, clear config.
- Portability: CUDA-first now, backend abstraction so ROCm can follow cleanly.

## Default Hardware Target
- Primary dev/deploy box: `NVIDIA DGX Spark (GB10 Grace Blackwell superchip)`
- Execution strategy:
1. Win on single GPU first.
2. Scale to single-node multi-GPU.
3. Move to cluster/disaggregated serving.
4. Add AMD ROCm backend without rewriting scheduler/API.

## What Claude Should Prioritize
1. Keep scheduler and KV cache correctness first.
2. Keep hot paths allocation-free on decode loops.
3. Keep CGo boundary narrow and explicit.
4. Keep operator UX simple (`serve`, `bench`, `doctor`, one config file).
5. Keep every performance claim benchmark-backed.

## Repo Navigation (Planned)
```text
cmd/gollmgo/                 # CLI entrypoint
internal/api/                # OpenAI-compatible API + middleware
internal/scheduler/          # batching, admission, preemption
internal/kvcache/            # paged KV cache + prefix cache
internal/engine/             # forward orchestration + CGo runner
internal/backend/            # CUDA first, ROCm abstraction later
internal/model/              # safetensors/GGUF loading
internal/quantize/           # fp16/bf16/fp8/int8 paths
internal/metrics/            # Prometheus + tracing
kernels/                     # .cu kernels + C API implementation
docs/                        # product + engineering docs
epics/                       # execution backlog
skills/                      # focused implementation playbooks
```

## Build / Test / Benchmark
```bash
make build            # build binary
make test             # CPU-only unit tests
make test-gpu         # GPU integration tests
make kernels          # compile CUDA kernels
make bench            # benchmark suite
make lint             # static checks
```

Go baseline: `Go 1.22.2+` for local development, with CI pinned to `Go 1.22.2`.

## Hard Constraints
- `CGO_ENABLED=1` for GPU builds.
- No GPU allocation outside engine/kvcache ownership boundaries.
- Scheduler tick path must never block on I/O or unbounded channels.
- No silent CPU fallback in any configured GPU execution path.
- Every new scheduling/attention optimization needs correctness tests and benchmark deltas.

## Performance Targets (Single GPU Exit)
- ShareGPT throughput: best-in-class on same hardware/config.
- TTFT P50 (128-token prompt): `<= 15ms`.
- ITL P50: `<= 10ms`.
- Under sustained load: stable P99 latency without queue collapse.

## Claude Task Workflow
1. Read `docs/roadmap.md`, `docs/architecture.md`, and relevant `epics/` file.
2. Implement smallest complete slice that ships value.
3. Add/adjust tests before deep optimization.
4. Run relevant benchmarks if touching scheduler/kernels/cache.
5. Update docs in same change when behavior/config changes.

## Anti-Patterns
- Premature cluster features before single-GPU parity.
- Feature additions without operator controls/observability.
- Large CGo API surfaces exposing CUDA/C++ internals to Go.
- Performance claims without reproducible benchmark runs.
