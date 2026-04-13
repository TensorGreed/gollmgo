# gollmgo

A production-grade LLM inference engine written in Go with CUDA kernels. Built for high throughput, low latency, and single-binary operational simplicity.

**Status:** Phase 1 — single-GPU foundation on NVIDIA DGX Spark (GB10 Grace Blackwell).

## Why gollmgo

- **Single binary** — no Python runtime, no dependency maze. `gollmgo serve` and you're running.
- **OpenAI-compatible API** — drop-in replacement via `base_url` override.
- **Continuous batching** — FCFS scheduler with mixed prefill/decode, memory-pressure preemption.
- **Paged KV cache** — block-allocated with refcounting. PagedAttention v1 kernel verified against naive attention (0.0000 max diff).
- **Backend abstraction** — CUDA-first, architecture ready for ROCm without scheduler/API rewrites.

## Architecture

```
Client → OpenAI API Server → Scheduler → Engine → KV Cache → Backend Runner → CUDA Kernels
```

Each component owns its boundary. The scheduler owns sequence state. The KV cache owns GPU memory for keys/values. The backend runner owns kernel dispatch. See [docs/architecture.md](docs/architecture.md) for the full design.

## Quick start

### Prerequisites

- Go 1.22.2+ (CI baseline: Go 1.22.2)
- CUDA Toolkit 12+ (13.0 tested)
- NVIDIA GPU (developed on DGX Spark GB10)
- `gcc` with C++17

### Build

```bash
# If go is not in PATH, set it:
# export GO=/path/to/go  OR  export PATH=/path/to/go/bin:$PATH

make kernels   # compile CUDA kernels → static libs
make build     # go build with CGO_ENABLED=1 -tags gpu
```

### Run

```bash
# Serve with default config (mock runner for development)
./bin/gollmgo serve --port 8080

# Serve with config file
./bin/gollmgo serve --config config.example.json

# Offline benchmark
./bin/gollmgo bench --mode offline --num-prompts 100 --prompt-len 128 --output-len 128

# Serving benchmark against a running server
./bin/gollmgo bench --mode serving --url http://localhost:8080 --num-prompts 100 --concurrency 10

# Health check
./bin/gollmgo doctor
```

### API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible chat (streaming + non-streaming) |
| `/v1/models` | GET | List loaded models |
| `/health/live` | GET | Liveness probe |
| `/health/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus-format metrics |

### Example request

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gollmgo-default",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "stream": true
  }'
```

## Project layout

```
cmd/gollmgo/             CLI entrypoint (serve, bench, doctor)
internal/
  api/                   OpenAI-compatible HTTP server + middleware
  scheduler/             FCFS continuous batching scheduler
  kvcache/               Paged block pool + block tables
  engine/                Forward orchestration, sampling, serving loop
  backend/               Runner interface + CUDA bridge
  backend/cuda/          CGo bridge to CUDA kernels
  model/                 Safetensors/GGUF loaders, HF BPE tokenizer
  config/                Config loading + validation
  metrics/               Prometheus counters
kernels/
  gollmgo_backend.cu     Backend lifecycle (create/warmup/destroy)
  gollmgo_ops.cu         Transformer ops (RMSNorm, RoPE, SiLU, embedding)
  gollmgo_model.cu       Eager + paged forward pass with cuBLAS GEMM
  gollmgo_paged_attn.cu  PagedAttention v1 kernel
  gollmgo_kvcache.cu     GPU KV cache management
docs/                    Architecture, engineering standards, roadmap
epics/                   Execution plans
decisions/               Architecture Decision Records
```

## Development

### Make targets

```bash
make build            # build binary (CGO_ENABLED=1, -tags gpu)
make test             # CPU-only unit tests (no GPU required)
make test-gpu         # GPU integration tests
make test-kernels     # CUDA kernel correctness tests (vs CPU reference)
make test-paged-attn  # Paged-vs-naive attention correctness test
make kernels          # compile CUDA kernels to static libs
make bench            # Go benchmark suite
make lint             # go vet
make clean            # remove build artifacts
```

### Running tests

```bash
# All CPU tests — runs everywhere, no GPU needed
make test

# GPU tests — requires NVIDIA GPU + compiled kernels
make kernels && make test-gpu

# Kernel correctness — standalone CUDA test binaries
make test-kernels
make test-paged-attn
```

### Test coverage

7 packages, 80+ tests including:
- Sequence state machine transitions and edge cases
- Scheduler admission, budget limits, preemption, deterministic ordering
- Block pool allocation, refcounting, exhaustion
- Engine token isolation (concurrent requests cannot steal tokens)
- Step-failure recovery (no stranded sequences)
- Queue saturation (enqueue failures don't hang)
- 3-second soak test under sustained concurrent load
- API request validation, streaming, non-streaming
- RMSNorm, SiLU, embedding kernel correctness vs CPU reference
- Paged attention vs naive attention (exact match)

### Config

See [config.example.json](config.example.json) for all options. Key settings:

| Setting | Default | Description |
|---|---|---|
| `port` | 8080 | HTTP listen port |
| `max_batch_size` | 64 | Max sequences per scheduler tick |
| `max_token_budget` | 4096 | Max tokens (prefill + decode) per tick |
| `max_queue_depth` | 256 | Backpressure: max waiting requests |
| `block_size` | 16 | KV cache block size (tokens per block) |
| `max_memory_fraction` | 0.9 | Fraction of GPU memory for KV cache |

## CUDA kernel inventory

| Kernel | File | Status |
|---|---|---|
| Embedding lookup (FP16) | `gollmgo_ops.cu` | Tested |
| RMSNorm (FP16, warp+block reduce) | `gollmgo_ops.cu` | Tested |
| RoPE (FP16, GQA-aware) | `gollmgo_ops.cu` | Tested |
| SiLU * Up (FP16) | `gollmgo_ops.cu` | Tested |
| Naive attention (FP16, causal, GQA) | `gollmgo_ops.cu` | Tested |
| Paged KV write (FP16) | `gollmgo_paged_attn.cu` | Tested |
| PagedAttention v1 (FP16, GQA) | `gollmgo_paged_attn.cu` | Tested |
| Residual add (FP16) | `gollmgo_ops.cu` | Used |
| FP16 → FP32 conversion | `gollmgo_ops.cu` | Used |
| Linear (cuBLAS HGEMM) | via cuBLAS | Used |

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 1 — Foundation | Single-GPU MVP with continuous batching | In progress |
| 2 — Performance | Chunked prefill, CUDA graphs, FP8, speculative decode | Planned |
| 3 — Production | Graceful drain, rate limits, zero-downtime reload | Planned |
| 4 — Multi-GPU | Tensor parallelism, multi-LoRA | Planned |
| 5 — Cluster | Disaggregated prefill/decode, distributed KV | Planned |
| 6 — AMD | ROCm backend via same Runner interface | Planned |

See [docs/roadmap.md](docs/roadmap.md) for details.

## License

Apache 2.0. See [LICENSE](LICENSE).
