# gollmgo

A production-grade LLM inference engine written in Go with CUDA kernels. Built for high throughput, low latency, and single-binary operational simplicity.

**Status:** Phase 2 — single-GPU performance milestones closed on NVIDIA DGX Spark (GB10 Grace Blackwell). Chunked prefill, paged attention v2, CUDA graph capture/replay, prefix caching, quantized FP8/INT8 paths, n-gram speculative decoding, SJF/priority schedulers, swap-mode preemption, and an automated regression gate are all landed. See [epics/epic-02-execution-plan.md](epics/epic-02-execution-plan.md).

## Why gollmgo

- **One command, any HuggingFace model.** `gollmgo serve --model meta-llama/Llama-3.1-8B-Instruct` auto-downloads weights + tokenizer + config to a local cache. No manual wget, no Python env.
- **Single binary.** No Python runtime, no dependency maze.
- **OpenAI-compatible API.** Drop-in replacement via `base_url` override.
- **Continuous batching** with mixed prefill/decode, chunked prefill for long prompts, and configurable scheduler policies (FCFS / SJF / priority).
- **Paged KV cache** with block refcounting, prefix caching, CUDA graph capture for warm-path decode, and swap-mode preemption.
- **Backend abstraction.** CUDA-first, architecture ready for ROCm without scheduler/API rewrites.

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
make kernels   # compile CUDA kernels → static libs
make build     # go build with CGO_ENABLED=1 -tags gpu
```

### Serve a model

`--model` accepts either a HuggingFace Hub repo id or a local path. HF repos are downloaded on first use and reused from cache after that.

```bash
# HF Hub — downloads to ~/.cache/gollmgo/hub on first run
bin/gollmgo serve --model meta-llama/Llama-3.2-3B-Instruct

# Pin a revision
bin/gollmgo serve --model meta-llama/Llama-3.1-8B-Instruct@main

# Gated / private repos — set a bearer token
HF_TOKEN=hf_xxx bin/gollmgo serve --model meta-llama/Llama-3.1-8B-Instruct

# Local path (file or directory)
bin/gollmgo serve --model /models/my-llama/

# With explicit config + port override
bin/gollmgo serve --config config.example.json --port 8080 \
  --model meta-llama/Llama-3.2-3B-Instruct

# Mock-runner dev mode (no --model) — API works end-to-end, output is dummy
bin/gollmgo serve --port 8080
```

### Environment check

```bash
bin/gollmgo doctor --model meta-llama/Llama-3.2-3B-Instruct
# → probes Go runtime, binary, nvidia-smi, nvcc, config validity,
#   model repo reachability + cache state; non-zero exit on any FAIL
```

### Benchmark

```bash
# Automated baseline capture — starts server, benches, stores result
GOLLMGO_BENCH_MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct bench/capture_baseline.sh

# Ad-hoc offline or serving benchmarks
bin/gollmgo bench --mode offline  --num-prompts 100 --prompt-len 128 --output-len 128
bin/gollmgo bench --mode serving  --url http://localhost:8080 --num-prompts 100 --concurrency 10

# Regression gate: compare a run against the stored baseline
bash bench/check_regression.sh bench/results/current.json
```

### API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible chat (streaming + non-streaming) |
| `/v1/models` | GET | Loaded model id (real repo id when HF-loaded) |
| `/health/live` | GET | Liveness probe |
| `/health/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus-format metrics |

### Example request

```bash
MODEL=$(curl -fsS http://localhost:8080/v1/models | jq -r '.data[0].id')
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Write two sentences about Go.\"}],
    \"max_tokens\": 64,
    \"stream\": true
  }"
```

## Cache and auth

| Variable | Purpose |
|---|---|
| `GOLLMGO_CACHE_DIR` | HF download cache root (overrides default) |
| `XDG_CACHE_HOME` | Fallback root — used as `$XDG_CACHE_HOME/gollmgo/hub` |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Bearer token for gated/private repos |

Default cache layout: `~/.cache/gollmgo/hub/models--<org>--<name>/`. Deterministic, inspectable, and safe to `rm -rf` to reclaim space.

## Project layout

```
cmd/gollmgo/                 CLI entrypoint (serve, bench, doctor)
internal/
  api/                       OpenAI-compatible HTTP server
  scheduler/                 FCFS / SJF / priority schedulers, chunked prefill, preemption
  kvcache/                   Paged block pool, prefix cache
  engine/                    Forward orchestration, speculative decoding, KV swap
  backend/                   Runner interface + CUDA bridge
  backend/cuda/              CGo bridge (bridge.go, model_bridge.go, kvcache_bridge.go)
  model/                     Safetensors + GGUF loaders, HF BPE tokenizer
  model/hfhub/               HuggingFace Hub repo resolver + cache
  config/                    Config loading + validation
  metrics/                   Prometheus counters
  benchcheck/                Regression gate checker
kernels/
  gollmgo_backend.cu         Backend lifecycle (create/warmup/destroy)
  gollmgo_ops.cu             Transformer ops (RMSNorm, RoPE, SiLU, embedding)
  gollmgo_model.cu           Eager + paged forward pass with cuBLAS GEMM + CUDA graphs
  gollmgo_paged_attn.cu      PagedAttention v1 and v2 kernels
  gollmgo_kvcache.cu         GPU KV cache management
bench/
  baseline_config.json       Frozen parameters for parity runs
  baseline_result.json       Pinned baseline (produced by capture_baseline.sh)
  thresholds.json            Regression thresholds
  capture_baseline.sh        One-shot: serve → bench → install baseline
docs/                        Architecture, engineering standards, roadmap, benchmarks
epics/                       Execution plans (Epic 01, Epic 02, ...)
decisions/                   Architecture Decision Records
```

## Development

### Make targets

```bash
make build                 # build binary (CGO_ENABLED=1, -tags gpu)
make test                  # CPU-only unit tests (no GPU required)
make test-gpu              # GPU integration tests
make test-kernels          # CUDA kernel correctness tests (vs CPU reference)
make test-paged-attn       # Paged-vs-naive + v2-vs-v1 parity sweep
make test-paged-attn-parity # Alias for the M2 parity gate
make kernels               # compile CUDA kernels to static libs
make bench                 # Go benchmark suite
make lint                  # go vet
make clean                 # remove build artifacts
```

### Running tests

```bash
# All CPU tests — runs everywhere, no GPU needed
make test

# GPU tests — requires NVIDIA GPU + compiled kernels
make kernels && make test-gpu

# Kernel correctness — standalone CUDA test binaries
make test-kernels
make test-paged-attn          # v1-vs-naive + v2-vs-v1 sweep (5 shape/GQA configs)
```

### Config

See [config.example.json](config.example.json) for all options. Key settings:

| Setting | Default | Description |
|---|---|---|
| `port` | 8080 | HTTP listen port |
| `model_path` | — | HF repo id (`"owner/name[@rev]"`) or local path |
| `scheduler_policy` | `fcfs` | `fcfs`, `sjf`, or `priority` |
| `preempt_mode` | `recompute` | `recompute` or `swap` (swap needs KVSwapper-capable runner) |
| `max_batch_size` | 64 | Max sequences per scheduler tick |
| `max_token_budget` | 4096 | Max tokens (prefill + decode) per tick |
| `max_queue_depth` | 256 | Backpressure: max waiting requests |
| `prefill_chunk_size` | 512 | Max prefill tokens per sequence per tick (0 disables chunking) |
| `block_size` | 16 | KV cache block size (tokens per block) |
| `max_memory_fraction` | 0.9 | Fraction of free GPU memory for KV cache |
| `prefix_caching` | false | Enable block-level prefix cache for KV reuse |
| `prefix_cache_max_blocks` | pool/2 | LRU cap on prefix cache size |
| `quantization` | — | `""`, `"fp8"`, or `"int8"` |
| `speculative.enabled` | false | Enable n-gram speculative decoding |
| `speculative.ngram_size` | 3 | N-gram window for the drafter |
| `speculative.num_draft_tokens` | 4 | K — drafts per step |
| `speculative.kill_threshold` | 0 | Disable drafter if acceptance rate drops below this |
| `hf_cache_dir` | `~/.cache/gollmgo/hub` | HF download cache root |
| `hf_token` | `$HF_TOKEN` | Bearer token for gated repos |

## CUDA kernel inventory

| Kernel | File | Status |
|---|---|---|
| Embedding lookup (FP16 / BF16) | `gollmgo_ops.cu` | Tested |
| RMSNorm (FP16 / BF16, warp+block reduce) | `gollmgo_ops.cu` | Tested |
| RoPE (FP16 / BF16, GQA-aware) | `gollmgo_ops.cu` | Tested |
| SiLU × Up (FP16 / BF16) | `gollmgo_ops.cu` | Tested |
| Naive attention (FP16, causal, GQA) | `gollmgo_ops.cu` | Reference for parity tests |
| Paged KV write (FP16 / BF16) | `gollmgo_paged_attn.cu` | Tested |
| PagedAttention v1 (FP16 / BF16, GQA) | `gollmgo_paged_attn.cu` | Tested |
| PagedAttention v2 (partitioned, long-context) | `gollmgo_paged_attn.cu` | Tested — v2-vs-v1 parity sweep |
| Residual add | `gollmgo_ops.cu` | Used |
| Linear (cuBLAS HGEMM / BGEMM, FP8/INT8 gated) | via cuBLAS | Used |
| CUDA graph capture + replay for decode | `gollmgo_model.cu` | Live, hit-rate metered |

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 1 — Foundation | Single-GPU MVP with continuous batching | **Complete** |
| 2 — Performance | Chunked prefill, CUDA graphs, prefix cache, FP8/INT8, speculative decode, SJF/priority, regression gate | **Complete** ([epics/epic-02-execution-plan.md](epics/epic-02-execution-plan.md)) |
| 3 — Production | Graceful drain, rate limits, zero-downtime reload | Planned |
| 4 — Multi-GPU | Tensor parallelism, multi-LoRA | Planned |
| 5 — Cluster | Disaggregated prefill/decode, distributed KV | Planned |
| 6 — AMD | ROCm backend via same Runner interface | Planned |

See [docs/roadmap.md](docs/roadmap.md) for details.

## Known follow-ups (tracked on Epic 2 board)

- `CUDARunner.SnapshotKV` / `RestoreKV` are defined interfaces with stub implementations; `Capabilities.KVSwap=false` until the device→host copies are wired, so swap-mode preemption transparently falls back to recompute on GPU today. Works end-to-end on the mock runner.
- Speculative decoding is wired into the serving loop and validated on the mock runner; GPU `Capabilities.SpeculativeDecoding=false` until multi-position decode dispatch is exercised.
- Sharded GGUF (Q4_0 / Q5_K / etc.) is not supported — GGUF loader handles F32/F16/BF16 only. Safetensors sharding (multi-file HF layouts) is fully supported.
- Partial/resume HTTP downloads — HF cache re-fetches the whole file on interruption. Follow-up with `Range` headers is cheap.

## License

Apache 2.0. See [LICENSE](LICENSE).
