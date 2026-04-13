# gollmgo - Product Roadmap

## Vision
Deliver a world-class inference engine that matches or beats vLLM across core serving workloads while being easier to deploy and operate.

Primary sequence:
1. Single-GPU excellence on `NVIDIA DGX Spark (GB10 Grace Blackwell superchip)`.
2. Single-node multi-GPU scaling.
3. Cluster/disaggregated serving.
4. AMD support without API/scheduler rewrites.

---

## Product Principles
- Performance is a feature.
- Operator ergonomics are first-class.
- Reproducible benchmarks decide priority.
- Architecture must preserve backend portability.

---

## Epic Mapping
- Phase 1: `epics/epic-01-foundation.md`
- Phase 2: `epics/epic-02-performance.md`
- Phase 3: `epics/epic-03-production-hardening.md`
- Phase 4: `epics/epic-04-multi-gpu.md`
- Phase 5: `epics/epic-05-cluster-mode.md`
- Phase 6: `epics/epic-06-amd-expansion.md`

---

## Phase 1 - Single-GPU MVP (Q2 2026)
Goal: ship a usable OpenAI-compatible server on DGX Spark with correct continuous batching and paged KV cache.

### Must-have
- [ ] CLI: `serve`, `bench`, `doctor`
- [ ] OpenAI-compatible API: `/v1/chat/completions`, streaming + non-streaming, `/v1/models`
- [ ] LLaMA-family support via safetensors and GGUF
- [ ] Correct FP16/BF16 forward path
- [ ] PagedAttention v1 + KV cache manager
- [ ] FCFS continuous batching scheduler
- [ ] Prometheus metrics + health/readiness endpoints
- [ ] Deterministic benchmark harness (throughput/latency/memory)

### Exit Criteria
- [ ] Stable serving on DGX Spark for 24h soak test
- [ ] No correctness drift against reference outputs on fixed prompts
- [ ] Throughput reaches at least 60% of vLLM baseline on same hardware

---

## Phase 2 - vLLM Parity+ on Single GPU (Q2-Q3 2026)
Goal: reach and exceed vLLM on selected production workloads.

### Must-have
- [ ] PagedAttention v2 (long-context optimized)
- [ ] Chunked prefill + mixed prefill/decode scheduling
- [ ] CUDA graph capture/replay for common batch shapes
- [ ] Prefix caching with block-level LRU
- [ ] FP8 and INT8 serving paths (hardware-gated)
- [ ] n-gram + draft-model speculative decoding
- [ ] Scheduler policies: FCFS, SJF, priority-aware preemption

### Exit Criteria
- [ ] `>= 1.0x` vLLM throughput on ShareGPT-like serving benchmark
- [ ] TTFT P50 `<= 15ms` for 128-token prompt
- [ ] ITL P50 `<= 10ms` on interactive workload
- [ ] Prefix-cache-heavy workload shows measurable compute savings

---

## Phase 3 - Production Hardening (Q3 2026)
Goal: enterprise-ready single-GPU product.

### Must-have
- [ ] Graceful drain + in-flight request cancellation
- [ ] Rate limits, API keys, audit-grade structured logs
- [ ] Config validation + safe startup checks (`doctor`)
- [ ] Zero-downtime model reload flow
- [ ] Metrics/traces/logs with request correlation IDs
- [ ] Regression gates in CI for latency and throughput

### Exit Criteria
- [ ] 99.9% successful request completion in stress tests
- [ ] Predictable P99 latency under 2x expected peak
- [ ] Release packaging and runbooks suitable for production teams

---

## Phase 4 - Single-Node Multi-GPU (Q4 2026)
Goal: scale up on one machine before scaling out.

### Must-have
- [ ] Tensor parallelism (2/4/8 GPU topologies)
- [ ] Multi-LoRA serving with bounded overhead
- [ ] Smarter admission control aware of per-device KV pressure
- [ ] Cross-device KV and activation transfer instrumentation

### Exit Criteria
- [ ] Near-linear scaling to at least 4 GPUs on target models
- [ ] Multi-LoRA throughput overhead under agreed SLO budget

---

## Phase 5 - Cluster Mode (Future)
Goal: disaggregated prefill/decode and resilient multi-node serving.

### Must-have
- [ ] Prefill and decode pools with remote KV handoff
- [ ] Distributed prefix cache coordination
- [ ] Load-aware request router and admission controller
- [ ] Node failure handling with graceful request recovery

### Exit Criteria
- [ ] Cluster remains available after single-node failures
- [ ] Horizontal scale increases throughput with controlled latency growth

---

## Phase 6 - AMD Expansion (Future)
Goal: preserve core architecture and add ROCm backend.

### Must-have
- [ ] Backend interface parity: CUDA and ROCm implementations
- [ ] HIP/ROCm attention + KV kernels
- [ ] Quantization and scheduling parity checks across vendors
- [ ] Cross-vendor benchmark matrix in CI

### Exit Criteria
- [ ] Feature parity for core API and scheduler semantics on AMD
- [ ] Published perf/latency comparisons vs NVIDIA path

---

## Scope Guardrails (v1)
- No training/fine-tuning platform.
- No dependence on Python runtime at serve time.
- No premature cluster abstractions before single-node goals are hit.

---

## Competitive Scorecard

| Dimension | gollmgo target | vLLM today |
|---|---|---|
| Serving API compatibility | OpenAI-compatible + operator-focused extensions | OpenAI-compatible |
| Single GPU throughput | Parity or better | Strong baseline |
| Operational simplicity | Single binary + minimal runtime deps | Python stack |
| Scheduler sophistication | Continuous batching + preemption + chunked prefill | Mature |
| Multi-LoRA and speculative decode | First-class | First-class |
| Multi-node architecture | Planned after single-node win | Available ecosystem |
