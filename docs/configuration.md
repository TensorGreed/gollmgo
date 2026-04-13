# gollmgo - Configuration Reference

Configuration source precedence:
1. CLI flags
2. Environment variables (`GOLLMGO_*`)
3. `config.yaml`

Environment override format: `GOLLMGO_<SECTION>_<KEY>`.
Example: `GOLLMGO_SERVER_PORT=9090`.

---

## Server

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  read_timeout: 30s
  write_timeout: 0s          # keep 0 for streaming responses
  shutdown_timeout: 30s
  max_request_size: 4MB
```

---

## Model

```yaml
model:
  path: "/models/Meta-Llama-3-8B-Instruct"
  dtype: "auto"             # auto | fp16 | bf16 | fp8 | int8 | gguf
  max_seq_len: 8192
  revision: "main"
  trust_remote_code: false
```

Notes:
- `auto` chooses a safe default based on model format and backend capability.
- `fp8` and `int8` require validated backend support and calibration metadata.

---

## Backend

```yaml
backend:
  provider: "cuda"          # cuda | rocm (future)
  device_ids: [0]
  stream_count: 1
  enable_cuda_graphs: true
  cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128]
  warmup_on_start: true
```

For DGX Spark single-GPU bring-up, keep `device_ids: [0]` and `stream_count: 1` unless profiling shows gains.

---

## KV Cache

```yaml
kvcache:
  block_size: 16
  max_memory_fraction: 0.90
  prefix_caching: true
  prefix_cache_max_blocks: 1024
  cpu_offload: false
  cpu_swap_space: 16GB
```

Tuning:
- Smaller `block_size` improves packing for short prompts.
- Larger `block_size` may improve kernel efficiency for long contexts.
- Enable `cpu_offload` only when preemption pressure justifies it.

---

## Scheduler

```yaml
scheduler:
  max_batch_size: 128
  max_prefill_tokens_per_step: 2048
  preemption_mode: "recompute"    # recompute | swap
  priority_policy: "fcfs"         # fcfs | sjf | priority_header
  tick_interval: 5ms
  max_waiting_sequences: 1000
```

Guidance:
- Increase `max_prefill_tokens_per_step` when TTFT is too high.
- Decrease it when decode latency degrades under mixed workloads.

---

## Attention / Decode

```yaml
attention:
  backend: "auto"            # auto | paged_v1 | paged_v2 | vendor_optimized
  prefer_long_context_kernel: true

speculative:
  enabled: false
  mode: "ngram"              # ngram | draft_model
  ngram_size: 3
  num_draft_tokens: 4
```

---

## Sampling Defaults

```yaml
sampling:
  temperature: 1.0
  top_p: 1.0
  top_k: -1
  max_tokens: 512
  repetition_penalty: 1.0
  stop: []
```

---

## Observability

```yaml
metrics:
  enabled: true
  path: "/metrics"
  pprof: false

tracing:
  enabled: false
  exporter: "otlp"
  endpoint: ""

logging:
  level: "info"
  format: "json"
  request_log: true
```

---

## Security / Access

```yaml
auth:
  enabled: false
  keys: []

limits:
  requests_per_minute: 0
  tokens_per_minute: 0
```

`0` means unlimited.

---

## Preset Profiles

### DGX Spark Interactive
```yaml
model:
  dtype: bf16
scheduler:
  max_batch_size: 64
  max_prefill_tokens_per_step: 1024
kvcache:
  block_size: 16
  max_memory_fraction: 0.88
```

### DGX Spark Throughput
```yaml
model:
  dtype: fp8
scheduler:
  max_batch_size: 128
  max_prefill_tokens_per_step: 4096
kvcache:
  block_size: 32
  max_memory_fraction: 0.92
speculative:
  enabled: true
  mode: ngram
```

---

## Validation Rules
- Startup fails fast on invalid enums or impossible memory budgets.
- `max_batch_size` must match backend preallocation limits.
- `swap` preemption requires `kvcache.cpu_offload: true`.
