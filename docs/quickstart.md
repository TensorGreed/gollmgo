# gollmgo - Quickstart

## Target Platform
This project is optimized first for `NVIDIA DGX Spark (GB10 Grace Blackwell superchip)`.

Secondary development may work on other CUDA GPUs, but all baseline numbers and tuning presets are anchored to DGX Spark.

---

## Prerequisites
- Linux
- NVIDIA driver and CUDA toolkit available in shell (`nvidia-smi`, `nvcc --version`)
- Go 1.22+
- `make`, `gcc/g++`, and build essentials

Not in v1: Windows/macOS serving path and ROCm runtime.

---

## 1. Build

```bash
git clone https://github.com/TensorGreed/gollmgo
cd gollmgo

make kernels
make build
./build/gollmgo --version
```

If build fails:
- `nvcc` missing: add CUDA toolkit to `PATH`.
- CGo disabled: ensure `CGO_ENABLED=1`.
- link errors: verify CUDA runtime/libs are installed.

---

## 2. Run Environment Check

```bash
./build/gollmgo doctor
```

Expected checks:
- CUDA runtime compatibility
- GPU capability detection
- free memory estimate for model + KV cache
- supported kernel path selection

---

## 3. Acquire a Model

```bash
./build/gollmgo download meta-llama/Meta-Llama-3-8B-Instruct --dir /models
```

Or point to an existing local safetensors/GGUF model path.

---

## 4. Create Config

```bash
cp config.example.yaml config.yaml
```

Minimal config:
```yaml
server:
  port: 8080
model:
  path: /models/Meta-Llama-3-8B-Instruct
scheduler:
  max_batch_size: 128
kvcache:
  max_memory_fraction: 0.90
```

Use `docs/configuration.md` for the full reference.

---

## 5. Start Server

```bash
./build/gollmgo serve --config config.yaml
```

Startup should report:
- detected GPU and backend path
- model load summary
- KV cache sizing
- listening address

---

## 6. Send a Request

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b-instruct",
    "messages": [{"role": "user", "content": "Write a haiku about compilers."}],
    "stream": false
  }'
```

Streaming:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b-instruct",
    "messages": [{"role": "user", "content": "Count to 10."}],
    "stream": true
  }'
```

---

## 7. Observe Runtime Health

```bash
curl http://localhost:8080/health/ready
curl http://localhost:8080/metrics | grep gollmgo
```

Watch:
- throughput
- TTFT/ITL histograms
- queue depth
- KV utilization and prefix cache hit rate

---

## 8. Benchmark

```bash
make bench BENCH=serving MODEL=/models/Meta-Llama-3-8B-Instruct DATASET=sharegpt QPS=10
```

For parity checks:
```bash
./bench/compare.sh --against vllm --model /models/Meta-Llama-3-8B-Instruct --dataset sharegpt
```

---

## Common Issues

### OOM at startup
Lower `kvcache.max_memory_fraction` and/or `scheduler.max_batch_size`.

### High first-request latency
Warmup and graph capture cost is expected. Run warmup before opening traffic.

### Bad tokenization output
Confirm model/tokenizer pair in config and model metadata.

### Throughput lower than expected
Check batch fill, prefill budget, graph capture status, and kernel path selection.
