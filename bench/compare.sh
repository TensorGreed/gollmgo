#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "error: required command not found: $1" >&2
        exit 1
    }
}

AGAINST=""
MODEL=""
CONFIG="bench/baseline_config.json"
BINARY="${GOLLMGO_BIN:-bin/gollmgo}"
G_PORT="${GOLLMGO_COMPARE_PORT:-8080}"
V_PORT="${VLLM_COMPARE_PORT:-8000}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
RESULT_DIR="bench/results"
SUMMARY="bench/PARITY_SUMMARY.md"
CONTAINER_NAME="gollmgo-vllm-parity"
G_PID=""

while [ $# -gt 0 ]; do
    case "$1" in
        --against)
            AGAINST="${2:-}"
            shift 2
            ;;
        --model)
            MODEL="${2:-}"
            shift 2
            ;;
        --config)
            CONFIG="${2:-}"
            shift 2
            ;;
        --summary)
            SUMMARY="${2:-}"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ "$AGAINST" != "vllm" ]; then
    echo "error: only --against vllm is currently supported" >&2
    exit 1
fi
if [ -z "$MODEL" ]; then
    echo "error: --model is required and must point to a local model path for vLLM parity" >&2
    exit 1
fi
if [ ! -e "$MODEL" ]; then
    echo "error: model path does not exist: $MODEL" >&2
    exit 1
fi
if [ ! -x "$BINARY" ]; then
    echo "error: $BINARY not found or not executable. Run 'make build' first." >&2
    exit 1
fi

need_cmd curl
need_cmd docker
need_cmd jq

mkdir -p "$RESULT_DIR"

if [ -d "$MODEL" ]; then
    MODEL_DIR="$MODEL"
    G_MODEL="$MODEL"
else
    MODEL_DIR="$(dirname "$MODEL")"
    G_MODEL="$MODEL"
fi

MAX_MODEL_LEN=2048
if [ -f "$MODEL_DIR/config.json" ]; then
    MAX_MODEL_LEN="$(jq -r '.max_position_embeddings // .n_positions // 2048' "$MODEL_DIR/config.json")"
fi

G_RESULT="$RESULT_DIR/gollmgo_parity.json"
V_RESULT="$RESULT_DIR/vllm_parity.json"
G_LOG="$RESULT_DIR/gollmgo_parity_server.log"
V_LOG="$RESULT_DIR/vllm_parity_server.log"

cleanup() {
    if [ -n "$G_PID" ] && kill -0 "$G_PID" >/dev/null 2>&1; then
        kill "$G_PID" >/dev/null 2>&1 || true
        wait "$G_PID" 2>/dev/null || true
    fi
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

wait_for_http() {
    local url="$1"
    local name="$2"
    for i in $(seq 1 120); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "error: $name did not become ready: $url" >&2
    return 1
}

echo "compare: starting gollmgo"
"$BINARY" serve --config "$CONFIG" --model "$G_MODEL" >"$G_LOG" 2>&1 &
G_PID=$!
wait_for_http "http://127.0.0.1:${G_PORT}/health/ready" "gollmgo"
"$BINARY" bench --mode serving --config "$CONFIG" --url "http://127.0.0.1:${G_PORT}" --model "$G_MODEL" >"$G_RESULT"
kill "$G_PID" >/dev/null 2>&1 || true
wait "$G_PID" 2>/dev/null || true
G_PID=""

echo "compare: starting vLLM"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d --rm \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --runtime nvidia \
    --ipc=host \
    -p "${V_PORT}:8000" \
    -v "${MODEL_DIR}:/model:ro" \
    "$VLLM_IMAGE" \
    --model /model \
    --served-model-name gollmgo-default \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype "$VLLM_DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    >/dev/null

wait_for_http "http://127.0.0.1:${V_PORT}/v1/models" "vLLM"
docker logs "$CONTAINER_NAME" >"$V_LOG" 2>&1 || true
"$BINARY" bench --mode serving --config "$CONFIG" --url "http://127.0.0.1:${V_PORT}" --model "$G_MODEL" >"$V_RESULT"
docker logs "$CONTAINER_NAME" >>"$V_LOG" 2>&1 || true
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

bash "$SCRIPT_DIR/render_parity_summary.sh" "$V_RESULT" "$G_RESULT" "$SUMMARY"

echo "compare: wrote"
echo "  $G_RESULT"
echo "  $V_RESULT"
echo "  $SUMMARY"
