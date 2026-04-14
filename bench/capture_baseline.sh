#!/usr/bin/env bash
# capture_baseline.sh — M8 baseline capture.
#
# Starts gollmgo, waits for readiness, runs the serving benchmark, strips
# the placeholder marker, and installs the result as bench/baseline_result.json.
#
# Usage:
#   # Mock-runner baseline (no model weights needed — smoke-tests the full loop):
#   bench/capture_baseline.sh
#
#   # Real model baseline on the DGX:
#   GOLLMGO_BENCH_MODEL_PATH=/path/to/weights bench/capture_baseline.sh
#
# Environment:
#   GOLLMGO_BENCH_MODEL_PATH  — model file or directory for `serve --model`.
#                               If unset, the mock runner is used.
#   GOLLMGO_BENCH_CONFIG      — server config (default: bench/baseline_config.json).
#   GOLLMGO_BENCH_URL         — server URL (default: http://localhost:8080).
#   GOLLMGO_BENCH_OUTPUT      — output path (default: bench/baseline_result.json).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${GOLLMGO_BENCH_CONFIG:-bench/baseline_config.json}"
URL="${GOLLMGO_BENCH_URL:-http://localhost:8080}"
OUTPUT="${GOLLMGO_BENCH_OUTPUT:-bench/baseline_result.json}"
BINARY="${GOLLMGO_BIN:-bin/gollmgo}"

if [ ! -x "$BINARY" ]; then
    echo "error: $BINARY not found or not executable. Run 'make build' first." >&2
    exit 1
fi

mkdir -p bench/results

# Build server command. Adds --model if GOLLMGO_BENCH_MODEL_PATH is set.
SERVE_ARGS=(serve --config "$CONFIG")
if [ -n "${GOLLMGO_BENCH_MODEL_PATH:-}" ]; then
    SERVE_ARGS+=(--model "$GOLLMGO_BENCH_MODEL_PATH")
    echo "capture_baseline: using real model at $GOLLMGO_BENCH_MODEL_PATH" >&2
else
    echo "capture_baseline: no GOLLMGO_BENCH_MODEL_PATH set, using mock runner" >&2
fi

LOG="$REPO_ROOT/bench/results/server.log"
PID_FILE="$REPO_ROOT/bench/results/server.pid"

# Spin up the server.
"$BINARY" "${SERVE_ARGS[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
trap 'kill "$SERVER_PID" 2>/dev/null || true; rm -f "$PID_FILE"' EXIT

echo "capture_baseline: server PID=$SERVER_PID; waiting for $URL/health/ready"
for i in $(seq 1 60); do
    if curl -fsS "$URL/health/ready" >/dev/null 2>&1; then
        echo "capture_baseline: server is ready"
        break
    fi
    if [ "$i" = "60" ]; then
        echo "error: server did not become ready within 60s. Tail of log:" >&2
        tail -n 40 "$LOG" >&2 || true
        exit 1
    fi
    sleep 1
done

# Run the serving benchmark. tee to stdout so the user sees progress.
TMPDIR="$(mktemp -d)"
RAW="$TMPDIR/raw.json"
"$BINARY" bench --mode serving --config "$CONFIG" --url "$URL" > "$RAW"

# Install as the frozen baseline, stripping placeholder markers so the
# regression gate will actually run against it.
if command -v jq >/dev/null 2>&1; then
    jq 'del(.placeholder, ._description)' "$RAW" > "$OUTPUT"
else
    echo "capture_baseline: jq not available — copying raw output verbatim" >&2
    echo "                  You may need to manually remove .placeholder/._description." >&2
    cp "$RAW" "$OUTPUT"
fi

echo "capture_baseline: wrote $OUTPUT"
echo "---"
cat "$OUTPUT"
echo "---"
echo "capture_baseline: done. The regression gate will now run against this baseline."
