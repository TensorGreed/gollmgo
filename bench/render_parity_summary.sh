#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <baseline.json> <current.json> [output.md]" >&2
    exit 1
fi

BASELINE="$1"
CURRENT="$2"
OUTPUT="${3:-bench/PARITY_SUMMARY.md}"
CONFIG="${CONFIG:-bench/baseline_config.json}"

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "error: required command not found: $1" >&2
        exit 1
    }
}

need_cmd jq

metric() {
    local file="$1"
    local key="$2"
    jq -r "$key" "$file"
}

delta_pct() {
    jq -nr --argjson base "$1" --argjson cur "$2" '
        if $base == 0 then 0 else (($cur - $base) / $base) * 100 end
    '
}

status_less_is_better() {
    jq -nr --argjson base "$1" --argjson cur "$2" '
        if $cur <= $base then "PASS" else "FAIL" end
    '
}

status_greater_is_better() {
    jq -nr --argjson base "$1" --argjson cur "$2" '
        if $cur >= $base then "PASS" else "FAIL" end
    '
}

fmt_delta() {
    jq -nr --argjson v "$1" 'if $v >= 0 then "+" + (($v * 100 | round) / 100 | tostring) + "%" else (($v * 100 | round) / 100 | tostring) + "%" end'
}

BASE_TPS="$(metric "$BASELINE" '.tokens_per_second')"
CUR_TPS="$(metric "$CURRENT" '.tokens_per_second')"
BASE_TTFT50="$(metric "$BASELINE" '.ttft_p50_ms')"
CUR_TTFT50="$(metric "$CURRENT" '.ttft_p50_ms')"
BASE_TTFT99="$(metric "$BASELINE" '.ttft_p99_ms')"
CUR_TTFT99="$(metric "$CURRENT" '.ttft_p99_ms')"
BASE_ITL50="$(metric "$BASELINE" '.itl_p50_ms')"
CUR_ITL50="$(metric "$CURRENT" '.itl_p50_ms')"
BASE_ITL99="$(metric "$BASELINE" '.itl_p99_ms')"
CUR_ITL99="$(metric "$CURRENT" '.itl_p99_ms')"
BASE_ERR="$(metric "$BASELINE" '.error_count')"
CUR_ERR="$(metric "$CURRENT" '.error_count')"

TPS_DELTA="$(delta_pct "$BASE_TPS" "$CUR_TPS")"
TTFT50_DELTA="$(delta_pct "$BASE_TTFT50" "$CUR_TTFT50")"
TTFT99_DELTA="$(delta_pct "$BASE_TTFT99" "$CUR_TTFT99")"
ITL50_DELTA="$(delta_pct "$BASE_ITL50" "$CUR_ITL50")"
ITL99_DELTA="$(delta_pct "$BASE_ITL99" "$CUR_ITL99")"
ERR_DELTA="$(delta_pct "$BASE_ERR" "$CUR_ERR")"

TPS_STATUS="$(status_greater_is_better "$BASE_TPS" "$CUR_TPS")"
TTFT50_STATUS="$(status_less_is_better "$BASE_TTFT50" "$CUR_TTFT50")"
TTFT99_STATUS="$(status_less_is_better "$BASE_TTFT99" "$CUR_TTFT99")"
ITL50_STATUS="$(status_less_is_better "$BASE_ITL50" "$CUR_ITL50")"
ITL99_STATUS="$(status_less_is_better "$BASE_ITL99" "$CUR_ITL99")"
ERR_STATUS="$(status_less_is_better "$BASE_ERR" "$CUR_ERR")"

SHA="$(metric "$CURRENT" '.git_sha')"
DATE_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
PROMPT_MODE="$(metric "$CURRENT" '.prompt_generation // "unknown"')"
TOKEN_MODE="$(metric "$CURRENT" '.token_count_mode // "unknown"')"
REPETITIONS="$(metric "$CURRENT" '.repetitions // 1')"
WARMUP="$(metric "$CURRENT" '.warmup_requests // 0')"
ARRIVAL_MODE="$(metric "$CURRENT" '.arrival_mode // "closed_loop"')"

THROUGHPUT_GATE="$(jq -nr --argjson cur "$CUR_TPS" --argjson base "$BASE_TPS" 'if $base > 0 and ($cur / $base) >= 1 then "[x]" else "[ ]" end')"
TTFT_GATE="$(jq -nr --argjson cur "$CUR_TTFT50" 'if $cur <= 15 then "[x]" else "[ ]" end')"
ITL_GATE="$(jq -nr --argjson cur "$CUR_ITL50" 'if $cur <= 10 then "[x]" else "[ ]" end')"
REGRESSION_GATE="$(jq -nr --argjson cur "$CUR_ERR" 'if $cur == 0 then "[x]" else "[ ]" end')"

cat >"$OUTPUT" <<EOF
# Parity Summary

## Environment
- Hardware: DGX Spark GB10
- Git SHA: $SHA
- Date: $DATE_UTC
- Config: $CONFIG
- Arrival mode: $ARRIVAL_MODE
- Warmup requests: $WARMUP
- Repetitions: $REPETITIONS
- Prompt generation: $PROMPT_MODE
- Token count mode: $TOKEN_MODE

## Results vs Baseline

| Metric | Baseline | Current | Delta | Status |
|--------|----------|---------|-------|--------|
| Throughput (tok/s) | $BASE_TPS | $CUR_TPS | $(fmt_delta "$TPS_DELTA") | $TPS_STATUS |
| TTFT P50 (ms) | $BASE_TTFT50 | $CUR_TTFT50 | $(fmt_delta "$TTFT50_DELTA") | $TTFT50_STATUS |
| TTFT P99 (ms) | $BASE_TTFT99 | $CUR_TTFT99 | $(fmt_delta "$TTFT99_DELTA") | $TTFT99_STATUS |
| ITL P50 (ms) | $BASE_ITL50 | $CUR_ITL50 | $(fmt_delta "$ITL50_DELTA") | $ITL50_STATUS |
| ITL P99 (ms) | $BASE_ITL99 | $CUR_ITL99 | $(fmt_delta "$ITL99_DELTA") | $ITL99_STATUS |
| Error Count | $BASE_ERR | $CUR_ERR | $(fmt_delta "$ERR_DELTA") | $ERR_STATUS |

## Exit Criteria
- $THROUGHPUT_GATE >= 1.0x vLLM throughput on primary workload
- $TTFT_GATE TTFT P50 <= 15ms
- $ITL_GATE ITL P50 <= 10ms
- $REGRESSION_GATE zero benchmark errors in the measured run

## Notes
- Baseline file: $BASELINE
- Current file: $CURRENT
EOF

echo "render_parity_summary: wrote $OUTPUT"
