#!/usr/bin/env bash
set -euo pipefail

# Usage: bench/check_regression.sh <current_result.json>
# Compares against bench/baseline_result.json using bench/thresholds.json
# Exit 0 = pass, exit 1 = regression detected

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <current_result.json>" >&2
    exit 1
fi

CURRENT="$1"
BASELINE="${BASELINE:-$REPO_ROOT/bench/baseline_result.json}"
THRESHOLDS="${THRESHOLDS:-$REPO_ROOT/bench/thresholds.json}"

exec go run "$REPO_ROOT/internal/benchcheck/cmd/main.go" \
    --baseline "$BASELINE" \
    --current "$CURRENT" \
    --thresholds "$THRESHOLDS"
