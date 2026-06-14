#!/usr/bin/env bash
# Start the standalone LMCache server (L1 CPU cache, LRU eviction)
# HTTP management: http://localhost:8080  (POST /clear-cache, GET /metrics)
# ZMQ data plane:  tcp://localhost:5555   (vLLM connects here)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$ROOT/.venv"

if [ ! -f "$VENV/bin/lmcache" ]; then
    echo "ERROR: venv not found or lmcache not installed. Run ./setup/install.sh first." >&2
    exit 1
fi

L1_SIZE="${LMCACHE_L1_SIZE_GB:-20}"

exec "$VENV/bin/lmcache" server \
    --l1-size-gb "$L1_SIZE" \
    --eviction-policy LRU \
    --chunk-size 16
