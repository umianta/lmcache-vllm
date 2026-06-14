#!/usr/bin/env bash
# Start the standalone LMCache server (L1 CPU cache, LRU eviction)
# HTTP management: http://localhost:8080  (POST /clear-cache, GET /metrics)
# ZMQ data plane:  tcp://localhost:5555   (vLLM connects here)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

if [ ! -f "$VENV/bin/lmcache" ]; then
    echo "ERROR: venv not found or lmcache not installed. Run ./install.sh first." >&2
    exit 1
fi

exec "$VENV/bin/lmcache" server \
    --l1-size-gb 20 \
    --eviction-policy LRU \
    --chunk-size 16
