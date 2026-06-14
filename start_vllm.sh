#!/usr/bin/env bash
# Start vLLM with LMCache MP connector pointing at the standalone lmcache server.
# Prerequisite: start_lmcache.sh must be running first (ZMQ on tcp://localhost:5555).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

if [ ! -f "$VENV/bin/vllm" ]; then
    echo "ERROR: venv not found or vllm not installed. Run ./install.sh first." >&2
    exit 1
fi

MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-8001}"
MAX_LEN="${VLLM_MAX_LEN:-8192}"
LMCACHE_HOST="${LMCACHE_HOST:-tcp://localhost}"
LMCACHE_PORT="${LMCACHE_PORT:-5555}"

# GB10 Grace Blackwell has unified CPU+GPU memory (121 GiB total).
# nvidia-smi reports N/A for free memory on this chip, so we can't auto-detect.
# Default 0.24 (~29 GiB) fits alongside the gemma server which holds ~54 GiB.
# Set VLLM_GPU_MEM=0.xx to override (e.g. 0.85 when running Qwen alone).
GPU_MEM="${VLLM_GPU_MEM:-0.24}"

KV_CONFIG=$(printf '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"%s","lmcache.mp.port":%s}}' \
    "$LMCACHE_HOST" "$LMCACHE_PORT")

# Wait for LMCache ZMQ to be ready before connecting
echo "Waiting for LMCache server on $LMCACHE_HOST:$LMCACHE_PORT ..."
for i in $(seq 1 30); do
    if "$VENV/bin/python3" -c "
import socket, sys
host = '${LMCACHE_HOST}'.replace('tcp://', '')
try:
    s = socket.create_connection((host, ${LMCACHE_PORT}), timeout=1)
    s.close(); sys.exit(0)
except: sys.exit(1)
" 2>/dev/null; then
        echo "LMCache is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: LMCache not reachable after 30s. Run ./start_lmcache.sh first." >&2
        exit 1
    fi
    sleep 1
done

echo "Starting vLLM"
echo "  model            : $MODEL"
echo "  port             : $PORT"
echo "  gpu-mem-util     : $GPU_MEM  (set VLLM_GPU_MEM=0.xx to override)"
echo "  max-model-len    : $MAX_LEN"
echo "  lmcache server   : $LMCACHE_HOST:$LMCACHE_PORT"
echo ""

exec "$VENV/bin/vllm" serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_LEN" \
    --kv-transfer-config "$KV_CONFIG"
