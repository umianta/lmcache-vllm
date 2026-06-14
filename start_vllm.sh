#!/usr/bin/env bash
# Start vLLM with LMCache MP connector pointing at the standalone lmcache server.
# Prerequisite: start_lmcache.sh must be running first (ZMQ on tcp://localhost:5555).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

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

echo "Starting vLLM"
echo "  model            : $MODEL"
echo "  port             : $PORT"
echo "  gpu-mem-util     : $GPU_MEM  (set VLLM_GPU_MEM=0.xx to override)"
echo "  max-model-len    : $MAX_LEN"
echo "  lmcache server   : $LMCACHE_HOST:$LMCACHE_PORT"
echo ""

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_LEN" \
    --kv-transfer-config "$KV_CONFIG"
