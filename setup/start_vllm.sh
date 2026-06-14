#!/usr/bin/env bash
# Start vLLM with LMCache MP connector pointing at the standalone lmcache server.
# Prerequisite: start_lmcache.sh must be running first (ZMQ on tcp://localhost:5555).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$ROOT/.venv"

if [ ! -f "$VENV/bin/vllm" ]; then
    echo "ERROR: venv not found or vllm not installed. Run ./setup/install.sh first." >&2
    exit 1
fi

# Kill stale EngineCore processes (survive failed runs as orphan root processes holding GPU memory).
# When found, also restart LMCache — each failed vLLM run leaves stale IPC KV-cache mappings
# in the LMCache process that accumulate and exhaust the profiling memory budget.
STALE=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
    | grep -i "EngineCore" | awk -F',' '{print $1}' | tr -d ' ' || true)
if [ -n "$STALE" ]; then
    echo "Killing stale EngineCore GPU process(es): $STALE"
    sudo kill -9 $STALE || true
    echo "Restarting LMCache to clear stale IPC mappings..."
    pkill -9 -f "$VENV/bin/lmcache" 2>/dev/null || true
    sleep 3
    "$VENV/bin/lmcache" server \
        --l1-size-gb "${LMCACHE_L1_SIZE_GB:-10}" \
        --eviction-policy LRU \
        --chunk-size 16 &
    LMCACHE_PID=$!
    echo "LMCache restarted (PID $LMCACHE_PID)"
fi

MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-8001}"
MAX_LEN="${VLLM_MAX_LEN:-8192}"
LMCACHE_HOST="${LMCACHE_HOST:-tcp://localhost}"
LMCACHE_PORT="${LMCACHE_PORT:-5555}"

# GB10 Grace Blackwell has unified CPU+GPU memory (121.6 GiB total).
# nvidia-smi reports N/A for free memory on this chip, but torch.cuda.mem_get_info()
# works correctly. Auto-detect free memory and use 90% of it to leave headroom.
# Override with VLLM_GPU_MEM=0.xx to skip detection.
if [ -n "${VLLM_GPU_MEM:-}" ]; then
    GPU_MEM="$VLLM_GPU_MEM"
else
    GPU_MEM=$("$VENV/bin/python3" -c "
import torch
free, total = torch.cuda.mem_get_info()
util = (free / total) * 0.90
print(f'{util:.2f}')
" 2>/dev/null || echo "0.75")
    echo "  auto-detected gpu-mem-util: $GPU_MEM  (90% of current free memory)"
fi

KV_CONFIG=$(printf '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"%s","lmcache.mp.port":%s}}' \
    "$LMCACHE_HOST" "$LMCACHE_PORT")

# Wait for LMCache HTTP management API (port 8080) — never probe ZMQ port directly;
# raw TCP on a ZMQ ROUTER socket sends a zero-part message and crashes the mq-server-thread.
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
echo "Waiting for LMCache server on http://127.0.0.1:$LMCACHE_HTTP_PORT ..."
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$LMCACHE_HTTP_PORT/" > /dev/null 2>&1; then
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

# Put venv/bin first so subprocesses (flashinfer JIT, ninja) can find venv tools
export PATH="$VENV/bin:$PATH"

exec env VLLM_ENGINE_READY_TIMEOUT_S=900 "$VENV/bin/vllm" serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_LEN" \
    --kv-transfer-config "$KV_CONFIG" \
    --override-generation-config '{"enable_thinking": false}'
