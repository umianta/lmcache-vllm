#!/usr/bin/env bash
# LMCache + vLLM observability dashboard
# Usage: ./observe.sh [interval_seconds]   (default: 3)

INTERVAL="${1:-3}"
VLLM_PORT="${VLLM_PORT:-8001}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"

BOLD='\033[1m'; DIM='\033[2m'; CYAN='\033[36m'; GREEN='\033[32m'
YELLOW='\033[33m'; RED='\033[31m'; BLUE='\033[34m'; RESET='\033[0m'

sep()    { printf "${DIM}  %-50s${RESET}\n" "$(printf '%.0s─' {1..50})"; }
header() { printf "\n${BOLD}${CYAN}▸ %s${RESET}\n" "$1"; sep; }
row()    { printf "  ${DIM}%-30s${RESET} %s\n" "$1" "$2"; }

# Extract a metric value (handles labeled and unlabeled metrics)
# Usage: _m "$raw" "metric_name"              → first matching value
#        _m "$raw" "metric_name" "label=val"  → value where line contains that label
_m() {
    local raw="$1" name="$2" filter="${3:-}"
    if [ -n "$filter" ]; then
        echo "$raw" | grep "^${name}{" | grep "$filter" | awk '{print $NF}' | head -1
    else
        echo "$raw" | grep -E "^${name}(\{| )" | awk '{print $NF}' | head -1
    fi
}

# Sum all values for a metric across all label combinations
_sum_label() {
    local raw="$1" name="$2"
    echo "$raw" | grep "^${name}{" | awk '{s+=$NF} END{printf "%.0f", s}'
}

# Average from histogram _sum / _count
_hist_avg() {
    local raw="$1" name="$2"
    local s c
    s=$(echo "$raw" | grep "^${name}_sum"  | awk '{print $NF}' | head -1)
    c=$(echo "$raw" | grep "^${name}_count" | awk '{print $NF}' | head -1)
    echo "${s:-0} ${c:-0}" | awk '{if($2>0) printf "%.3f", $1/$2; else print "—"}'
}

# Format bytes to human readable
_bytes() { echo "${1:-0}" | awk '{
    if($1>=1073741824) printf "%.2f GB", $1/1073741824
    else if($1>=1048576) printf "%.1f MB", $1/1048576
    else if($1>=1024) printf "%.0f KB", $1/1024
    else printf "%.0f B", $1
}'; }

# Hit rate % from two values
_rate() { echo "${1:-0} ${2:-0}" | awk '{t=$1+$2; if(t>0) printf "%.1f%%", $1/t*100; else print "—"}'; }

gpu_section() {
    header "GPU  (nvidia-smi)"
    local proc gpu
    proc=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
           --format=csv,noheader 2>/dev/null) || { echo "  nvidia-smi unavailable"; return; }
    gpu=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total \
          --format=csv,noheader 2>/dev/null)

    while IFS=',' read -r pid pname mem; do
        row "Process [PID $pid]" "$(echo "$pname" | xargs) — ${BOLD}$(echo "$mem" | xargs)${RESET}"
    done <<< "$proc"

    echo "$gpu" | awk -F',' '{
        printf "  \033[2m%-30s\033[0m %s°C  %s  util %s  VRAM %s / %s\n",
        "Hardware", $1, $2, $3, $4, $5}'
}

vllm_section() {
    header "vLLM  (port $VLLM_PORT)"
    local raw
    raw=$(curl -sf "http://localhost:$VLLM_PORT/metrics" 2>/dev/null) || {
        row "Status" "${RED}unreachable${RESET}"; return; }

    # Requests
    local running waiting preempt success_stop success_len success_err
    running=$(_m "$raw" "vllm:num_requests_running")
    waiting=$(_m "$raw" "vllm:num_requests_waiting")
    preempt=$(_m "$raw" "vllm:num_preemptions_total")
    success_stop=$(_m "$raw" "vllm:request_success_total" 'finished_reason="stop"')
    success_len=$( _m "$raw" "vllm:request_success_total" 'finished_reason="length"')
    success_err=$( _m "$raw" "vllm:request_success_total" 'finished_reason="error"')

    row "Requests running" "${GREEN}${running:-0}${RESET}  waiting: ${YELLOW}${waiting:-0}${RESET}  preempted: ${preempt:-0}"
    row "Completed" "stop: ${success_stop:-0}  length: ${success_len:-0}  error: ${RED}${success_err:-0}${RESET}"

    # Throughput
    local ptok gtok
    ptok=$(_m "$raw" "vllm:avg_prompt_throughput_toks_per_s")
    gtok=$(_m "$raw" "vllm:avg_generation_throughput_toks_per_s")
    row "Throughput" "prompt: ${ptok:-0} tok/s   gen: ${gtok:-0} tok/s"

    # Token totals
    local total_prompt total_cached total_gen
    total_prompt=$(_m "$raw" "vllm:prompt_tokens_total")
    total_cached=$(_m "$raw" "vllm:prompt_tokens_cached_total")
    total_gen=$(_m "$raw" "vllm:generation_tokens_total")
    row "Tokens (cumulative)" "prompt: ${total_prompt:-0}  cached: ${total_cached:-0}  gen: ${total_gen:-0}"

    # Token sources
    local src_compute src_local src_external
    src_compute=$( _m "$raw" "vllm:prompt_tokens_by_source_total" 'source="local_compute"')
    src_local=$(   _m "$raw" "vllm:prompt_tokens_by_source_total" 'source="local_cache_hit"')
    src_external=$(  _m "$raw" "vllm:prompt_tokens_by_source_total" 'source="external_kv_transfer"')
    row "Token sources" "computed: ${src_compute:-0}  local-hit: ${src_local:-0}  ext-transfer: ${src_external:-0}"

    # KV cache
    local kv_pct gpu_blocks block_size
    kv_pct=$(   _m "$raw" "vllm:kv_cache_usage_perc")
    gpu_blocks=$(_m "$raw" "vllm:cache_config_info" 'num_gpu_blocks=')
    block_size=$( echo "$raw" | grep "vllm:cache_config_info" | grep -o 'block_size="[0-9]*"' | head -1 | grep -o '[0-9]*')
    row "KV cache" "usage: $(echo "${kv_pct:-0}" | awk '{printf "%.1f%%", $1*100}')  gpu-blocks: ${gpu_blocks:-?}  block-size: ${block_size:-?} tok"

    # Prefix cache (local)
    local pc_q pc_h
    pc_q=$(_m "$raw" "vllm:prefix_cache_queries_total")
    pc_h=$(_m "$raw" "vllm:prefix_cache_hits_total")
    row "Prefix cache (local)" "queries: ${pc_q:-0}  hits: ${pc_h:-0}  rate: $(_rate "${pc_h:-0}" "$(echo "${pc_q:-0} ${pc_h:-0}" | awk '{printf "%.0f", $1-$2}')")"

    # External prefix cache (LMCache)
    local ec_q ec_h
    ec_q=$(_m "$raw" "vllm:external_prefix_cache_queries_total")
    ec_h=$(_m "$raw" "vllm:external_prefix_cache_hits_total")
    row "Prefix cache (LMCache)" "queries: ${ec_q:-0}  hits: ${ec_h:-0}  rate: $(_rate "${ec_h:-0}" "$(echo "${ec_q:-0} ${ec_h:-0}" | awk '{printf "%.0f", $1-$2}')")"

    # Latency
    local ttft e2e itl
    ttft=$(_hist_avg "$raw" "vllm:time_to_first_token_seconds")
    e2e=$( _hist_avg "$raw" "vllm:e2e_request_latency_seconds")
    itl=$( _hist_avg "$raw" "vllm:inter_token_latency_seconds")
    row "Latency (avg)" "TTFT: ${ttft}s  E2E: ${e2e}s  ITL: ${itl}s"
}

lmcache_section() {
    header "LMCache  (port $LMCACHE_HTTP_PORT)"
    local raw
    raw=$(curl -sf "http://localhost:$LMCACHE_HTTP_PORT/metrics" 2>/dev/null) || {
        row "Status" "${RED}unreachable${RESET}"; return; }

    local mem_bytes usage_ratio prefetch_jobs queue_depth lag dropped ticks
    mem_bytes=$(    _m "$raw" "lmcache_mp_l1_memory_usage_bytes")
    usage_ratio=$(  _m "$raw" "lmcache_mp_l1_usage_ratio")
    prefetch_jobs=$(  _m "$raw" "lmcache_mp_active_prefetch_jobs")
    queue_depth=$(  _m "$raw" "lmcache_mp_event_bus_queue_depth")
    lag=$(          _m "$raw" "lmcache_mp_event_bus_drain_lag_seconds")
    dropped=$(      _m "$raw" "lmcache_mp_event_bus_dropped_events_total")
    ticks=$(        _m "$raw" "lmcache_mp_l1_eviction_loop_ticks_total")

    row "L1 cache size" "$(_bytes "${mem_bytes:-0}")  ($(echo "${usage_ratio:-0}" | awk '{printf "%.1f%%", $1*100}') full)"
    row "Active prefetch jobs" "${prefetch_jobs:-0}"
    row "Event bus" "queue: ${queue_depth:-0}  lag: ${lag:-0}s  dropped: ${RED}${dropped:-0}${RESET}"
    row "Eviction loop ticks" "${ticks:-0}"
}

clear
while true; do
    tput cup 0 0 2>/dev/null || clear
    printf "${BOLD}LMCache Observability${RESET}  $(date '+%H:%M:%S')  ${DIM}[↻ ${INTERVAL}s  VLLM_PORT=${VLLM_PORT}  LMCACHE_HTTP_PORT=${LMCACHE_HTTP_PORT}]${RESET}\n"
    gpu_section
    vllm_section
    lmcache_section
    printf "\n${DIM}  Ctrl+C to exit${RESET}\n"
    sleep "$INTERVAL"
done
