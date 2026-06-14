#!/usr/bin/env bash
# Run the official LMCache benchmark against a running vLLM + LMCache stack.
#
# Prerequisites:
#   1. ./setup/start_lmcache.sh  (LMCache server: HTTP :8080, ZMQ :5555)
#   2. ./setup/start_vllm.sh     (vLLM serve:    HTTP :8001)
#
# Uses `lmcache bench engine` from the local venv to send a long-doc-qa workload
# and report TTFT, decoding speed, and throughput.
#
# Usage:
#   ./benchmark.sh [options]
#
# Options:
#   --engine-url URL      vLLM endpoint (default: http://localhost:8001)
#   --lmcache-url URL     LMCache HTTP management endpoint (default: http://localhost:8080)
#   --config FILE         Benchmark config JSON (default: configs/bench_config.json)
#   --output-dir DIR      Directory to write bench_results.json (default: results)
#   --interactive         Run interactive mode to generate a new config, then exit
#   -h, --help            Show this help and exit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$ROOT/.venv"

log()  { printf '[benchmark] %s\n' "$*"; }
die()  { printf '[benchmark][ERROR] %s\n' "$*" >&2; exit 1; }

# --- Defaults -----------------------------------------------------------------
ENGINE_URL="${ENGINE_URL:-http://localhost:8001}"
LMCACHE_URL="${LMCACHE_URL:-http://localhost:8080}"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/bench_config.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results}"
INTERACTIVE=false

usage() {
  sed -n '/^# Usage:/,/^[^#]/{ /^#/{ s/^# \?//; p } }' "$0"
}

# --- Argument parsing ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine-url)    ENGINE_URL="${2:?--engine-url requires a value}";    shift 2 ;;
    --lmcache-url)   LMCACHE_URL="${2:?--lmcache-url requires a value}";  shift 2 ;;
    --config)        CONFIG_FILE="${2:?--config requires a value}";        shift 2 ;;
    --output-dir)    OUTPUT_DIR="${2:?--output-dir requires a value}";     shift 2 ;;
    --interactive)   INTERACTIVE=true; shift ;;
    -h|--help)       usage; exit 0 ;;
    *)               usage >&2; die "Unknown argument: $1" ;;
  esac
done

# --- Preflight ----------------------------------------------------------------
[[ -f "$VENV/bin/lmcache" ]] || die "lmcache not found in venv. Run ./setup/install.sh first."

log "Checking LMCache server at $LMCACHE_URL ..."
curl -sf "$LMCACHE_URL/" > /dev/null 2>&1 \
  || die "LMCache not reachable at $LMCACHE_URL. Run ./start_lmcache.sh first."
log "LMCache: OK"

log "Checking vLLM engine at $ENGINE_URL ..."
curl -sf "$ENGINE_URL/health" > /dev/null 2>&1 \
  || die "vLLM not reachable at $ENGINE_URL. Run ./start_vllm.sh first."
log "vLLM: OK"

mkdir -p "$OUTPUT_DIR"

# --- Interactive mode: generate a fresh bench_config.json --------------------
if [[ "$INTERACTIVE" == true ]]; then
  log "Running interactive config export (prompts will appear below)."
  log "Answer the prompts, then copy the exported bench_config.json to $SCRIPT_DIR/configs/."
  "$VENV/bin/lmcache" bench engine --lmcache-url "$LMCACHE_URL"
  exit 0
fi

# --- Replay mode: run benchmark from config file -----------------------------
[[ -f "$CONFIG_FILE" ]] || die "Config not found: $CONFIG_FILE. Run --interactive first or supply --config."

RESULT_FILE="$OUTPUT_DIR/bench_results_$(date +%Y%m%d_%H%M%S).json"

log "Running lmcache bench engine"
log "  engine-url  : $ENGINE_URL"
log "  lmcache-url : $LMCACHE_URL"
log "  config      : $CONFIG_FILE"
log "  output      : $RESULT_FILE"
log ""

# lmcache bench engine writes results to stdout as JSON when --output-json is
# supported; otherwise capture full stdout. The tool always prints a human-
# readable summary — tee lets us show it live and capture it simultaneously.
"$VENV/bin/lmcache" bench engine \
  --engine-url  "$ENGINE_URL" \
  --lmcache-url "$LMCACHE_URL" \
  --config      "$CONFIG_FILE" \
  | tee "$RESULT_FILE"

log ""
log "Benchmark complete. Raw output saved to: $RESULT_FILE"
