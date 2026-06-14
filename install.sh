#!/usr/bin/env bash
# Install vLLM + LMCache into a local venv.
# Run once. Requires Python 3.12 and CUDA toolkit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# ── Create venv if needed ──────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV ..."
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"
pip install --upgrade pip

# ── Install vLLM (pulls in torch, cuda deps) ──────────────────────────────────
# Pin to the version that ships with this LMCache build (0.23.0 confirmed).
pip install "vllm==0.8.5"

# ── Install LMCache with vLLM integration ─────────────────────────────────────
pip install "lmcache[vllm]==0.4.7"

echo ""
echo "Install complete."
echo "  vllm   : $(python -c 'import vllm; print(vllm.__version__)')"
echo "  lmcache: $(python -c 'import lmcache; print(lmcache.__version__)')"
echo ""
echo "Next steps:"
echo "  1. ./start_lmcache.sh        # terminal 1 — cache server"
echo "  2. ./start_vllm.sh           # terminal 2 — vLLM"
echo "  3. python src/experiments.py # terminal 3 — benchmarks"
