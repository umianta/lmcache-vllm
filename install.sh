#!/usr/bin/env bash
# Install vLLM + LMCache into a local venv.
# Run once. Requires Python 3.12 and CUDA toolkit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"

# ── Create venv ────────────────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV ..."
    python3 -m venv "$VENV"
fi

echo "Upgrading pip..."
"$PIP" install --upgrade pip

# ── Install vLLM ──────────────────────────────────────────────────────────────
echo "Installing vLLM..."
"$PIP" install "vllm==0.8.5"

# ── Install LMCache ───────────────────────────────────────────────────────────
echo "Installing LMCache..."
"$PIP" install "lmcache[vllm]==0.4.7"

echo ""
echo "Install complete."
echo "  vllm   : $("$PYTHON" -c 'import vllm; print(vllm.__version__)')"
echo "  lmcache: $("$PYTHON" -c 'import lmcache; print(lmcache.__version__)')"
echo ""
echo "Next steps:"
echo "  1. ./start_lmcache.sh        # terminal 1 — cache server"
echo "  2. ./start_vllm.sh           # terminal 2 — vLLM"
echo "  3. source .venv/bin/activate && python src/experiments.py"
