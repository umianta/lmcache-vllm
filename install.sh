#!/usr/bin/env bash
# Install vLLM + LMCache into a local venv.
# Run once. Requires Python 3.12 and CUDA toolkit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PYTHON="$VENV/bin/python3"

# ── Create venv ────────────────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV ..."
    python3 -m venv "$VENV"
fi

# Fix stale shebangs if venv was copied from another directory
STALE=$(grep -rl "#!" "$VENV/bin/" 2>/dev/null | xargs grep -l "^#!.*python" 2>/dev/null | xargs grep -lv "^#!$VENV" 2>/dev/null || true)
if [ -n "$STALE" ]; then
    echo "Fixing stale shebangs in .venv/bin/ ..."
    echo "$STALE" | xargs sed -i "s|^#!.*/bin/python|#!$VENV/bin/python|"
fi

# Bootstrap pip if missing (Ubuntu may omit it; ensurepip installs pip3/pip3.x)
if ! "$PYTHON" -m pip --version &>/dev/null; then
    echo "Bootstrapping pip..."
    "$PYTHON" -m ensurepip --upgrade
fi

# Use 'python -m pip' — works regardless of whether 'pip' or 'pip3' binary exists
echo "Upgrading pip..."
"$PYTHON" -m pip install --upgrade pip

# ── Skip if already installed ─────────────────────────────────────────────────
VLLM_OK=$("$PYTHON" -c "import vllm; print(vllm.__version__)" 2>/dev/null || true)
LMCACHE_OK=$("$PYTHON" -c "import lmcache; print(lmcache.__version__)" 2>/dev/null || true)

if [ -n "$VLLM_OK" ] && [ -n "$LMCACHE_OK" ]; then
    echo ""
    echo "Already installed:"
    echo "  vllm   : $VLLM_OK"
    echo "  lmcache: $LMCACHE_OK"
    echo ""
    echo "Run ./start_lmcache.sh and ./start_vllm.sh to start."
    exit 0
fi

# ── Install vLLM ──────────────────────────────────────────────────────────────
if [ -z "$VLLM_OK" ]; then
    echo "Installing vLLM..."
    "$PYTHON" -m pip install "vllm==0.8.5"
fi

# ── Install LMCache ───────────────────────────────────────────────────────────
if [ -z "$LMCACHE_OK" ]; then
    echo "Installing LMCache..."
    "$PYTHON" -m pip install "lmcache[vllm]==0.4.7"
fi

echo ""
echo "Install complete."
echo "  vllm   : $("$PYTHON" -c 'import vllm; print(vllm.__version__)')"
echo "  lmcache: $("$PYTHON" -c 'import lmcache; print(lmcache.__version__)')"
echo ""
echo "Next steps:"
echo "  1. ./start_lmcache.sh        # terminal 1 — cache server"
echo "  2. ./start_vllm.sh           # terminal 2 — vLLM"
echo "  3. source .venv/bin/activate && python src/experiments.py"
