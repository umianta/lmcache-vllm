# LMCache + vLLM — KV Cache Offload

Scripts and benchmarks for running [LMCache](https://lmcache.ai) KV-cache offloading
alongside [vLLM](https://vllm.ai). Tested on NVIDIA DGX Spark but works on any
CUDA-capable machine meeting the requirements below.

## How it works

LMCache stores the KV cache computed during prefill in CPU memory. Subsequent requests
that share a long prefix skip prefill entirely — the official benchmark reports
**~7.43× speedup** on warm requests.

```
Client request
  └─► vLLM (GPU)
        ├─► [COLD] compute KV → write to LMCache server (CPU RAM)
        └─► [WARM] read KV from LMCache server → skip prefill (~7× faster)

LMCache standalone server
  ├─► HTTP :8080  —  management API  (clear cache, metrics)
  └─► ZMQ  :5555  —  data plane      (vLLM connects here)
```

## Infrastructure requirements

| Resource | Minimum | Recommended |
|---|---|---|
| GPU | 16 GB VRAM (CUDA 12+) | 40 GB+ (A100, H100, GB10) |
| CPU RAM | 16 GB (for L1 cache) | 64 GB+ |
| Disk | 20 GB (model weights) | SSD, 100 GB+ |
| Python | 3.10+ | 3.12 |
| CUDA | 12.1+ | 12.4+ |
| OS | Linux (Ubuntu 22.04+) | Ubuntu 24.04 |

> **Multi-GPU:** vLLM supports tensor parallelism via `--tensor-parallel-size`.
> Set it to the number of GPUs. LMCache works the same regardless of GPU count.

> **Unified memory (DGX Spark / GB200):** `nvidia-smi` reports `[N/A]` for free
> memory on Grace Blackwell chips. Set `VLLM_GPU_MEM` manually — default is `0.24`
> (safe for multi-model nodes). Use `0.85` when running a single model.

## Repository layout

```
.
├── install.sh           # one-time setup: create venv, install vLLM + LMCache
├── start_lmcache.sh     # start the standalone LMCache server
├── start_vllm.sh        # start vLLM wired to LMCache
├── configs/
│   └── lmcache_cpu.yaml # example LMCache config (CPU-only L1 backend)
└── src/
    ├── cache_utils.py   # clear cache via HTTP (POST /clear-cache)
    ├── experiments.py   # benchmark suite — cold vs warm timing
    └── offloadtocpu.py  # minimal reference: in-process vLLM + LMCache
```

## Quick start

### 1 — Install

On Ubuntu, ensure `python3-venv` is available before running the installer:

```bash
sudo apt install python3-venv python3-full
```

Then clone and install:

```bash
git clone https://github.com/umianta/lmcache-vllm
cd lmcache-vllm
./install.sh
```

Creates `.venv/` and installs `vllm` and `lmcache[vllm]` into it using the venv's
own pip — no system packages are touched.

### 2 — Start LMCache server (terminal 1)

```bash
./start_lmcache.sh
```

Starts a standalone LMCache process with a 20 GB CPU L1 cache and LRU eviction.
Adjust `--l1-size-gb` in the script to match your available RAM.

### 3 — Start vLLM (terminal 2)

```bash
./start_vllm.sh
```

Starts `vllm serve Qwen/Qwen3-8B` on port **8001** connected to LMCache over ZMQ.

**Override defaults with environment variables:**

```bash
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct ./start_vllm.sh
VLLM_GPU_MEM=0.85 ./start_vllm.sh     # single-model node
VLLM_PORT=8002 ./start_vllm.sh        # different port
```

### 4 — Run benchmarks (terminal 3)

```bash
source .venv/bin/activate

python src/experiments.py --list       # see all experiments
python src/experiments.py baseline     # cold vs warm reference
python src/experiments.py all          # run everything
```

### 5 — Clear the cache

```bash
python src/cache_utils.py
# or directly:
curl -X POST http://localhost:8080/clear-cache
```

---

## Experiments

| Name | What it tests |
|---|---|
| `baseline` | Single long prefix — cold vs warm reference timing |
| `multi_suffix` | 5 suffixes on the same prefix — batch cache-hit exercise |
| `short_prefix` | Very short prefix — near-zero benefit, measures cold overhead |
| `long_prefix` | ~7 k token prefix — stresses the CPU buffer at scale |
| `three_runs` | Three passes — verifies cache stability beyond first warm hit |
| `long_output` | 128-token output — isolates decode latency from prefill savings |

### Reading the output

```
  [COLD]   suffix='Hello, my name is'   2.41s  → ' John'
  [COLD]   pass total: 2.41s

  [WARM 1] suffix='Hello, my name is'   0.32s  → ' John'
  [WARM 1] pass total: 0.32s

  Speedup (cold / warm-1): 7.53x  (doc reference ≈ 7.43x)
```

- **COLD** — first pass, KV computed from scratch and written to CPU RAM
- **WARM N** — Nth pass, KV read from CPU RAM, prefill skipped
- Speedup is `cold_total / warm1_total`

---

## Configuration reference

### `start_vllm.sh` environment variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_MODEL` | `Qwen/Qwen3-8B` | Model to serve (any HuggingFace ID) |
| `VLLM_PORT` | `8001` | vLLM HTTP port |
| `VLLM_GPU_MEM` | `0.24` | GPU memory fraction (0.0–1.0) |
| `VLLM_MAX_LEN` | `8192` | Max context length in tokens |
| `LMCACHE_HOST` | `tcp://localhost` | LMCache ZMQ host |
| `LMCACHE_PORT` | `5555` | LMCache ZMQ port |

### `start_lmcache.sh` flags

| Flag | Default | Description |
|---|---|---|
| `--l1-size-gb` | `20` | CPU RAM budget for the L1 cache |
| `--eviction-policy` | `LRU` | `LRU` or `FIFO` |
| `--chunk-size` | `16` | Token chunk size for cache blocks |

---

## Troubleshooting

### `ValueError: Free memory ... is less than desired GPU memory utilization`

Another process is using the GPU. Check usage and lower the fraction:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
VLLM_GPU_MEM=0.20 ./start_vllm.sh
```

### `error: externally-managed-environment` during install

Ubuntu 22.04+ prevents pip from installing into the system Python. The fix is to
ensure `python3-venv` is installed so the venv can be created with its own pip:

```bash
sudo apt install python3-venv python3-full
./install.sh
```

The scripts never touch the system Python — all binaries are called as
`.venv/bin/pip`, `.venv/bin/vllm`, etc.

### `.venv/bin/pip: No such file or directory` during install

The venv was created without pip (happens on some Ubuntu setups). Delete it and
reinstall — `install.sh` now bootstraps pip automatically via `ensurepip`:

```bash
rm -rf .venv
./install.sh
```

### `exec: lmcache: not found` or `exec: vllm: not found`

The venv does not exist or `./install.sh` did not complete. Run it first:

```bash
./install.sh
```

### `Cannot reach http://localhost:8080`

Start the LMCache server first (`./start_lmcache.sh`) before running vLLM or
`cache_utils.py`.

### SSH disconnects under high GPU load

On high-load servers `pam_systemd` can time out creating new SSH sessions (common
with VS Code Remote). Fix:

```bash
sudo sed -i 's/^session\toptional\tpam_systemd\.so/# &/' /etc/pam.d/common-session
sudo systemctl restart ssh
```

---

## Tested with

| Component | Version |
|---|---|
| vLLM | 0.23.0 |
| LMCache | 0.4.7 |
| Python | 3.12 |
| CUDA | 12.4 |
| Model | Qwen/Qwen3-8B |
| Hardware | NVIDIA DGX Spark (GB10, 128 GiB unified memory) |

## References

- [LMCache documentation](https://docs.lmcache.ai)
- [LMCache multiprocess quickstart](https://docs.lmcache.ai/mp/quickstart.html)
- [vLLM KV transfer config](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)
