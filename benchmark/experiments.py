"""
LMCache KV Cache offload experiments — against a live vLLM server.
Target: http://localhost:8001/v1/completions (OpenAI-compatible)

Run:
    python src/experiments.py baseline
    python src/experiments.py all
    python src/experiments.py --list
"""

import time
import argparse
from dataclasses import dataclass, field

import httpx

BASE_URL = "http://localhost:8001/v1/completions"
MODEL    = "Qwen/Qwen3-8B"          # must match whatever the server loaded


# ── Experiment registry ────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    name: str
    description: str
    prefix_reps: int = 1000           # "Hello, how are you? " repeated N times
    suffixes: list[str] = field(default_factory=lambda: ["Hello, my name is"])
    max_tokens: int = 10
    temperature: float = 0.0
    runs: int = 2                     # passes per experiment (run-0 = cold, rest = warm)


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "baseline": ExperimentConfig(
        name="baseline",
        description="Single prompt, long shared prefix — establishes cold/warm timing",
    ),

    "multi_suffix": ExperimentConfig(
        name="multi_suffix",
        description="Several suffixes on same prefix — batch cache-hit exercise",
        suffixes=[
            "Hello, my name is",
            "What is your favourite colour?",
            "Tell me a joke.",
            "Summarise the above in one sentence.",
            "List three key points.",
        ],
    ),

    "short_prefix": ExperimentConfig(
        name="short_prefix",
        prefix_reps=5,
        description="Very short prefix — near-zero cache benefit, measures cold overhead",
    ),

    "long_prefix": ExperimentConfig(
        name="long_prefix",
        prefix_reps=1200,
        description="Large prefix (~7k tokens) — stresses CPU offload buffer at scale",
    ),

    "three_runs": ExperimentConfig(
        name="three_runs",
        description="Three passes — verifies cache is stable after first warm hit",
        runs=3,
    ),

    "long_output": ExperimentConfig(
        name="long_output",
        description="Longer generation (128 tokens) — isolates decode vs prefill savings",
        max_tokens=128,
    ),
}


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def complete(
    prompt: str,
    max_tokens: int = 10,
    temperature: float = 0.0,
    client: httpx.Client | None = None,
) -> tuple[str, float]:
    """POST to /v1/completions, return (text, elapsed_seconds)."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    _client = client or httpx.Client(timeout=300)
    t0 = time.perf_counter()
    resp = _client.post(BASE_URL, json=payload)
    elapsed = time.perf_counter() - t0
    if not resp.is_success:
        raise httpx.HTTPStatusError(
            f"{resp.status_code} — {resp.text}",
            request=resp.request,
            response=resp,
        )
    text = resp.json()["choices"][0]["text"]
    return text, elapsed


def check_server() -> None:
    try:
        r = httpx.get("http://localhost:8001/health", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise SystemExit(f"Server not reachable at http://localhost:8001 — {e}") from e


# ── Core runner ───────────────────────────────────────────────────────────────

def run_experiment(cfg: ExperimentConfig) -> None:
    print(f"\n{'═' * 62}")
    print(f"  EXPERIMENT : {cfg.name}")
    print(f"  {cfg.description}")
    print(f"  prefix_reps={cfg.prefix_reps}  suffixes={len(cfg.suffixes)}  "
          f"max_tokens={cfg.max_tokens}  runs={cfg.runs}")
    print(f"{'═' * 62}\n")

    shared_prefix = "Hello, how are you? " * cfg.prefix_reps
    prompts = [shared_prefix + s for s in cfg.suffixes]

    run_timings: list[float] = []   # total wall time per pass

    with httpx.Client(timeout=300) as client:
        for run_idx in range(cfg.runs):
            label = "COLD" if run_idx == 0 else f"WARM {run_idx}"
            pass_t0 = time.perf_counter()

            for prompt, suffix in zip(prompts, cfg.suffixes):
                text, t = complete(prompt, cfg.max_tokens, cfg.temperature, client)
                print(f"  [{label}] suffix={suffix[:30]!r:<32} {t:.2f}s  → {text.strip()!r}")

            pass_elapsed = time.perf_counter() - pass_t0
            run_timings.append(pass_elapsed)
            print(f"  [{label}] pass total: {pass_elapsed:.2f}s\n")

    if len(run_timings) >= 2:
        speedup = run_timings[0] / run_timings[1]
        print(f"  Speedup (cold / warm-1): {speedup:.2f}x  (doc reference ≈ 7.43x)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    global BASE_URL
    parser = argparse.ArgumentParser(
        description="LMCache KV offload experiments against http://localhost:8001"
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        default="baseline",
        choices=list(EXPERIMENTS) + ["all"],
        help="Experiment to run (default: baseline)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print available experiments and exit",
    )
    parser.add_argument(
        "--base-url", default=BASE_URL,
        help=f"Override server URL (default: {BASE_URL})",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:")
        for name, cfg in EXPERIMENTS.items():
            print(f"  {name:<15}  {cfg.description}")
        return

    BASE_URL = args.base_url

    check_server()

    targets = list(EXPERIMENTS.values()) if args.experiment == "all" else [EXPERIMENTS[args.experiment]]

    wall_t0 = time.perf_counter()
    for cfg in targets:
        run_experiment(cfg)
    print(f"\n{'─' * 62}")
    print(f"  Total wall time: {time.perf_counter() - wall_t0:.1f}s")


if __name__ == "__main__":
    main()
