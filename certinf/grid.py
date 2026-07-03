"""Calibration grid runner (A4 outputs) — non-headline, pre-registration-free.

Every record produced here has `prereg_ref=None` (A2: calibration samples are
NEVER headline samples — exploration and confirmation are strictly
separated). A "cell" is one (model, context_length, P) triple, where `P`
plays the role of that cell's `P_max`: each sample is certified via the
model's standard escalation ladder (see DEFAULT_P_GRID), capped at the cell's
`P`, so the per-sample "required P" (the first rung that certifies) can vary
below the cap — the quantiles of that required P are this module's headline
A4 output. The widths canary runs once per cell (before the cell's records
are trusted), not once per sample (see certinf.certify's docstring for the
per-sample alternative granularity certify_sample itself supports).

Grid runs are meant to be nohup background CPU jobs (see the plan's Task 1.3
Step 4 command); `run_grid` is the config-driven entry point for that.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import random
import statistics
import time

from certinf import canary, certify, corpus

DEFAULT_P_GRID = {
    "tinystories": [96, 128, 160, 192, 224, 256],
    "gpt2": [320, 384, 448],
}

_ABSTAIN_TAXONOMY = ("near-tie", "width", "guard", "bug", "timeout")


def quantile(sorted_xs: list[float], q: float) -> float:
    """Nearest-rank-with-interpolation quantile on a pre-sorted list (matches
    the foothold sweep's convention in engine-seed/sweep.py)."""
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    i = int(pos)
    frac = pos - i
    if i + 1 < len(sorted_xs):
        return sorted_xs[i] * (1 - frac) + sorted_xs[i + 1] * frac
    return sorted_xs[i]


def bootstrap_ci(data: list[float], stat_fn=None, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Stdlib bootstrap 95% CI (default alpha=0.05) for a statistic of `data`.

    `stat_fn` defaults to the median. Resampling uses
    `random.Random(seed).choices` (with replacement), per the plan's stdlib
    bootstrap discipline (no numpy on this path). Returns (lo, hi); the point
    estimate `stat_fn(data)` satisfies `lo <= point <= hi` whenever the
    resample distribution is not pathological (degenerate/constant data
    collapses lo == point == hi, which is still a valid, if trivial, bound).
    """
    if stat_fn is None:
        stat_fn = statistics.median
    if not data:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(data)
    stats = []
    for _ in range(n_boot):
        sample = rng.choices(data, k=n)
        stats.append(stat_fn(sample))
    stats.sort()
    lo = quantile(stats, alpha / 2)
    hi = quantile(stats, 1 - alpha / 2)
    point = stat_fn(data)
    lo = min(lo, point)
    hi = max(hi, point)
    return (lo, hi)


def _certify_one(args) -> dict:
    (model, weights_path, corpus_path, prompt_index, P_grid, P_max,
     context_length) = args
    return certify.certify_sample(
        model, weights_path, corpus_path, prompt_index=prompt_index,
        P_grid=P_grid, P_max=P_max, prereg_ref=None, run_harness=False,
        context_length=context_length, run_canary=False,
    )


def run_cell(model: str, weights_path: str, corpus_path: str,
            context_length: int, P: int, n_samples: int, jobs: int = 1) -> dict:
    """Certify `n_samples` windows at `context_length`, escalation capped at
    `P`, and return the A4 per-cell summary. Writes
    `certificates/calibration/<model>_ctx<L>_P<P>.jsonl`."""
    corpus_doc = corpus.load(corpus_path)
    windows = corpus_doc["windows"].get(str(context_length))
    if windows is None:
        raise ValueError(f"corpus {corpus_path!r} has no windows at "
                         f"context_length={context_length}")
    n_samples = min(n_samples, len(windows))
    P_grid = [p for p in DEFAULT_P_GRID.get(model, [P]) if p <= P]
    if not P_grid:
        P_grid = [P]

    # Widths canary once per cell (before the cell's records are trusted).
    canary.assert_no_precision_floor(model, windows[0], P_grid[0])

    tasks = [(model, weights_path, corpus_path, i, P_grid, P, context_length)
            for i in range(n_samples)]

    t0 = time.time()
    if jobs > 1:
        with multiprocessing.Pool(jobs) as pool:
            records = pool.map(_certify_one, tasks)
    else:
        records = [_certify_one(t) for t in tasks]
    cell_runtime_s = time.time() - t0

    out_path = f"certificates/calibration/{model}_ctx{context_length}_P{P}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    required_p = sorted(r["precision_P"] for r in records if r["status"] == "CERTIFIED")
    float_gaps = sorted(r["top1_top2_float_gap"] for r in records)
    runtimes = sorted(r["runtime_s"] for r in records)
    peak_rss = sorted(r["peak_rss_mb"] for r in records)
    taxonomy = {k: 0 for k in _ABSTAIN_TAXONOMY}
    for r in records:
        if r["abstain_reason"] is not None:
            taxonomy[r["abstain_reason"]] = taxonomy.get(r["abstain_reason"], 0) + 1

    def _quantiles_with_ci(xs):
        if not xs:
            return {"max": None, "p95": None, "p99": None,
                    "p95_ci": (None, None), "p99_ci": (None, None)}
        return {
            "max": xs[-1],
            "p95": quantile(xs, 0.95),
            "p99": quantile(xs, 0.99),
            "p95_ci": bootstrap_ci(xs, lambda d: quantile(sorted(d), 0.95)),
            "p99_ci": bootstrap_ci(xs, lambda d: quantile(sorted(d), 0.99)),
        }

    summary = {
        "model": model,
        "context_length": context_length,
        "P": P,
        "P_grid": P_grid,
        "n_samples": n_samples,
        "n_certified": len(required_p),
        "n_abstained": n_samples - len(required_p),
        "all_prereg_ref_null": all(r["prereg_ref"] is None for r in records),
        "required_P": _quantiles_with_ci([float(x) for x in required_p]),
        "top1_top2_float_gap": {
            "quantiles": _quantiles_with_ci(float_gaps),
            "histogram_sample": float_gaps[:50],
        },
        "abstain_taxonomy": taxonomy,
        "runtime_s": _quantiles_with_ci(runtimes),
        "peak_rss_mb": _quantiles_with_ci(peak_rss),
        "cell_runtime_s": cell_runtime_s,
        "records_path": out_path,
    }
    return summary


def run_grid(config: dict) -> dict:
    """Run the full calibration grid over models x contexts x P_grid.

    `config`: {"cells": [{"model", "weights_path", "corpus_path",
    "context_length", "P", "n_samples", "jobs"}, ...]}. Writes
    `calibration/summary.json` (all cell summaries) in addition to each
    cell's own JSONL under certificates/calibration/.
    """
    cells = []
    for cell_cfg in config["cells"]:
        cells.append(run_cell(**cell_cfg))
    out = {"cells": cells}
    os.makedirs("calibration", exist_ok=True)
    with open("calibration/summary.json", "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    for cell in cfg.get("cells", []):
        cell.setdefault("jobs", args.jobs)
    run_grid(cfg)
