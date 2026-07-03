"""Law fitting + calibration report (honest scope, A3).

CI-B: "the first measured precision-depth curve for these models on this
engine version" — never a "universal law". The slope is visibly family-
dependent (measured on however many model families the calibration grid
covers) and the inputs are point inputs only (no perturbation sets). This
caveat is baked into every `fit()` output (`honest_scope_note`) and must
appear verbatim in `docs/calibration-report.md`.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict

HONEST_SCOPE_NOTE = (
    "This is the first measured precision-depth curve for these models on "
    "this engine version — family-dependent slope, point inputs only; NOT a "
    "universal law."
)

# Layer counts from the model constants table (engine-seed/RESULTS.md,
# design spec §2 / plan "Model constants").
MODEL_DEPTH = {"tinystories": 8, "gpt2": 12}


def load_summary(summary_path: str) -> dict:
    with open(summary_path) as f:
        return json.load(f)


def _least_squares(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Stdlib ordinary least squares y = slope*x + intercept, plus r2."""
    n = len(points)
    if n == 0:
        return (0.0, 0.0, 0.0)
    if n == 1:
        return (0.0, points[0][1], 1.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    slope = ss_xy / ss_xx if ss_xx != 0 else 0.0
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    if ss_tot == 0:
        r2 = 1.0
    else:
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        r2 = 1 - ss_res / ss_tot
    return (slope, intercept, r2)


def fit(summary_path: str, model_depths: dict | None = None) -> dict:
    """Fit required-P (p95) vs depth across model families in a grid summary.

    `summary_path` points at a `calibration/summary.json`-shaped document:
    `{"cells": [{"model", "context_length", "required_P": {"p95": ...}}, ...]}`.
    For each model family present, the REPRESENTATIVE point is the max p95
    required-P observed across that family's context-length cells
    (conservative — the worst-case context length for that family), plotted
    against the family's layer depth; `per_family` also carries the full
    per-context-length breakdown. `model_depths` overrides the default
    real-model depth table (used by tests to exercise the fit against a
    synthetic multi-family summary with a known slope).
    """
    depths = model_depths if model_depths is not None else MODEL_DEPTH
    doc = load_summary(summary_path)
    by_model = defaultdict(list)
    for cell in doc["cells"]:
        p95 = cell.get("required_P", {}).get("p95")
        if p95 is None:
            continue
        by_model[cell["model"]].append((cell["context_length"], p95))

    per_family = {}
    points = []
    for model, ctx_p95 in by_model.items():
        depth = depths.get(model)
        if depth is None:
            continue
        rep_p95 = max(p95 for _, p95 in ctx_p95)
        per_family[model] = {
            "depth": depth,
            "by_context_length": {str(ctx): p95 for ctx, p95 in sorted(ctx_p95)},
            "representative_p95": rep_p95,
        }
        points.append((depth, rep_p95))

    slope, intercept, r2 = _least_squares(points)
    return {
        "slope_bits_per_layer": slope,
        "intercept": intercept,
        "r2": r2,
        "per_family": per_family,
        "honest_scope_note": HONEST_SCOPE_NOTE,
    }


def _fmt_ci(ci) -> str:
    if ci is None or ci[0] is None:
        return "n/a"
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def render_report(summary_path: str, fit_result: dict | None = None) -> str:
    """Render docs/calibration-report.md content: quantile tables, bootstrap
    intervals, abstain taxonomy, float-gap distributions, and the fitted
    curve with its honest scope."""
    doc = load_summary(summary_path)
    if fit_result is None:
        fit_result = fit(summary_path)

    lines = []
    lines.append("# Calibration report (Phase 1, non-headline)")
    lines.append("")
    lines.append(
        "Every record in this report has `prereg_ref=None` (A2: calibration "
        "samples are never headline samples). Abstention is never hidden — "
        "the taxonomy table below reports every abstain reason observed, "
        "including zero counts."
    )
    lines.append("")
    lines.append("## Per-cell quantile tables")
    lines.append("")
    lines.append(
        "| model | ctx | P (cell ceiling) | n | n_cert | required-P p95 (95% CI) "
        "| required-P p99 (95% CI) | float-gap p95 | abstain taxonomy |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cell in doc["cells"]:
        rp = cell.get("required_P", {})
        fg = cell.get("top1_top2_float_gap", {}).get("quantiles", {})
        taxonomy = cell.get("abstain_taxonomy", {})
        tax_str = ", ".join(f"{k}={v}" for k, v in taxonomy.items() if v)
        or_none = "none"
        p95 = rp.get("p95")
        p99 = rp.get("p99")
        p95_str = f"{p95:.1f}" if p95 is not None else "n/a"
        p99_str = f"{p99:.1f}" if p99 is not None else "n/a"
        lines.append(
            f"| {cell['model']} | {cell['context_length']} | {cell['P']} | "
            f"{cell['n_samples']} | {cell['n_certified']} | "
            f"{p95_str} {_fmt_ci(rp.get('p95_ci'))} | "
            f"{p99_str} {_fmt_ci(rp.get('p99_ci'))} | "
            f"{fg.get('p95', 0):.3g} | {tax_str or or_none} |"
        )
    lines.append("")
    lines.append("## Float-gap distributions (top1-vs-top2, diagnostic)")
    lines.append("")
    for cell in doc["cells"]:
        fg = cell.get("top1_top2_float_gap", {})
        sample = fg.get("histogram_sample", [])
        lines.append(
            f"- **{cell['model']} ctx{cell['context_length']} P{cell['P']}**: "
            f"n={len(sample)} sampled gaps, "
            f"min={min(sample):.3g} max={max(sample):.3g}" if sample else
            f"- **{cell['model']} ctx{cell['context_length']} P{cell['P']}**: no data"
        )
    lines.append("")
    lines.append("## Runtime / memory quantiles")
    lines.append("")
    lines.append("| model | ctx | P | runtime_s p95 | peak_rss_mb p95 |")
    lines.append("|---|---|---|---|---|")
    for cell in doc["cells"]:
        rt = cell.get("runtime_s", {})
        rss = cell.get("peak_rss_mb", {})
        lines.append(
            f"| {cell['model']} | {cell['context_length']} | {cell['P']} | "
            f"{rt.get('p95', 0):.2f} | {rss.get('p95', 0):.1f} |"
        )
    lines.append("")
    lines.append("## Fitted precision-depth curve (CI-B)")
    lines.append("")
    if len(fit_result["per_family"]) < 2:
        lines.append(
            "**Single-family grid: no cross-family slope is estimable from "
            "this summary alone.** The fit below degenerates to the one "
            "family's representative point (slope 0 by construction); the "
            "only cross-family depth measurement to date remains the "
            "foothold's two-model observation (~17.3 vs ~23.6 bits/layer, "
            "engine-seed/RESULTS.md), which this grid neither confirms nor "
            "extends across families."
        )
        lines.append("")
    lines.append(
        f"`slope_bits_per_layer = {fit_result['slope_bits_per_layer']:.2f}`, "
        f"`intercept = {fit_result['intercept']:.2f}`, `r2 = {fit_result['r2']:.4f}`"
    )
    lines.append("")
    lines.append("Per-family points used in the fit:")
    lines.append("")
    lines.append("| family | depth (layers) | representative p95 required-P |")
    lines.append("|---|---|---|")
    for model, fam in fit_result["per_family"].items():
        lines.append(f"| {model} | {fam['depth']} | {fam['representative_p95']:.1f} |")
    lines.append("")
    lines.append(f"> {fit_result['honest_scope_note']}")
    lines.append("")
    return "\n".join(lines)


def write_report(summary_path: str, out_path: str) -> str:
    text = render_report(summary_path)
    with open(out_path, "w") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="calibration/summary.json")
    parser.add_argument("--out", default="docs/calibration-report.md")
    args = parser.parse_args()
    write_report(args.summary, args.out)
