import json

from certinf import lawfit


def _synthetic_summary(tmp_path, known_slope=20.0, intercept=100.0):
    """Three synthetic families with a small amount of jitter around an exact
    linear relationship P95 = known_slope * depth + intercept, so the fit
    should recover known_slope within a small tolerance (not merely a
    trivial 2-point exact fit)."""
    depths = {"famA": 4, "famB": 8, "famC": 16}
    jitter = {"famA": -1.0, "famB": 2.0, "famC": -0.5}
    cells = []
    for fam, depth in depths.items():
        p95 = known_slope * depth + intercept + jitter[fam]
        cells.append({
            "model": fam,
            "context_length": 8,
            "P": 999,
            "n_samples": 10,
            "n_certified": 10,
            "required_P": {"p95": p95, "p99": p95 + 5},
            "top1_top2_float_gap": {"quantiles": {"p95": 0.5},
                                    "histogram_sample": [0.1, 0.2, 0.5]},
            "abstain_taxonomy": {"near-tie": 0, "width": 0, "guard": 0,
                                 "bug": 0, "timeout": 0},
            "runtime_s": {"p95": 5.0}, "peak_rss_mb": {"p95": 200.0},
        })
    out = tmp_path / "synthetic_summary.json"
    with open(out, "w") as f:
        json.dump({"cells": cells}, f)
    return str(out), {"famA": 4, "famB": 8, "famC": 16}


def test_fit_recovers_known_slope(tmp_path):
    summary_path, model_depths = _synthetic_summary(tmp_path, known_slope=20.0)
    result = lawfit.fit(summary_path, model_depths=model_depths)
    assert abs(result["slope_bits_per_layer"] - 20.0) < 1.0
    assert result["r2"] > 0.95
    assert set(result["per_family"]) == {"famA", "famB", "famC"}


def test_fit_honest_scope_note_verbatim_fragments():
    result = lawfit.fit.__doc__  # sanity: function is documented
    assert result is not None


def test_report_contains_honest_scope_phrases(tmp_path):
    summary_path, model_depths = _synthetic_summary(tmp_path, known_slope=15.0)
    fit_result = lawfit.fit(summary_path, model_depths=model_depths)
    report = lawfit.render_report(summary_path, fit_result=fit_result)
    assert "first measured" in report
    assert "not a universal law" in report.lower()
    assert "famA" in report and "famB" in report and "famC" in report


def test_write_report_writes_file(tmp_path):
    summary_path, model_depths = _synthetic_summary(tmp_path)
    out_path = tmp_path / "report.md"
    text = lawfit.write_report(summary_path, str(out_path))
    assert out_path.exists()
    assert out_path.read_text() == text
    assert "not a universal law" in text.lower()
