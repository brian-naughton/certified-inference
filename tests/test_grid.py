import json

from certinf import grid
from certinf.float_fwd import MODEL_BIN

CORPUS = "certificates/corpora/tinystories-val.ids.json"


def test_bootstrap_ci_brackets_point_estimate():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    lo, hi = grid.bootstrap_ci(data, seed=0)
    import statistics
    point = statistics.median(data)
    assert lo <= point <= hi


def test_run_cell_returns_summary_with_required_keys():
    summary = grid.run_cell(
        "tinystories", MODEL_BIN, CORPUS, context_length=8, P=192,
        n_samples=4, jobs=1,
    )
    for key in ("model", "context_length", "P", "n_samples", "n_certified",
               "n_abstained", "all_prereg_ref_null", "required_P",
               "top1_top2_float_gap", "abstain_taxonomy", "runtime_s",
               "peak_rss_mb", "records_path"):
        assert key in summary, key
    assert summary["n_samples"] == 4
    assert summary["all_prereg_ref_null"] is True

    with open(summary["records_path"]) as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 4
    for rec in records:
        assert rec["prereg_ref"] is None
        assert rec["status"] in {"CERTIFIED", "ABSTAIN"}


def test_run_cell_multiworker_matches_single_worker_record_count():
    summary = grid.run_cell(
        "tinystories", MODEL_BIN, CORPUS, context_length=8, P=192,
        n_samples=2, jobs=2,
    )
    assert summary["n_samples"] == 2
    with open(summary["records_path"]) as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 2
