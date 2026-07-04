"""Tests for the GPT-2-small confirmation-set driver (Task 2.3).

These tests exercise the driver's plumbing (deterministic index selection,
`prereg_ref=None` labelling, once-per-run canary, output artifacts, the "no
population claim" note) fast, via monkeypatching — the certifier's own GPT-2
dispatch (full vocab, `require_full`, exact-Fraction margin) is already
covered by `tests/test_certify.py` and is not re-tested here. A real
full-vocabulary GPT-2 forward pass takes minutes, so no test here performs
one; unit tests pass `run_canary=False` (the real run asserts the widths
canary once per run — covered by its own test below).
"""
import json
import os
from fractions import Fraction

import pytest

from certinf import gpt2_confirmation_run as G
from certinf import schema

FIXTURE_CORPUS = os.path.join(os.path.dirname(__file__), "fixtures",
                              "tinystories-dev.ids.json")


def _fake_record(prompt_index: int, context_length: int = 16,
                 status: str = "CERTIFIED") -> dict:
    if status == "CERTIFIED":
        return schema.build_record(
            model="gpt2", checkpoint_sha256="ab" * 32, corpus_sha256="cd" * 32,
            prompt_index=prompt_index, context_length=context_length,
            token_ids=[1, 2, 3], precision_P=320, P_max=448,
            escalation_trace=[320], status="CERTIFIED", argmax_token=42,
            margin_lo=Fraction(1, 3), abstain_reason=None,
            logit_width_max=Fraction(1, 10**9), top1_top2_float_gap=5.0,
            guard_audit={}, runtime_s=1.0, peak_rss_mb=1.0, phi1=True,
            phi2_joint=True, harness_transcript_sha256="ef" * 32,
            prereg_ref=None,
        )
    return schema.build_record(
        model="gpt2", checkpoint_sha256="ab" * 32, corpus_sha256="cd" * 32,
        prompt_index=prompt_index, context_length=context_length,
        token_ids=[1, 2, 3], precision_P=384, P_max=448,
        escalation_trace=[320, 384], status="ABSTAIN", argmax_token=None,
        margin_lo=None, abstain_reason="width",
        logit_width_max=Fraction(1, 10), top1_top2_float_gap=1e-6,
        guard_audit={}, runtime_s=1.0, peak_rss_mb=1.0, phi1=False,
        phi2_joint=None, harness_transcript_sha256=None, prereg_ref=None,
    )


def test_default_config_matches_task_spec():
    """The 2.3 reshape's resolved config: P_grid=[320,384], P_max=448,
    ctx=16, n=8 (superseding the brief's n=30 per the 2026-07-03 whole-corpus
    evaluation Q3 / SHOULD #4)."""
    assert G._DEFAULT_P_GRID == [320, 384]
    assert G._DEFAULT_P_MAX == 448
    assert G._DEFAULT_CONTEXT_LENGTH == 16
    assert G._DEFAULT_N == 8


def test_run_indices_are_first_n_deterministic(tmp_path, monkeypatch):
    calls = []

    def fake_certify_one(task):
        (_w, _c, idx, _pg, _pm, _ctx) = task
        calls.append(idx)
        return _fake_record(idx)

    monkeypatch.setattr(G, "_certify_one", fake_certify_one)
    summary = G.run(FIXTURE_CORPUS, str(tmp_path),
                    weights_path="/dev/null", n=8, context_length=16,
                    run_canary=False)
    assert summary["indices"] == list(range(8))
    assert calls == list(range(8))


def test_run_asserts_canary_once_before_any_sample(tmp_path, monkeypatch):
    """The widths canary must run exactly once, BEFORE the first sample
    (grid.py's per-cell granularity), on the run's first window at the
    ladder's first rung — and a tripped canary must abort the run before any
    record is produced."""
    events = []
    monkeypatch.setattr(
        G.canary, "assert_no_precision_floor",
        lambda model, ids, P: events.append(("canary", model, P)))

    def fake_certify_one(task):
        events.append(("sample", task[2]))
        return _fake_record(task[2])

    monkeypatch.setattr(G, "_certify_one", fake_certify_one)
    out1 = tmp_path / "ok"
    G.run(FIXTURE_CORPUS, str(out1), weights_path="/dev/null",
          n=2, context_length=16)
    assert events[0] == ("canary", "gpt2", 320)
    assert sum(1 for e in events if e[0] == "canary") == 1
    assert events[1:] == [("sample", 0), ("sample", 1)]

    # tripped canary -> run aborts, no records written
    def tripped(model, ids, P):
        raise AssertionError("precision floor")

    monkeypatch.setattr(G.canary, "assert_no_precision_floor", tripped)
    out2 = tmp_path / "tripped"
    with pytest.raises(AssertionError):
        G.run(FIXTURE_CORPUS, str(out2), weights_path="/dev/null",
              n=2, context_length=16)
    assert not os.path.exists(str(out2 / "gpt2-confirmation.cert.jsonl"))


def test_run_labels_prereg_ref_none_and_no_population_claim_note(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        G, "_certify_one", lambda task: _fake_record(task[2]))
    summary = G.run(FIXTURE_CORPUS, str(tmp_path), weights_path="/dev/null",
                    n=3, context_length=16, run_canary=False)
    assert summary["prereg_ref"] is None
    assert "no population claim" in summary["note"]
    assert "checker_status" in summary
    assert "not" in summary["checker_status"].lower() or \
        "future work" in summary["checker_status"].lower()

    with open(summary["cert_path"]) as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 3
    for rec in records:
        assert rec["prereg_ref"] is None
        schema.validate_record(rec)  # no raise


def test_run_writes_meta_and_single_fullvocab_cert(tmp_path, monkeypatch):
    monkeypatch.setattr(
        G, "_certify_one", lambda task: _fake_record(task[2]))
    summary = G.run(FIXTURE_CORPUS, str(tmp_path), weights_path="/dev/null",
                    n=2, context_length=16, run_canary=False)

    meta_path = os.path.join(str(tmp_path), "gpt2-confirmation.cert.meta.json")
    single_path = os.path.join(str(tmp_path), "gpt2-small-fullvocab.cert.json")
    assert os.path.exists(meta_path)
    assert os.path.exists(single_path)
    assert summary["single_cert_path"] == single_path

    with open(single_path) as f:
        single_rec = json.load(f)
    schema.validate_record(single_rec)
    assert single_rec["status"] == "CERTIFIED"
    assert single_rec["prompt_index"] == 0  # first (and here, only) record


def test_run_records_abstain_taxonomy_when_present(tmp_path, monkeypatch):
    def fake_certify_one(task):
        idx = task[2]
        return _fake_record(idx, status="ABSTAIN" if idx == 1 else "CERTIFIED")

    monkeypatch.setattr(G, "_certify_one", fake_certify_one)
    summary = G.run(FIXTURE_CORPUS, str(tmp_path), weights_path="/dev/null",
                    n=2, context_length=16, run_canary=False)
    assert summary["k_phi1"] == 1
    assert summary["abstain_taxonomy"] == {"width": 1}
    # the single fullvocab cert is still extracted from the CERTIFIED one
    with open(summary["single_cert_path"]) as f:
        single_rec = json.load(f)
    assert single_rec["prompt_index"] == 0


def test_certify_one_calls_real_certify_sample_with_expected_kwargs(
        monkeypatch):
    """Proves `_certify_one` (hence the driver) calls the real
    `certinf.certify.certify_sample` with `model='gpt2'`, `prereg_ref=None`,
    `run_harness=True`, `run_canary=False` (the canary is asserted once per
    RUN by `run()`, not per sample — grid.py's granularity), and the task's
    own P_grid/P_max/context_length — without a real forward pass."""
    captured = {}

    def fake_certify_sample(model, weights_path, corpus_path, *,
                            prompt_index, P_grid, P_max, prereg_ref=None,
                            run_harness=False, context_length=8,
                            run_canary=True, **_kw):
        captured.update(model=model, weights_path=weights_path,
                        corpus_path=corpus_path, prompt_index=prompt_index,
                        P_grid=P_grid, P_max=P_max, prereg_ref=prereg_ref,
                        run_harness=run_harness, context_length=context_length,
                        run_canary=run_canary)
        return _fake_record(prompt_index, context_length=context_length)

    monkeypatch.setattr(G.certify, "certify_sample", fake_certify_sample)
    task = ("/dev/null", FIXTURE_CORPUS, 3, [320, 384], 448, 16)
    rec = G._certify_one(task)

    assert captured["model"] == "gpt2"
    assert captured["prereg_ref"] is None
    assert captured["run_harness"] is True
    assert captured["run_canary"] is False
    assert captured["context_length"] == 16
    assert captured["P_grid"] == [320, 384]
    assert captured["P_max"] == 448
    assert captured["prompt_index"] == 3
    schema.validate_record(rec)
