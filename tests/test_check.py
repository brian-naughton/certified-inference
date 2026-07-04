"""Independent torch-free checker (certificates/check.py).

These tests re-derive real certificate records from the committed hex weight
export + committed corpus — no torch — so they exercise the actual trust
boundary the checker closes. They reuse the first couple of committed
TinyStories calibration records (which certify against the real weights) as
fixtures, rewriting only the provenance fields needed to stand up a headline
(pre-registered) cert on a small fixture corpus.
"""
import json
import os
import subprocess
import sys
from fractions import Fraction

import pytest

from certinf import corpus, prereg, schema

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHECK = os.path.join(_REPO, "certificates", "check.py")
_WEIGHTS = os.path.join(_REPO, "certificates", "tinystories-1M.weights.json")
_CORPUS = os.path.join(_REPO, "certificates", "corpora", "tinystories-val.ids.json")
_CALIB = os.path.join(_REPO, "certificates", "calibration",
                      "tinystories_ctx8_P256.jsonl")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(_WEIGHTS) and os.path.exists(_CALIB)),
    reason="committed hex weights / calibration cert not present",
)


def _real_records(n):
    """First `n` committed ctx8 calibration records (CERTIFIED, prereg_ref null)."""
    out = []
    with open(_CALIB) as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) == n:
                break
    return out


def _run(**kw):
    """Invoke the checker CLI; return (returncode, stdout)."""
    cmd = [sys.executable, _CHECK,
           "--weights", kw.get("weights", _WEIGHTS),
           "--corpus", kw["corpus"],
           "--cert", kw["cert"]]
    if kw.get("prereg"):
        cmd += ["--prereg", kw["prereg"]]
    if kw.get("sample"):
        cmd += ["--sample", str(kw["sample"])]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def _write_cert(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


# --------------------------------------------------------------------------- #
# calibration mode (prereg_ref null)
# --------------------------------------------------------------------------- #
def test_verified_calibration(tmp_path):
    """A calibration cert (prereg_ref null) of a real record re-derives and
    passes with a 'no population claim' VERIFIED line."""
    cert = tmp_path / "calib.jsonl"
    _write_cert(cert, _real_records(1))
    rc, out = _run(corpus=_CORPUS, cert=str(cert))
    assert rc == 0, out
    assert "VERIFIED" in out and "no population claim" in out


def test_forged_margin_fails(tmp_path):
    """Tampering margin_lo breaks the bit-identical re-derivation."""
    rec = _real_records(1)[0]
    rec["margin_lo"] = [str(int(rec["margin_lo"][0]) + 1), rec["margin_lo"][1]]
    cert = tmp_path / "forged.jsonl"
    _write_cert(cert, [rec])
    rc, out = _run(corpus=_CORPUS, cert=str(cert))
    assert rc == 1, out
    assert "FAILED" in out and "margin_lo" in out


def test_forged_argmax_fails(tmp_path):
    """Tampering argmax_token is caught by the interval-re-derived argmax."""
    rec = _real_records(1)[0]
    rec["argmax_token"] = rec["argmax_token"] + 1
    cert = tmp_path / "forged_arg.jsonl"
    _write_cert(cert, [rec])
    rc, out = _run(corpus=_CORPUS, cert=str(cert))
    assert rc == 1, out
    assert "FAILED" in out and "argmax" in out


def test_corpus_sha_mismatch_fails(tmp_path):
    """A record whose corpus_sha256 disagrees with the passed corpus fails
    before any re-derivation."""
    rec = _real_records(1)[0]
    rec["corpus_sha256"] = "0" * 64
    cert = tmp_path / "badcorpus.jsonl"
    _write_cert(cert, [rec])
    rc, out = _run(corpus=_CORPUS, cert=str(cert))
    assert rc == 1, out
    assert "FAILED" in out and "corpus sha" in out


def test_token_ids_tamper_fails(tmp_path):
    """token_ids that don't match the corpus window at prompt_index fail."""
    rec = _real_records(1)[0]
    rec["token_ids"] = [1] + rec["token_ids"][1:]
    cert = tmp_path / "badids.jsonl"
    _write_cert(cert, [rec])
    rc, out = _run(corpus=_CORPUS, cert=str(cert))
    assert rc == 1, out
    assert "FAILED" in out and "token_ids" in out


def test_phi1_forged_false_on_certified_record_fails_schema_validation(tmp_path):
    """A record whose (re-derivable) status is CERTIFIED but whose phi1 is
    forged to False is malformed — schema.validate_record's cross-field
    invariant (phi1 == (status == "CERTIFIED")) rejects it before any
    re-derivation runs, independent of the phi1-blind population-claim
    counting (I1)."""
    rec = _real_records(1)[0]
    assert rec["status"] == "CERTIFIED"
    rec["phi1"] = False
    cert = tmp_path / "phi1_forged_false.jsonl"
    _write_cert(cert, [rec])
    rc, out = _run(corpus=_CORPUS, cert=str(cert))
    assert rc == 1, out
    assert "FAILED" in out and "phi1" in out


def test_phi1_forged_true_on_abstain_record_fails_schema_validation(tmp_path):
    """The symmetric forgery: an ABSTAIN record with phi1 forged to True is
    equally malformed and equally rejected (I1). Built synthetically (none
    of the committed calibration fixtures happen to abstain) — this only
    needs to be schema-shaped, since schema validation runs before any
    weight/corpus re-derivation, so a bogus checkpoint/corpus sha does not
    stop it from reaching (and failing) that check first."""
    rec = schema.build_record(
        model="tinystories", checkpoint_sha256="a" * 64, corpus_sha256="b" * 64,
        prompt_index=0, context_length=8, token_ids=[1, 2, 3],
        precision_P=256, P_max=256, escalation_trace=[256],
        status="ABSTAIN", argmax_token=None, margin_lo=None,
        abstain_reason="width", logit_width_max=Fraction(1, 10),
        top1_top2_float_gap=1e-6, guard_audit={}, runtime_s=1.0,
        peak_rss_mb=1.0, phi1=False, phi2_joint=None,
        harness_transcript_sha256=None, prereg_ref=None,
    )
    rec["phi1"] = True                      # forge
    cert = tmp_path / "phi1_forged_true.jsonl"
    _write_cert(cert, [rec])
    rc, out = _run(corpus=_CORPUS, cert=str(cert))
    assert rc == 1, out
    assert "FAILED" in out and "phi1" in out


def test_checkpoint_sha_mismatch_fails(tmp_path):
    """A record claiming a different checkpoint than the weights export fails."""
    rec = _real_records(1)[0]
    rec["checkpoint_sha256"] = "b" * 64
    cert = tmp_path / "badckpt.jsonl"
    _write_cert(cert, [rec])
    rc, out = _run(corpus=_CORPUS, cert=str(cert))
    assert rc == 1, out
    assert "FAILED" in out and "checkpoint_sha256" in out


# --------------------------------------------------------------------------- #
# headline mode (A2 pre-registration + A6 completeness)
# --------------------------------------------------------------------------- #
def _fixture_corpus(tmp_path, indices):
    """Build a fixture corpus whose ctx8 windows are the real token-id windows
    for `indices` (so the real records re-derive against it)."""
    real = _real_records(max(indices) + 1)
    src = corpus.load(_CORPUS)["windows"]["8"]
    wins = [list(src[i]) for i in indices]
    path = tmp_path / "fixture.ids.json"
    doc = corpus.build_fixed({8: wins}, out_path=str(path))
    return str(path), doc["corpus_sha256"], real


def _freeze_matching(tmp_path, corpus_path, n, target_multiset):
    """Freeze a pre-registration whose drawn sample-index multiset equals
    `target_multiset` (search seeds — the fixture corpus has few windows so a
    hit is found immediately)."""
    from collections import Counter
    out_dir = tmp_path / "freeze"
    for seed in range(2000):
        doc = prereg.freeze(dict(
            model="tinystories", checkpoint_sha256="0" * 64, context_length=8,
            P_max=256, n=n, delta="0.05",
            delta_split={"phi1": "0.025", "phi2_joint": "0.025"}, seed=seed,
            phi_definitions={"phi1": "certified argmax"},
            escalation_policy={"P_grid": [96, 128, 160], "P_max": 256},
        ), corpus_path, str(out_dir))
        idx = json.loads((out_dir / "sample-index.json").read_text())["indices"]
        if Counter(idx) == Counter(target_multiset):
            return str(out_dir / "prereg.json"), doc["prereg_sha256"]
    raise AssertionError("no seed produced the target index multiset")


def _headline_rec(real_rec, corpus_sha, prereg_sha):
    rec = dict(real_rec)
    rec["corpus_sha256"] = corpus_sha
    rec["prereg_ref"] = prereg_sha
    return rec


def test_verified_headline_complete(tmp_path):
    """A headline cert whose prompt_index multiset exactly covers the frozen
    sample index re-derives and passes as a headline (population) claim."""
    corpus_path, csha, real = _fixture_corpus(tmp_path, [0])
    pj, psha = _freeze_matching(tmp_path, corpus_path, n=1, target_multiset=[0])
    rec = _headline_rec(real[0], csha, psha)
    rec["prompt_index"] = 0
    cert = tmp_path / "headline.jsonl"
    _write_cert(cert, [rec])
    rc, out = _run(corpus=corpus_path, cert=str(cert), prereg=pj)
    assert rc == 0, out
    assert "VERIFIED" in out and "headline" in out


# --------------------------------------------------------------------------- #
# Task 2.5: checker announces the verified population bound
# --------------------------------------------------------------------------- #
def test_verified_headline_complete_prints_population_claim_lines(tmp_path):
    """After a full (non-sampled), A6-complete headline VERIFIED pass, the
    checker prints one Hoeffding population-claim line per pre-registered
    property (phi1, phi2_joint) — the checker's own re-derivation is the
    last word on the number a reader would quote."""
    corpus_path, csha, real = _fixture_corpus(tmp_path, [0])
    pj, psha = _freeze_matching(tmp_path, corpus_path, n=1, target_multiset=[0])
    rec = _headline_rec(real[0], csha, psha)
    rec["prompt_index"] = 0
    cert = tmp_path / "headline.jsonl"
    _write_cert(cert, [rec])
    rc, out = _run(corpus=corpus_path, cert=str(cert), prereg=pj)
    assert rc == 0, out
    assert out.count("population claim: p >=") == 2
    assert "(phi1)" in out
    # phi2_joint cannot be re-derived torch-free (harness-leg claim), so its
    # line must carry the trust caveat inline; phi1's line stays clean (I2).
    assert "(phi2_joint; harness leg provenance-audited, not re-derived)" in out
    # the printed line comes strictly after the VERIFIED line (checker's word
    # is the last word)
    assert out.index("VERIFIED") < out.index("population claim")


def test_headline_sampled_mode_suppresses_population_claim(tmp_path):
    """A --sample run only re-derives a subset of records and never asserts
    A6 completeness, so it must never print a population claim even though
    it VERIFIEDs."""
    corpus_path, csha, real = _fixture_corpus(tmp_path, [0])
    pj, psha = _freeze_matching(tmp_path, corpus_path, n=1, target_multiset=[0])
    rec = _headline_rec(real[0], csha, psha)
    rec["prompt_index"] = 0
    cert = tmp_path / "headline.jsonl"
    _write_cert(cert, [rec])
    rc, out = _run(corpus=corpus_path, cert=str(cert), prereg=pj, sample=1)
    assert rc == 0, out
    assert "VERIFIED" in out
    assert "population claim: p >=" not in out


def test_verified_calibration_prints_no_population_claim(tmp_path):
    """A calibration cert (no population claim, A2) never prints a
    population-claim line."""
    cert = tmp_path / "calib.jsonl"
    _write_cert(cert, _real_records(1))
    rc, out = _run(corpus=_CORPUS, cert=str(cert))
    assert rc == 0, out
    assert "population claim: p >=" not in out


def test_headline_dropped_record_fails(tmp_path):
    """A headline cert that silently drops a frozen index fails A6 (no
    --sample), even though the surviving record re-derives fine."""
    corpus_path, csha, real = _fixture_corpus(tmp_path, [0, 1])
    pj, psha = _freeze_matching(tmp_path, corpus_path, n=2, target_multiset=[0, 1])
    rec0 = _headline_rec(real[0], csha, psha)
    rec0["prompt_index"] = 0
    cert = tmp_path / "dropped.jsonl"
    _write_cert(cert, [rec0])                       # index 1 omitted
    rc, out = _run(corpus=corpus_path, cert=str(cert), prereg=pj)
    assert rc == 1, out
    assert "does not cover the pre-registered index set" in out


def test_headline_requires_prereg(tmp_path):
    """A headline cert (non-null prereg_ref) without --prereg is refused."""
    corpus_path, csha, real = _fixture_corpus(tmp_path, [0])
    rec = _headline_rec(real[0], csha, "a" * 64)
    rec["prompt_index"] = 0
    cert = tmp_path / "noprereg.jsonl"
    _write_cert(cert, [rec])
    rc, out = _run(corpus=corpus_path, cert=str(cert))
    assert rc == 1, out
    assert "requires --prereg" in out


def test_mixed_prereg_ref_fails(tmp_path):
    """A cert mixing null and non-null prereg_ref is refused (A2)."""
    corpus_path, csha, real = _fixture_corpus(tmp_path, [0, 1])
    r0 = _headline_rec(real[0], csha, "a" * 64)
    r0["prompt_index"] = 0
    r1 = dict(real[1])
    r1["corpus_sha256"] = csha
    r1["prompt_index"] = 1                          # prereg_ref stays null
    cert = tmp_path / "mixed.jsonl"
    _write_cert(cert, [r0, r1])
    rc, out = _run(corpus=corpus_path, cert=str(cert))
    assert rc == 1, out
    assert "mixed prereg_ref" in out


def test_calibration_with_prereg_flag_fails(tmp_path):
    """Passing --prereg for a calibration cert is refused (no population
    claim)."""
    corpus_path, csha, real = _fixture_corpus(tmp_path, [0])
    pj, _ = _freeze_matching(tmp_path, corpus_path, n=1, target_multiset=[0])
    rec = dict(real[0])
    rec["corpus_sha256"] = csha
    rec["prompt_index"] = 0                         # prereg_ref null
    cert = tmp_path / "calib_pr.jsonl"
    _write_cert(cert, [rec])
    rc, out = _run(corpus=corpus_path, cert=str(cert), prereg=pj)
    assert rc == 1, out
    assert "calibration makes no population claim" in out
