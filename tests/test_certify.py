import os
import time
from fractions import Fraction

import pytest

from certinf import certify, schema
from certinf.float_fwd import MODEL_BIN

FIXTURE_CORPUS = os.path.join(os.path.dirname(__file__), "fixtures",
                              "tinystories-dev.ids.json")


def _fake_gpt2_certified_st(*, exact_margin=Fraction(1, 3),
                            float_margin=0.3333333):
    """A minimal interval_forward_gpt2 return value that certifies, carrying a
    DELIBERATELY non-dyadic exact Fraction margin (1/3, impossible to obtain
    from a float) alongside a distinct rounded float, so a test can prove the
    certifier records the exact value and not the float."""
    return {
        "logits": {
            "competitor_set": "FULL VOCAB (50257)",
            "gap_top1_top2_float": 5.0,
            "widths_max": 1e-30,
            "argmax_certified_among_chosen": True,
            "top1_id": 42,
            "certified_gap_lower_bound": float_margin,
            "certified_gap_lower_bound_exact": exact_margin,
        },
        "guard_thresholds": {"precision_bits": 320},
        "guard_audit": [],
    }


def test_certify_sample_returns_schema_valid_record():
    rec = certify.certify_sample(
        "tinystories", MODEL_BIN, FIXTURE_CORPUS, prompt_index=0,
        P_grid=[128, 160, 192], P_max=192,
    )
    schema.validate_record(rec)  # no raise
    assert rec["status"] in {"CERTIFIED", "ABSTAIN"}


def test_foothold_baseline_prompt_certifies():
    """The foothold seq-8 'Once upon a time there was a little' prompt (dev
    fixture index 0) must certify at P=192, with phi1=True, a positive
    margin_lo, and an escalation_trace starting at 128 and stopping at the
    first certifying P."""
    t0 = time.time()
    rec = certify.certify_sample(
        "tinystories", MODEL_BIN, FIXTURE_CORPUS, prompt_index=0,
        P_grid=[128, 160, 192], P_max=192,
    )
    elapsed = time.time() - t0
    assert rec["status"] == "CERTIFIED"
    assert rec["phi1"] is True
    assert rec["margin_lo"] is not None
    num, den = rec["margin_lo"]
    assert int(num) > 0 and int(den) > 0
    assert rec["escalation_trace"][0] == 128
    assert rec["escalation_trace"][-1] == rec["precision_P"]
    print(f"\n[timing] foothold seq-8 certify_sample: {elapsed:.1f}s")


def test_escalation_trace_stops_at_first_certifying_P():
    """This prompt abstains (width) at P=128 and certifies at P=160 (measured
    directly against the engine); the escalation loop must not try P=192."""
    rec = certify.certify_sample(
        "tinystories", MODEL_BIN, FIXTURE_CORPUS, prompt_index=0,
        P_grid=[128, 160, 192], P_max=192,
    )
    assert rec["status"] == "CERTIFIED"
    assert rec["escalation_trace"] == [128, 160]
    assert rec["precision_P"] == 160


# --------------------------------------------------------------------------- #
# C1: GPT-2 CERTIFIED path must compare the FULL vocabulary, never top-200.
# --------------------------------------------------------------------------- #
def test_certify_gpt2_dispatches_full_vocab_and_require_full(monkeypatch):
    """certify_sample('gpt2', ...) must call the GPT-2 engine with the string
    'full' competitor set AND require_full=True. Under the pre-fix code the
    int sentinel n_logits=-1 was passed (which the engine silently downgraded
    to top-200), so this test fails on the bug it guards."""
    from certinf import gpt2_interval
    captured = {}

    def fake_forward(ids, n_logits="auto", log=None, sd=None,
                     require_full=False):
        captured["n_logits"] = n_logits
        captured["require_full"] = require_full
        return _fake_gpt2_certified_st()

    monkeypatch.setattr(gpt2_interval, "interval_forward_gpt2", fake_forward)
    rec = certify.certify_sample(
        "gpt2", MODEL_BIN, FIXTURE_CORPUS, prompt_index=0,
        P_grid=[320], P_max=320, run_canary=False,
    )
    assert captured["n_logits"] == "full"
    assert captured["require_full"] is True
    assert rec["status"] == "CERTIFIED"
    schema.validate_record(rec)


def test_choose_competitor_set_require_full_forces_full_vocab():
    """The engine helper must resolve require_full to the full vocab even when
    the 'auto' runtime budget said top-200 (full_vocab=False). Under a top-200
    regression the chosen set would be length 200, so this test would fail."""
    from certinf import gpt2_interval as G
    chosen, desc = G._choose_competitor_set(
        "auto", full_vocab=False, require_full=True,
        top200=list(range(200)))
    assert len(chosen) == G.VOCAB_SIZE == 50257
    assert set(chosen) == set(range(50257))
    assert "FULL VOCAB" in desc


def test_choose_competitor_set_non_certified_can_be_top200():
    """Without require_full and with the budget declining full vocab, the
    (non-certificate) fallback is still top-200 — documents that the guard is
    scoped to the certified path only."""
    from certinf import gpt2_interval as G
    chosen, desc = G._choose_competitor_set(
        "auto", full_vocab=False, require_full=False,
        top200=list(range(200)))
    assert len(chosen) == 200
    assert desc == "float top-200"


def test_certify_gpt2_margin_lo_is_exact_fraction(monkeypatch):
    """I2(chunk2): the recorded margin_lo must be the EXACT Fraction
    (certified_gap_lower_bound_exact), not the rounded float. The fake carries
    exact=1/3; a denominator of 3 cannot arise from any float, so this proves
    the exact key was read."""
    from certinf import gpt2_interval
    monkeypatch.setattr(gpt2_interval, "interval_forward_gpt2",
                        lambda *a, **k: _fake_gpt2_certified_st())
    rec = certify.certify_sample(
        "gpt2", MODEL_BIN, FIXTURE_CORPUS, prompt_index=0,
        P_grid=[320], P_max=320, run_canary=False,
    )
    assert rec["status"] == "CERTIFIED"
    num, den = rec["margin_lo"]
    assert Fraction(int(num), int(den)) == Fraction(1, 3)


# --------------------------------------------------------------------------- #
# M7(chunk2): a genuinely non-separating prompt must ABSTAIN, never certify.
# --------------------------------------------------------------------------- #
def test_non_separating_prompt_abstains():
    """A genuinely non-separating case must ABSTAIN, never certify. The
    repeated-token fixture (index 2) turns out to separate cleanly, so instead
    we starve the engine of precision (P=32): at 32 bits the interval widths
    swamp the top1-vs-top2 gap, so the certifier cannot separate the argmax and
    must return ABSTAIN(width) with a null argmax/margin — the certifier refuses
    to emit an unverified certificate rather than certifying on slack widths."""
    rec = certify.certify_sample(
        "tinystories", MODEL_BIN, FIXTURE_CORPUS, prompt_index=0,
        P_grid=[32], P_max=32, run_canary=False,
    )
    schema.validate_record(rec)
    assert rec["status"] == "ABSTAIN"
    assert rec["phi1"] is False
    assert rec["argmax_token"] is None
    assert rec["margin_lo"] is None
    assert rec["abstain_reason"] == "width"
