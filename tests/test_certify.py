import os
import time

from certinf import certify, schema
from certinf.float_fwd import MODEL_BIN

FIXTURE_CORPUS = os.path.join(os.path.dirname(__file__), "fixtures",
                              "tinystories-dev.ids.json")


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
