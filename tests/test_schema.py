from fractions import Fraction

import pytest

from certinf import schema


def _base_kwargs(**overrides):
    kwargs = dict(
        model="tinystories",
        checkpoint_sha256="a" * 64,
        corpus_sha256="b" * 64,
        prompt_index=0,
        context_length=8,
        token_ids=[7454, 2402, 257, 640, 612, 373, 257, 1310],
        precision_P=192,
        P_max=256,
        escalation_trace=[128, 160, 192],
        status="CERTIFIED",
        argmax_token=612,
        margin_lo=Fraction(1, 3),
        abstain_reason=None,
        logit_width_max=Fraction(1, 1 << 100),
        top1_top2_float_gap=0.42,
        guard_audit={"exp_guard_hits": 0, "tanh_guard_hits": 0},
        runtime_s=6.9,
        peak_rss_mb=123.4,
        phi1=True,
        phi2_joint=None,
        harness_transcript_sha256=None,
        prereg_ref=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_fully_populated_record_validates():
    rec = schema.build_record(**_base_kwargs())
    schema.validate_record(rec)  # no raise
    assert rec["schema_version"] == schema.SCHEMA_VERSION


def test_dropping_required_key_raises_naming_it():
    rec = schema.build_record(**_base_kwargs())
    del rec["runtime_s"]
    with pytest.raises(ValueError, match="runtime_s"):
        schema.validate_record(rec)


def test_abstain_with_nonnull_argmax_token_raises():
    rec = schema.build_record(**_base_kwargs(
        status="ABSTAIN", argmax_token=None, margin_lo=None,
        abstain_reason="near-tie", phi1=False,
    ))
    rec["argmax_token"] = 612  # corrupt after the fact
    with pytest.raises(ValueError):
        schema.validate_record(rec)


def test_certified_with_nonnull_abstain_reason_raises():
    rec = schema.build_record(**_base_kwargs())
    rec["abstain_reason"] = "near-tie"  # corrupt after the fact
    with pytest.raises(ValueError):
        schema.validate_record(rec)


def test_phi2_joint_true_without_transcript_sha_raises():
    rec = schema.build_record(**_base_kwargs())
    rec["phi2_joint"] = True
    with pytest.raises(ValueError, match="harness_transcript_sha256"):
        schema.validate_record(rec)


def test_calibration_record_validates_but_headline_check_raises():
    rec = schema.build_record(**_base_kwargs(prereg_ref=None))
    schema.validate_record(rec)  # calibration record is fine
    with pytest.raises(ValueError, match="prereg_ref"):
        schema.validate_headline_record(rec)


def test_headline_record_with_prereg_ref_validates():
    rec = schema.build_record(**_base_kwargs(prereg_ref="c" * 64))
    schema.validate_headline_record(rec)  # no raise


# --------------------------------------------------------------------------- #
# I2(chunk1): invalid enum values must raise.
# --------------------------------------------------------------------------- #
def test_invalid_status_raises():
    rec = schema.build_record(**_base_kwargs())
    rec["status"] = "MAYBE"
    with pytest.raises(ValueError, match="status"):
        schema.validate_record(rec)


def test_invalid_abstain_reason_raises():
    rec = schema.build_record(**_base_kwargs(
        status="ABSTAIN", argmax_token=None, margin_lo=None,
        abstain_reason="near-tie", phi1=False,
    ))
    rec["abstain_reason"] = "gremlin"
    with pytest.raises(ValueError, match="abstain_reason"):
        schema.validate_record(rec)


# --------------------------------------------------------------------------- #
# M4(chunk1): schema hardening — version, unknown keys, types, sha shape.
# --------------------------------------------------------------------------- #
def test_unrecognised_schema_version_raises():
    rec = schema.build_record(**_base_kwargs())
    rec["schema_version"] = "9.9"
    with pytest.raises(ValueError, match="schema_version"):
        schema.validate_record(rec)


def test_unknown_top_level_key_raises():
    rec = schema.build_record(**_base_kwargs())
    rec["surprise"] = 1
    with pytest.raises(ValueError, match="unknown top-level key"):
        schema.validate_record(rec)


def test_non_int_token_ids_raises():
    rec = schema.build_record(**_base_kwargs())
    rec["token_ids"] = [1, 2, "three"]
    with pytest.raises(ValueError, match="token_ids"):
        schema.validate_record(rec)


def test_non_numeric_runtime_s_raises():
    rec = schema.build_record(**_base_kwargs())
    rec["runtime_s"] = "fast"
    with pytest.raises(ValueError, match="runtime_s"):
        schema.validate_record(rec)


def test_malformed_sha_field_raises():
    rec = schema.build_record(**_base_kwargs())
    rec["checkpoint_sha256"] = "not-a-sha"
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        schema.validate_record(rec)
