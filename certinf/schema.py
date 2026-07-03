"""Versioned per-sample certificate JSON schema.

Exact-real caveat: the certified object is the exact-real transformer — each
float32 weight read as its exact dyadic rational, real matrix sums, real
exp/division in softmax, real tanh/gelu_new, argmax over exact-real logits
with a positive certified margin. Interval soundness is analytic (outward
rounding). PyTorch/float32 runs are conformance evidence only, never the
theorem. "Exact-real semantics" is the reals denoted by the checkpoint's
stored float32 tensor payload, decoder + sha256 pinned. Argmax claims state
uniqueness explicitly: the certified top-1 lower bound strictly exceeds every
other logit's upper bound over the full vocabulary.

Abstention is never hidden: an ABSTAIN record is a result, not a failure, and
every abstain record carries an A4 taxonomy reason (near-tie / width / guard /
bug / timeout).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction

SCHEMA_VERSION = "1.0"

_STATUSES = {"CERTIFIED", "ABSTAIN"}
_ABSTAIN_REASONS = {"near-tie", "width", "guard", "bug", "timeout"}

_REQUIRED_FIELDS = (
    "schema_version", "model", "checkpoint_sha256", "corpus_sha256",
    "prompt_index", "context_length", "token_ids",
    "precision_P", "P_max", "escalation_trace",
    "status", "argmax_token", "margin_lo", "abstain_reason",
    "logit_width_max", "top1_top2_float_gap", "guard_audit",
    "runtime_s", "peak_rss_mb", "phi1", "phi2_joint",
    "harness_transcript_sha256", "prereg_ref",
)


def _frac_to_pair(v) -> list[str]:
    """Serialise a Fraction as ["num", "den"] decimal strings (arbitrary
    precision — the same discipline as certified-grokking)."""
    f = v if isinstance(v, Fraction) else Fraction(v)
    return [str(f.numerator), str(f.denominator)]


def _pair_to_frac(pair) -> Fraction:
    num, den = pair
    return Fraction(int(num), int(den))


@dataclass
class PerSampleRecord:
    """A single certified/abstain record — the certificate schema's unit."""

    schema_version: str
    model: str
    checkpoint_sha256: str
    corpus_sha256: str
    prompt_index: int
    context_length: int
    token_ids: list
    precision_P: int
    P_max: int
    escalation_trace: list
    status: str
    argmax_token: int | None
    margin_lo: list | None            # ["num","den"] or None
    abstain_reason: str | None
    logit_width_max: list             # ["num","den"]
    top1_top2_float_gap: float
    guard_audit: dict
    runtime_s: float
    peak_rss_mb: float
    phi1: bool
    phi2_joint: bool | None
    harness_transcript_sha256: str | None
    prereg_ref: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def build_record(
    *,
    model: str,
    checkpoint_sha256: str,
    corpus_sha256: str,
    prompt_index: int,
    context_length: int,
    token_ids: list,
    precision_P: int,
    P_max: int,
    escalation_trace: list,
    status: str,
    argmax_token: int | None,
    margin_lo,
    abstain_reason: str | None,
    logit_width_max,
    top1_top2_float_gap: float,
    guard_audit: dict,
    runtime_s: float,
    peak_rss_mb: float,
    phi1: bool,
    phi2_joint: bool | None = None,
    harness_transcript_sha256: str | None = None,
    prereg_ref: str | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> dict:
    """Build a schema-version-tagged per-sample certificate record (a dict).

    `margin_lo` / `logit_width_max` accept a Fraction, a ["num","den"] pair,
    or None (margin_lo only) and are serialised as ["num","den"] decimal
    strings.
    """
    rec = PerSampleRecord(
        schema_version=schema_version,
        model=model,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=corpus_sha256,
        prompt_index=prompt_index,
        context_length=context_length,
        token_ids=list(token_ids),
        precision_P=precision_P,
        P_max=P_max,
        escalation_trace=list(escalation_trace),
        status=status,
        argmax_token=argmax_token,
        margin_lo=(None if margin_lo is None else
                   (margin_lo if isinstance(margin_lo, list) else _frac_to_pair(margin_lo))),
        abstain_reason=abstain_reason,
        logit_width_max=(logit_width_max if isinstance(logit_width_max, list)
                         else _frac_to_pair(logit_width_max)),
        top1_top2_float_gap=float(top1_top2_float_gap),
        guard_audit=guard_audit,
        runtime_s=float(runtime_s),
        peak_rss_mb=float(peak_rss_mb),
        phi1=bool(phi1),
        phi2_joint=phi2_joint,
        harness_transcript_sha256=harness_transcript_sha256,
        prereg_ref=prereg_ref,
    )
    d = rec.to_dict()
    validate_record(d)
    return d


def validate_record(rec: dict) -> None:
    """Raise ValueError naming the missing/ill-typed field, else return None.

    Enforces the cross-field invariants: status/argmax/abstain_reason
    consistency; phi2_joint=True requires a harness transcript sha; phi1 is
    equivalent to status == CERTIFIED.
    """
    for f in _REQUIRED_FIELDS:
        if f not in rec:
            raise ValueError(f"missing required field: {f!r}")

    if rec["status"] not in _STATUSES:
        raise ValueError(f"invalid status: {rec['status']!r} (must be one of {_STATUSES})")

    if not isinstance(rec["prompt_index"], int):
        raise ValueError("prompt_index must be an int")
    if not isinstance(rec["context_length"], int):
        raise ValueError("context_length must be an int")
    if not isinstance(rec["precision_P"], int):
        raise ValueError("precision_P must be an int")
    if not isinstance(rec["P_max"], int):
        raise ValueError("P_max must be an int")
    if not isinstance(rec["escalation_trace"], list) or not rec["escalation_trace"]:
        raise ValueError("escalation_trace must be a non-empty list")

    if rec["status"] == "CERTIFIED":
        if rec["argmax_token"] is None:
            raise ValueError("status CERTIFIED requires a non-null argmax_token")
        if rec["margin_lo"] is None:
            raise ValueError("status CERTIFIED requires a non-null margin_lo")
        if rec["abstain_reason"] is not None:
            raise ValueError("status CERTIFIED requires a null abstain_reason "
                             f"(got {rec['abstain_reason']!r})")
        if rec["phi1"] is not True:
            raise ValueError("status CERTIFIED requires phi1=True")
    else:  # ABSTAIN
        if rec["argmax_token"] is not None:
            raise ValueError("status ABSTAIN requires a null argmax_token "
                             f"(got {rec['argmax_token']!r})")
        if rec["margin_lo"] is not None:
            raise ValueError("status ABSTAIN requires a null margin_lo")
        if rec["abstain_reason"] not in _ABSTAIN_REASONS:
            raise ValueError(f"status ABSTAIN requires abstain_reason in "
                             f"{_ABSTAIN_REASONS} (got {rec['abstain_reason']!r})")
        if rec["phi1"] is not False:
            raise ValueError("status ABSTAIN requires phi1=False")

    if rec["phi2_joint"] is True and rec["harness_transcript_sha256"] is None:
        raise ValueError("phi2_joint=True requires a non-null harness_transcript_sha256 "
                         "(A1: no transcript => no joint claim)")

    if rec["logit_width_max"] is None or not (
        isinstance(rec["logit_width_max"], list) and len(rec["logit_width_max"]) == 2
    ):
        raise ValueError("logit_width_max must be a [\"num\",\"den\"] pair")
    if rec["margin_lo"] is not None and not (
        isinstance(rec["margin_lo"], list) and len(rec["margin_lo"]) == 2
    ):
        raise ValueError("margin_lo must be null or a [\"num\",\"den\"] pair")


def validate_headline_record(rec: dict) -> None:
    """As validate_record, plus A2: a headline record must carry a non-null
    prereg_ref (calibration records use prereg_ref=None; headline never
    does)."""
    validate_record(rec)
    if rec.get("prereg_ref") is None:
        raise ValueError("headline record requires a non-null prereg_ref (A2: "
                         "calibration samples are never headline samples)")
