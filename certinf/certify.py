"""Per-sample certifier: full-vocabulary certified argmax + adaptive precision
escalation + A4 abstain taxonomy.

Exact-real caveat: the certified object is the exact-real transformer (float32
weights read as exact dyadic rationals, evaluated with real arithmetic,
argmax over exact-real logits with a positive certified margin). Interval
soundness is analytic (outward rounding). PyTorch/float32 runs are
conformance evidence only, never the theorem. The certified claim
(`argmax_token`, `margin_lo`) is derived PURELY from the interval endpoints
(the same `lo_i > hi_i` comparison the audited engine performs) — the float64
top-1 is used only to pick which vocabulary token to test for separation
(a fast, sound-preserving heuristic: if that candidate's interval doesn't
separate, the sample abstains rather than certifying an unverified claim) and
to feed the A4 abstain taxonomy (`top1_top2_float_gap` is diagnostic only,
never part of the certified claim itself).

Abstention is never hidden: an ABSTAIN record is a result, not a failure.

Widths canary: `canary.assert_no_precision_floor` is asserted once per
sample (per this module's own budget — see Task 1.3 which instead asserts it
once per calibration cell for speed; both satisfy the widths-canary-on-every-
run constraint at different granularities, documented here and in
certinf/grid.py).
"""
from __future__ import annotations

import hashlib
import resource
import sys
import time
from fractions import Fraction

from certinf import canary, corpus, exact, schema
from certinf import ival_ext as E

# Below this absolute float64-computed top1-vs-top2 logit gap, a non-
# certified sample is classified as a genuine near-tie (abstain_reason=
# "near-tie") rather than an engine-precision shortfall (abstain_reason=
# "width"): gaps this small are far below what the measured ~17-24 bits/layer
# amplification could plausibly resolve by escalating P within any practical
# budget, so they are treated as reflecting proximity in the model's own
# real-valued function rather than engine slack. This is a documented design
# choice (the brief describes the taxonomy qualitatively, not with exact
# thresholds); see docs/PROVENANCE.md for the constant's rationale.
NEAR_TIE_ABS_GAP = 1e-4


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _peak_rss_mb() -> float:
    """Process peak RSS in MB (ru_maxrss is bytes on macOS/BSD, KB on Linux)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = (1024 * 1024) if sys.platform == "darwin" else 1024
    return raw / divisor


def _run_interval_forward(model: str, ids: list[int], weights_path: str,
                          n_logits: int, sd_cache: dict) -> dict:
    """Run one full interval forward at the currently-set precision.

    `sd_cache` memoises the loaded state dict (and, for TinyStories, the
    precision-independent prepared weights) across escalation steps within
    one certify_sample call — the weights themselves do not depend on P.
    """
    if model == "tinystories":
        import torch

        from certinf.interval_fwd import interval_forward, prepare_weights
        if "sd" not in sd_cache:
            sd_cache["sd"] = torch.load(weights_path, map_location="cpu",
                                        weights_only=True)
            sd_cache["W"] = prepare_weights(sd_cache["sd"])
        return interval_forward(len(ids), n_logits=n_logits,
                                log=lambda *a, **k: None, ids=ids,
                                sd=sd_cache["sd"], W=sd_cache["W"])
    if model == "gpt2":
        import torch

        from certinf.gpt2_interval import interval_forward_gpt2
        if "sd" not in sd_cache:
            sd_cache["sd"] = torch.load(weights_path, map_location="cpu",
                                        weights_only=True)
        return interval_forward_gpt2(ids, n_logits=n_logits,
                                     log=lambda *a, **k: None,
                                     sd=sd_cache["sd"])
    raise ValueError(f"unknown model: {model!r}")


def _classify_abstain(float_gap: float, width_max: float) -> str:
    """A4 taxonomy for a non-certified sample: 'near-tie' (genuine model
    near-tie, escalation unlikely to help) vs 'width' (engine precision
    shortfall — escalate). See NEAR_TIE_ABS_GAP for the threshold rationale."""
    if float_gap < NEAR_TIE_ABS_GAP:
        return "near-tie"
    return "width"


def certify_sample(
    model: str,
    weights_path: str,
    corpus_path: str,
    prompt_index: int,
    P_grid: list[int],
    P_max: int,
    prereg_ref: str | None = None,
    run_harness: bool = False,
    context_length: int = 8,
    n_logits: int = -1,
    run_canary: bool = True,
) -> dict:
    """Certify one prompt's full-vocabulary next-token argmax, escalating P.

    `context_length` selects which window list in the corpus to draw
    `prompt_index` from (not part of the brief's literal call signature, but
    required to resolve a corpus with multiple context lengths; defaults to
    8, the headline TinyStories-1M context, so the brief's exact call
    `certify_sample("tinystories", weights, corpus, prompt_index=0,
    P_grid=[128,160,192], P_max=192)` is unaffected).
    """
    t0 = time.time()
    corpus_doc = corpus.load(corpus_path)
    corpus_sha256 = corpus_doc["corpus_sha256"]
    windows = corpus_doc["windows"].get(str(context_length))
    if windows is None:
        raise ValueError(f"corpus {corpus_path!r} has no windows at "
                         f"context_length={context_length}")
    ids = windows[prompt_index]
    checkpoint_sha256 = _sha256_file(weights_path)

    sd_cache: dict = {}
    escalation_trace: list[int] = []
    P_list = [p for p in P_grid if p <= P_max]
    if not P_list:
        raise ValueError(f"P_grid {P_grid!r} has no entries <= P_max={P_max}")

    status = "ABSTAIN"
    argmax_token = None
    margin_lo = None
    abstain_reason = "bug"          # overwritten below unless something throws
    top1_top2_float_gap = 0.0
    logit_width_max = Fraction(0)
    guard_audit: dict = {}
    phi1 = False
    canary_result = None

    for P in P_list:
        escalation_trace.append(P)
        exact.set_precision(P)
        E._GELU_C = None   # invalidate stale sqrt(2/pi) cache (Task 0.2's
                           # precision-keyed fix is not yet in this tree)

        if run_canary and canary_result is None:
            # Once per sample (this module's chosen granularity — Task 1.3's
            # grid runner instead asserts it once per calibration cell for
            # speed; both satisfy the widths-canary constraint).
            try:
                canary_result = canary.assert_no_precision_floor(model, ids, P)
            except AssertionError:
                # A tripped canary means the enclosures at this precision are
                # not trustworthy — abstain, never certify on unaudited widths.
                status, abstain_reason, phi1 = "ABSTAIN", "guard", False
                break

        try:
            st = _run_interval_forward(model, ids, weights_path, n_logits, sd_cache)
        except Exception:
            status, abstain_reason, phi1 = "ABSTAIN", "bug", False
            break

        lg = st["logits"]
        top1_top2_float_gap = float(lg["gap_top1_top2"]
                                    if "gap_top1_top2" in lg
                                    else lg["gap_top1_top2_float"])
        logit_width_max = Fraction(lg["widths_max"])
        guard_audit = {"guard_thresholds": st["guard_thresholds"],
                       "per_layer": st["guard_audit"]}

        if lg["argmax_certified_among_chosen"]:
            status = "CERTIFIED"
            argmax_token = lg["top1_id"]
            margin_lo = lg.get("certified_gap_lower_bound")
            abstain_reason = None
            phi1 = True
            break

        reason = _classify_abstain(top1_top2_float_gap, float(logit_width_max))
        abstain_reason = reason
        if reason != "width":
            break   # near-tie is terminal: escalation is not expected to help
        # else: "width" -> continue the ladder (cap at P_max, already applied)

    runtime_s = time.time() - t0
    peak_rss_mb = _peak_rss_mb()

    phi2_joint = None
    harness_transcript_sha256 = None
    if run_harness:
        from certinf import harness

        harness_transcript_sha256 = harness.transcript_sha256()
        harness_top1 = harness.top1(model, weights_path, ids)
        phi2_joint = bool(status == "CERTIFIED" and harness_top1 == argmax_token)

    return schema.build_record(
        model=model,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=corpus_sha256,
        prompt_index=prompt_index,
        context_length=context_length,
        token_ids=ids,
        precision_P=escalation_trace[-1],
        P_max=P_max,
        escalation_trace=escalation_trace,
        status=status,
        argmax_token=argmax_token,
        margin_lo=margin_lo,
        abstain_reason=abstain_reason,
        logit_width_max=logit_width_max,
        top1_top2_float_gap=top1_top2_float_gap,
        guard_audit=guard_audit,
        runtime_s=runtime_s,
        peak_rss_mb=peak_rss_mb,
        phi1=phi1,
        phi2_joint=phi2_joint,
        harness_transcript_sha256=harness_transcript_sha256,
        prereg_ref=prereg_ref,
    )
