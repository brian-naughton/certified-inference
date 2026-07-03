"""Widths-halve-when-P-doubles canary — a standing soundness/tightness gate.

Enclosure widths track ~2^-P times a precision-INDEPENDENT amplification, so
the certified bit-width -log2(width) roughly doubles as P -> 2P. A bug that
introduces a precision-independent floor (e.g. a fixed Taylor-tail term count,
a hard-coded pi string) leaves a sublayer whose width stops shrinking. This
gate runs one reference forward at P and one at 2P and asserts every sublayer
keeps tracking 2^-P: width(2P) <= width(P) * 2^-(P - margin). Caught 2 of 4
foothold bugs; now standing. Widths here are ENCLOSURE widths of the exact-real
quantities; PyTorch is not on this path.

Exact-real caveat: the certified object is the exact-real transformer (float32
weights read as exact dyadic rationals, evaluated with real arithmetic);
PyTorch/float32 runs are conformance evidence only, never the theorem.
"""
from __future__ import annotations

import math

from certinf import exact
from certinf import ival_ext as E


def _run(model: str, ids: list[int], P: int) -> dict:
    exact.set_precision(P)
    # Dead backward-compat shim (Task 0.2): the real sqrt(2/pi) cache in
    # ival_ext.py is now keyed per-precision (E._GELU_C_CACHE), so
    # set_precision() alone suffices and this line is a harmless no-op.
    # Left belt-and-braces.
    E._GELU_C = None
    if model == "tinystories":
        from certinf.interval_fwd import interval_forward, prepare_weights
        from certinf.float_fwd import load_sd
        sd = load_sd()
        W = prepare_weights(sd)
        st = interval_forward(len(ids), n_logits=200, log=lambda *a, **k: None,
                              ids=ids, sd=sd, W=W)
    elif model == "gpt2":
        from certinf.gpt2_interval import interval_forward_gpt2
        st = interval_forward_gpt2(ids, n_logits=200, log=lambda *a, **k: None)
    else:
        raise ValueError(model)
    return st


def width_profile(model: str, ids: list[int], P: int) -> dict:
    st = _run(model, ids, P)
    nl2 = {}
    for label, max_w, _med in st["sublayer"]:
        nl2[label] = (-math.log2(max_w)) if max_w > 0 else float("inf")
    nl2["logits"] = (-math.log2(st["logits"]["widths_max"])
                     if st["logits"]["widths_max"] > 0 else float("inf"))
    return {"sublayer": st["sublayer"], "neg_log2": nl2,
            "logits_max_w": st["logits"]["widths_max"]}


def assert_no_precision_floor(model: str, ids: list[int], P: int,
                              margin: int = 8) -> dict:
    """Assert every sublayer width at 2P is <= width(P) * 2^-(P - margin).

    Equivalently -log2(width) must grow by at least (P - margin) from P to 2P.
    """
    lo = width_profile(model, ids, P)
    hi = width_profile(model, ids, 2 * P)
    per = []
    first_bad = None
    for label, nl_lo in lo["neg_log2"].items():
        nl_hi = hi["neg_log2"].get(label, nl_lo)
        if math.isinf(nl_hi):
            # width at 2P is EXACTLY zero (e.g. point-value embeddings): no
            # floor is possible by construction, regardless of nl_lo.
            gained = float("inf")
            ok = True
        else:
            gained = nl_hi - nl_lo           # bits of width recovered by doubling P
            ok = gained >= (P - margin)
        per.append({"label": label, "nl2_P": nl_lo, "nl2_2P": nl_hi,
                    "gained_bits": gained, "ok": ok})
        if not ok and first_bad is None:
            first_bad = (label, gained)
    exact.set_precision(P)                   # restore
    if first_bad is not None:
        raise AssertionError(
            f"precision floor at sublayer {first_bad[0]!r}: doubling P={P}->{2*P} "
            f"recovered only {first_bad[1]:.1f} bits (< P-margin={P-margin}); "
            f"a bug-introduced floor is the likely cause")
    return {"P": P, "per_sublayer": per}
