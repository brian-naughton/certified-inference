#!/usr/bin/env python3
"""Rigorous interval forward pass for GPT-2 small (E2: the prediction test).

The measured TinyStories law (~18 bits/layer width amplification) predicts
GPT-2 small (12 layers) certifies at P ~ 300-400. This runs ONE certified
interval forward at P=384 (CLI-overridable) on a seq-8 prompt and reports
certified?/widths/gap/headroom + the per-layer width table + guard audit,
in the same JSON format as the TinyStories runs.

Architecture deltas vs the GPT-Neo harness (all validated against the float
reference gpt2_float.py first):
  - Conv1D layout: weights stored (in, out) — TRANSPOSED vs nn.Linear;
    c_attn fused qkv 768->2304, c_proj 768->768, c_fc 768->3072,
    mlp c_proj 3072->768, ALL with biases.
  - attention scale 1/sqrt(64) = 1/8: EXACT rational (mul_scalar, outward).
  - 12 layers, 12 heads, d_head 64; pre-LN (eps 1e-5) + final ln_f;
    gelu_new; learned wpe; TIED unembedding (wte.T).

Matmul core: per output row the float32 weights are converted EXACTLY to a
common-denominator integer form (N_i / 2**K via frexp; float32 mantissas are
<= 24 bits so the conversion is exact — asserted on startup), the interval
dot is accumulated EXACTLY in units 2**-(P+K) with sign-dispatched endpoint
selection, and a single outward shift produces the result. This is sound and
at least as tight as per-term rounding; enclosure is asserted against
Fraction arithmetic on random data at startup (see also test_ext.py).

Weights are converted per layer and freed afterwards (GPT-2 small has 124M
parameters; converting everything up front is memory-prohibitive in pure
Python).

Full-vocab decision (runtime protocol): after layer 0 the total runtime is
projected; the float top-200 competitor set is always certified, the full
50,257-vocab unembed only if the projection stays under FULL_VOCAB_BUDGET_S.
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
from fractions import Fraction
from math import frexp
from typing import List, Sequence, Tuple

import torch

from certinf import exact
from certinf.exact import Ival
from certinf import ival_ext as E
from certinf.gpt2_float import (GPT2_BIN, N_LAYERS, N_HEADS, D_MODEL, D_HEAD,
                        PROMPT_IDS, load_sd, forward)

LN_EPS = Fraction(1e-5)          # exact dyadic of the double (config eps)
ATTN_SCALE = Fraction(1, 8)      # 1/sqrt(64) — EXACT rational
FULL_VOCAB_BUDGET_S = 2700       # projected-runtime budget for full vocab

VOCAB_SIZE = 50257               # GPT-2 small vocabulary

Row = Tuple[List[int], int]      # (numerators N_i, common shift K): w_i = N_i/2**K


def _choose_competitor_set(n_logits: str, full_vocab: bool | None,
                           require_full: bool, top200: List[int]
                           ) -> Tuple[List[int], str]:
    """Select the certified competitor set (full vocab vs float top-200).

    `require_full` is the certificate-safety flag: a certificate must be
    derived over the FULL vocabulary (top-1 lower bound strictly above EVERY
    other logit's upper bound), so on a certificate-producing call the
    competitor set is forced to the full 50,257 tokens and this function
    RAISES rather than silently degrading to a partial set — closing the
    silent auto-downgrade (top-200 / runtime-budget fallback) that would
    otherwise let a "certificate" rest on an incomplete competitor set.

    Args:
        n_logits: "full" (force full vocab) or "auto" (budget-driven).
        full_vocab: the "auto" budget decision (True/False), or None if the
            budget projection never ran.
        require_full: certificate-safety flag (see above).
        top200: the float top-200 competitor ids (the non-full fallback set).

    Returns:
        (chosen_ids, competitor_description).

    Raises:
        RuntimeError: require_full is set but the resolved set is not the full
            vocabulary (defends against any future partial-set regression).
    """
    if require_full or n_logits == "full" or full_vocab:
        chosen = list(range(VOCAB_SIZE))
        comp_desc = f"FULL VOCAB ({VOCAB_SIZE})"
    else:
        chosen = top200
        comp_desc = "float top-200"
    if require_full and len(chosen) != VOCAB_SIZE:
        raise RuntimeError(
            "require_full certified path resolved to a partial competitor "
            f"set (len {len(chosen)} != vocab {VOCAB_SIZE}); refusing to "
            "certify over an incomplete set")
    return chosen, comp_desc


# --------------------------------------------------------------------------- #
# exact weight conversion + exact-accumulation interval dot
# --------------------------------------------------------------------------- #
def row_to_common(vals: Sequence[float]) -> Row:
    """Exact common-denominator form of a float32-origin weight row.

    v = m * 2**e (frexp, |m| in [0.5,1)); n0 = m * 2**24 is an exact integer
    for any float32 value (mantissa <= 24 bits), so v = n0 / 2**(24-e) exactly.
    Trailing zeros are stripped per element, then all elements are lifted to
    the row max shift K (>= 0), giving w_i = N_i / 2**K exactly.
    """
    ns, ks = [], []
    for v in vals:
        if v == 0.0:
            ns.append(0)
            ks.append(0)
            continue
        m, e = frexp(v)
        n = int(m * (1 << 24))          # exact for float32-origin values
        k = 24 - e
        while n and (n & 1) == 0:
            n >>= 1
            k -= 1
        ns.append(n)
        ks.append(k)
    K = max(0, max(ks))
    return [n << (K - k) if n else 0 for n, k in zip(ns, ks)], K


def convert_matrix_T(t: torch.Tensor) -> List[Row]:
    """Conv1D matrix (in, out) -> per-OUTPUT exact rows (transpose first)."""
    return [row_to_common(r) for r in t.T.contiguous().tolist()]


def bias_ivals(t: torch.Tensor) -> List[Ival]:
    return [Ival.point(Fraction(v)) for v in t.tolist()]


def dot_exact(x: Sequence[Ival], rows: List[Row],
              bias: List[Ival] | None) -> List[Ival]:
    """y_j = sum_i x_i * w_ji (+ b_j); exact accumulation, one outward shift.

    For an exact scalar w and interval x, x*w's endpoints are (xlo*w, xhi*w)
    ordered by sign(w); summing those exact integer products in units
    2**-(P+K) is exact; the final floor/ceil shift by K is the only rounding.
    """
    xlo = [iv.lo_i for iv in x]
    xhi = [iv.hi_i for iv in x]
    out = []
    for j, (N, K) in enumerate(rows):
        lo = 0
        hi = 0
        for xl, xh, n in zip(xlo, xhi, N):
            if n >= 0:
                lo += xl * n
                hi += xh * n
            else:
                lo += xh * n
                hi += xl * n
        iv = Ival(lo >> K, -((-hi) >> K))
        if bias is not None:
            iv = iv + bias[j]
        out.append(iv)
    return out


def _self_test(sd: dict) -> None:
    """Startup soundness checks for the new conversion + dot (see docstring)."""
    # (a) conversion exactness on real weights
    row = sd["h.0.attn.c_attn.weight"].T[0].tolist()
    N, K = row_to_common(row)
    for v, n in zip(row[:512], N[:512]):
        assert Fraction(n, 1 << K) == Fraction(v), "conversion not exact"
    # (b) dot enclosure vs exact Fraction hull on random interval data
    rng = random.Random(1)
    S = exact._SCALE
    for _ in range(50):
        ws = [rng.uniform(-2, 2) for _ in range(16)]
        # simulate float32-origin weights
        ws = [float(torch.tensor(w, dtype=torch.float32)) for w in ws]
        xs = []
        for _ in range(16):
            c = rng.uniform(-3, 3)
            r = abs(rng.gauss(0, 1e-6))
            lo = int(math.floor((c - r) * S))
            hi = int(math.ceil((c + r) * S))
            xs.append(Ival(lo, hi))
        rows = [row_to_common(ws)]
        got = dot_exact(xs, rows, None)[0]
        lo_true = sum(min(Fraction(x.lo_i, S) * Fraction(w),
                          Fraction(x.hi_i, S) * Fraction(w))
                      for x, w in zip(xs, ws))
        hi_true = sum(max(Fraction(x.lo_i, S) * Fraction(w),
                          Fraction(x.hi_i, S) * Fraction(w))
                      for x, w in zip(xs, ws))
        assert Fraction(got.lo_i, S) <= lo_true and \
            Fraction(got.hi_i, S) >= hi_true, "dot_exact not an enclosure"
    print("[self-test] exact conversion + dot enclosure: OK", flush=True)


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
def wstats(res) -> Tuple[float, float]:
    ws = []
    for pos in res:
        for iv in pos:
            d = iv.hi_i - iv.lo_i
            try:
                ws.append(float(Fraction(d, exact._SCALE)))
            except OverflowError:
                ws.append(float("inf"))
    return max(ws), statistics.median(ws)


def interval_forward_gpt2(ids: List[int], n_logits: str = "auto",
                          log=print, sd: dict | None = None,
                          require_full: bool = False) -> dict:
    """Rigorous interval forward for GPT-2 small.

    `require_full=True` (set by the certifier) forces the FULL 50,257-vocab
    competitor set and disables the "auto" runtime-budget downgrade to
    top-200: a certificate is never emitted over a partial competitor set
    (see `_choose_competitor_set`).
    """
    if sd is None:
        sd = load_sd()
    _self_test(sd)
    T = len(ids)
    t0 = time.time()
    scale = exact._SCALE

    # embeddings: exact rational tok + pos
    wte, wpe = sd["wte.weight"], sd["wpe.weight"]
    res = [[Ival.point(Fraction(a) + Fraction(b))
            for a, b in zip(wte[i].tolist(), wpe[t].tolist())]
           for t, i in enumerate(ids)]

    stats = {"sublayer": [], "guard_audit": [], "layer_times_s": []}
    stats["guard_thresholds"] = E.guard_audit_block()

    def record(label, r):
        mx, md = wstats(r)
        stats["sublayer"].append((label, mx, md))
        log(f"[{time.time()-t0:7.1f}s] {label:<26} max_w {mx:.3e}  med_w {md:.3e}",
            flush=True)

    record("embed", res)
    full_vocab = None

    for L in range(N_LAYERS):
        tL = time.time()
        p = f"h.{L}."
        E.reset_tracking()
        # convert this layer's weights exactly (freed at end of layer)
        c_attn = convert_matrix_T(sd[p + "attn.c_attn.weight"])
        c_attn_b = bias_ivals(sd[p + "attn.c_attn.bias"])
        a_proj = convert_matrix_T(sd[p + "attn.c_proj.weight"])
        a_proj_b = bias_ivals(sd[p + "attn.c_proj.bias"])
        c_fc = convert_matrix_T(sd[p + "mlp.c_fc.weight"])
        c_fc_b = bias_ivals(sd[p + "mlp.c_fc.bias"])
        m_proj = convert_matrix_T(sd[p + "mlp.c_proj.weight"])
        m_proj_b = bias_ivals(sd[p + "mlp.c_proj.bias"])
        ln1_g = [Fraction(v) for v in sd[p + "ln_1.weight"].tolist()]
        ln1_b = [Fraction(v) for v in sd[p + "ln_1.bias"].tolist()]
        ln2_g = [Fraction(v) for v in sd[p + "ln_2.weight"].tolist()]
        ln2_b = [Fraction(v) for v in sd[p + "ln_2.bias"].tolist()]

        ln1 = [E.layernorm_ival(res[t], ln1_g, ln1_b, LN_EPS) for t in range(T)]
        record(f"L{L}.ln_1", ln1)
        qkv = [dot_exact(ln1[t], c_attn, c_attn_b) for t in range(T)]
        q = [row[:D_MODEL] for row in qkv]
        k = [row[D_MODEL:2 * D_MODEL] for row in qkv]
        v = [row[2 * D_MODEL:] for row in qkv]

        attn_out = [[None] * D_MODEL for _ in range(T)]
        for h in range(N_HEADS):
            s0 = h * D_HEAD
            for t in range(T):
                scores = []
                for j in range(t + 1):
                    acc = q[t][s0] * k[j][s0]
                    for c in range(1, D_HEAD):
                        acc = acc + q[t][s0 + c] * k[j][s0 + c]
                    scores.append(acc.mul_scalar(ATTN_SCALE))  # 1/8 exact
                probs = E.softmax_ival(scores)
                for c in range(D_HEAD):
                    acc = probs[0] * v[0][s0 + c]
                    for j in range(1, t + 1):
                        acc = acc + probs[j] * v[j][s0 + c]
                    attn_out[t][s0 + c] = acc
        record(f"L{L}.attn(pre-proj)", attn_out)
        proj = [dot_exact(attn_out[t], a_proj, a_proj_b) for t in range(T)]
        res = [[a + b for a, b in zip(res[t], proj[t])] for t in range(T)]
        record(f"L{L}.resid+attn", res)

        ln2 = [E.layernorm_ival(res[t], ln2_g, ln2_b, LN_EPS) for t in range(T)]
        record(f"L{L}.ln_2", ln2)
        hidden = [dot_exact(ln2[t], c_fc, c_fc_b) for t in range(T)]
        hidden = [[E.gelu_new_ival(u) for u in hidden[t]] for t in range(T)]
        record(f"L{L}.gelu", hidden)
        mlp = [dot_exact(hidden[t], m_proj, m_proj_b) for t in range(T)]
        res = [[a + b for a, b in zip(res[t], mlp[t])] for t in range(T)]
        record(f"L{L}.resid+mlp  [LAYER {L}]", res)

        tk = E.get_tracking()
        real = lambda x: float(Fraction(x, scale)) if x is not None else None
        stats["guard_audit"].append({
            "layer": L,
            "softmax_shift_min_x": real(tk["softmax_shift_min"]),
            "softmax_shift_max_x": real(tk["softmax_shift_max"]),
            "gelu_inner_min_x": real(tk["gelu_inner_min"]),
            "gelu_inner_max_x": real(tk["gelu_inner_max"]),
            "exp_guard_hits": tk["exp_guard_hits"],
            "tanh_guard_hits": tk["tanh_guard_hits"],
        })
        del c_attn, a_proj, c_fc, m_proj
        layer_s = time.time() - tL
        stats["layer_times_s"].append(round(layer_s, 1))

        if L == 0 and n_logits == "auto" and not require_full:
            # runtime protocol: project total cost, decide the competitor set.
            # Skipped entirely under require_full (certified path): the full
            # vocab is mandatory there and the budget downgrade must not apply.
            proj_total = layer_s * N_LAYERS * 1.1 + layer_s * 1.6
            full_vocab = proj_total < FULL_VOCAB_BUDGET_S
            log(f"[decision] layer 0 took {layer_s:.0f}s; projected total "
                f"~{proj_total:.0f}s -> "
                f"{'FULL 50,257-vocab' if full_vocab else 'float TOP-200'} "
                f"competitor set", flush=True)

    ln_f_g = [Fraction(v) for v in sd["ln_f.weight"].tolist()]
    ln_f_b = [Fraction(v) for v in sd["ln_f.bias"].tolist()]
    fin = E.layernorm_ival(res[-1], ln_f_g, ln_f_b, LN_EPS)
    record("ln_f(last pos)", [fin])

    # ---- logits: certified competitor set ----
    logits_f = forward(sd, ids, torch.float64)[-1]
    top200 = torch.topk(logits_f, 200).indices.tolist()
    chosen, comp_desc = _choose_competitor_set(n_logits, full_vocab,
                                               require_full, top200)
    xlo = [iv.lo_i for iv in fin]
    xhi = [iv.hi_i for iv in fin]
    logit_ivs = {}
    for i in chosen:
        N, K = row_to_common(wte[i].tolist())
        lo = 0
        hi = 0
        for xl, xh, n in zip(xlo, xhi, N):
            if n >= 0:
                lo += xl * n
                hi += xh * n
            else:
                lo += xh * n
                hi += xl * n
        logit_ivs[i] = Ival(lo >> K, -((-hi) >> K))
    log(f"[{time.time()-t0:7.1f}s] {len(chosen)} logits computed "
        f"({comp_desc})", flush=True)

    lw = [float(Fraction(iv.hi_i - iv.lo_i, scale)) for iv in logit_ivs.values()]
    gap = float(torch.topk(logits_f, 2).values.diff().abs())
    t1 = top200[0]
    sep = all(logit_ivs[t1].lo_i > iv.hi_i
              for i, iv in logit_ivs.items() if i != t1)
    cert_gap = min(float(Fraction(logit_ivs[t1].lo_i - iv.hi_i, scale))
                   for i, iv in logit_ivs.items() if i != t1)
    # exact-Fraction form of cert_gap (additive: same lo_i/hi_i comparison,
    # kept as a Fraction instead of rounded to float — certinf.certify needs
    # an exact certified margin_lo, not an approximation).
    cert_gap_exact = min(
        (Fraction(logit_ivs[t1].lo_i - iv.hi_i, scale)
         for i, iv in logit_ivs.items() if i != t1),
        default=None,
    )
    stats["logits"] = {
        "prompt_ids": ids,
        "competitor_set": comp_desc,
        "widths_max": max(lw), "widths_med": statistics.median(lw),
        "gap_top1_top2_float": gap,
        "certified_gap_lower_bound": cert_gap,
        "certified_gap_lower_bound_exact": cert_gap_exact,
        "headroom_bits": math.log2(gap / max(lw)) if max(lw) > 0 else float("inf"),
        "ratio_maxw_over_gap": max(lw) / gap,
        "argmax_certified_among_chosen": sep,
        "top1_id": t1,
        "top10": [(i, float(logits_f[i]),
                   float(logit_ivs[i].lo), float(logit_ivs[i].hi))
                  for i in top200[:10]],
    }
    stats["runtime_s"] = time.time() - t0
    return stats


if __name__ == "__main__":
    prec = int(sys.argv[1]) if len(sys.argv) > 1 else 384
    nlog = sys.argv[2] if len(sys.argv) > 2 else "auto"
    exact.set_precision(prec)
    E._GELU_C = None
    print(f"=== GPT-2 small interval forward: seq_len={len(PROMPT_IDS)}, "
          f"precision={prec} bits, competitors={nlog} ===", flush=True)
    print(f"prompt ids: {PROMPT_IDS}", flush=True)
    st = interval_forward_gpt2(PROMPT_IDS, n_logits=nlog)
    print(json.dumps(st["logits"], indent=1), flush=True)
    print(f"runtime {st['runtime_s']:.1f}s", flush=True)
    with open(f"widths_gpt2_p{prec}.json", "w") as f:
        json.dump(st, f, indent=1)
