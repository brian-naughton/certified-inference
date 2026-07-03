#!/usr/bin/env python3
"""Rigorous interval forward pass for TinyStories-1M; per-sublayer width log.

Weights (float32) are treated as exact dyadic rationals: w = n / 2**k, so
matmul accumulates via integer shifts with outward rounding per term.
Semantics mirror float_fwd.py (GPT-Neo: no qkv bias, no attention scaling,
causal mask = exact -inf in the interval version, gelu_new, LN eps 1e-5,
tied unembedding).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from fractions import Fraction
from typing import List, Sequence, Tuple

import torch

from certinf import exact
from certinf.exact import Ival, _ceil_div
from certinf import ival_ext as E
from certinf.float_fwd import (MODEL_BIN, N_LAYERS, N_HEADS, D_MODEL, D_HEAD,
                       PROMPT_IDS_16, load_sd, forward)

LN_EPS = Fraction(1e-5)   # exact dyadic of the double
WPair = Tuple[int, int]   # (numerator n, shift k): weight = n / 2**k


def to_pairs_mat(t: torch.Tensor) -> List[List[WPair]]:
    """Convert a float32 matrix to exact dyadic (n, k) pairs, row-major."""
    out = []
    for row in t.tolist():
        r = []
        for v in row:
            f = Fraction(v)
            r.append((f.numerator, f.denominator.bit_length() - 1))
        out.append(r)
    return out


def to_fracs(t: torch.Tensor) -> List[Fraction]:
    return [Fraction(v) for v in t.tolist()]


def dot_rows(x: Sequence[Ival], W: List[List[WPair]],
             bias: List[Fraction] | None) -> List[Ival]:
    """y_j = sum_i x_i * W[j][i] (+ b_j), outward rounding per term."""
    out = []
    for j, row in enumerate(W):
        lo = 0
        hi = 0
        for xi, (n, k) in zip(x, row):
            if n >= 0:
                lo += (xi.lo_i * n) >> k
                hi += -((-(xi.hi_i * n)) >> k)
            else:
                lo += (xi.hi_i * n) >> k
                hi += -((-(xi.lo_i * n)) >> k)
        iv = Ival(lo, hi)
        if bias is not None:
            iv = iv + Ival.point(bias[j])
        out.append(iv)
    return out


def wstats(res: List[List[Ival]]) -> Tuple[float, float]:
    """(max, median) enclosure width across all positions/dims."""
    ws = []
    for pos in res:
        for iv in pos:
            d = iv.hi_i - iv.lo_i
            try:
                ws.append(float(Fraction(d, exact._SCALE)))
            except OverflowError:
                ws.append(float("inf"))
    return max(ws), statistics.median(ws)


def prepare_weights(sd: dict) -> dict:
    """Convert the state dict to exact dyadic pairs / fractions (reusable)."""
    W = {}
    for L in range(N_LAYERS):
        p = f"transformer.h.{L}."
        for name in ("attn.attention.q_proj.weight", "attn.attention.k_proj.weight",
                     "attn.attention.v_proj.weight", "attn.attention.out_proj.weight",
                     "mlp.c_fc.weight", "mlp.c_proj.weight"):
            W[p + name] = to_pairs_mat(sd[p + name])
        for name in ("attn.attention.out_proj.bias", "mlp.c_fc.bias",
                     "mlp.c_proj.bias", "ln_1.weight", "ln_1.bias",
                     "ln_2.weight", "ln_2.bias"):
            W[p + name] = to_fracs(sd[p + name])
    for name in ("ln_f.weight", "ln_f.bias"):
        W["transformer." + name] = to_fracs(sd["transformer." + name])
    return W


def interval_forward(seq_len: int, n_logits: int = 200,
                     log=print, ids: list | None = None,
                     sd: dict | None = None, W: dict | None = None) -> dict:
    """n_logits = -1 means the FULL vocab (complete certified argmax).

    ids defaults to PROMPT_IDS_16[:seq_len]; sd/W may be passed pre-loaded /
    pre-converted for prompt sweeps.
    """
    if sd is None:
        sd = load_sd()
    if ids is None:
        ids = PROMPT_IDS_16[:seq_len]
    seq_len = len(ids)
    t0 = time.time()
    if W is None:
        W = prepare_weights(sd)
    log(f"[{time.time()-t0:6.1f}s] weights converted")

    # --- embeddings (exact rational sum of tok + pos) -----------------------
    wte, wpe = sd["transformer.wte.weight"], sd["transformer.wpe.weight"]
    res: List[List[Ival]] = []
    for t, tok_id in enumerate(ids):
        row = [Ival.point(Fraction(a) + Fraction(b))
               for a, b in zip(wte[tok_id].tolist(), wpe[t].tolist())]
        res.append(row)

    stats = {"sublayer": []}   # list of (label, max_w, med_w)
    # guard-activation audit (requested during adversarial review): per-layer shifted-softmax argument and
    # tanh/GELU inner endpoint ranges + fast-path hit counts, so the P-dependent
    # exp/tanh fast paths are auditable from the artifact JSON alone.
    stats["guard_audit"] = []
    scale = exact._SCALE
    stats["guard_thresholds"] = E.guard_audit_block()

    def _snap(layer: int) -> None:
        tk = E.get_tracking()

        def real(v):
            return float(Fraction(v, scale)) if v is not None else None
        stats["guard_audit"].append({
            "layer": layer,
            "softmax_shift_min_x": real(tk["softmax_shift_min"]),
            "softmax_shift_max_x": real(tk["softmax_shift_max"]),
            "gelu_inner_min_x": real(tk["gelu_inner_min"]),
            "gelu_inner_max_x": real(tk["gelu_inner_max"]),
            "exp_guard_hits": tk["exp_guard_hits"],
            "tanh_guard_hits": tk["tanh_guard_hits"],
        })

    def record(label: str, r):
        mx, md = wstats(r)
        stats["sublayer"].append((label, mx, md))
        log(f"[{time.time()-t0:6.1f}s] {label:<26} max_w {mx:.3e}  med_w {md:.3e}")

    record("embed", res)

    # --- blocks --------------------------------------------------------------
    for L in range(N_LAYERS):
        p = f"transformer.h.{L}."
        E.reset_tracking()   # per-layer guard-activation audit
        ln1 = [E.layernorm_ival(res[t], W[p + "ln_1.weight"],
                                W[p + "ln_1.bias"], LN_EPS)
               for t in range(seq_len)]
        record(f"L{L}.ln_1", ln1)

        q = [dot_rows(ln1[t], W[p + "attn.attention.q_proj.weight"], None)
             for t in range(seq_len)]
        k = [dot_rows(ln1[t], W[p + "attn.attention.k_proj.weight"], None)
             for t in range(seq_len)]
        v = [dot_rows(ln1[t], W[p + "attn.attention.v_proj.weight"], None)
             for t in range(seq_len)]

        attn_out = [[None] * D_MODEL for _ in range(seq_len)]
        for h in range(N_HEADS):
            s0 = h * D_HEAD
            for t in range(seq_len):
                scores = []
                for j in range(t + 1):
                    acc = q[t][s0] * k[j][s0]
                    for c in range(1, D_HEAD):
                        acc = acc + q[t][s0 + c] * k[j][s0 + c]
                    scores.append(acc)          # NO 1/sqrt(d_head) (GPT-Neo)
                probs = E.softmax_ival(scores)
                for c in range(D_HEAD):
                    acc = probs[0] * v[0][s0 + c]
                    for j in range(1, t + 1):
                        acc = acc + probs[j] * v[j][s0 + c]
                    attn_out[t][s0 + c] = acc
        record(f"L{L}.attn(pre-proj)", attn_out)

        proj = [dot_rows(attn_out[t], W[p + "attn.attention.out_proj.weight"],
                         W[p + "attn.attention.out_proj.bias"])
                for t in range(seq_len)]
        res = [[a + b for a, b in zip(res[t], proj[t])] for t in range(seq_len)]
        record(f"L{L}.resid+attn", res)

        ln2 = [E.layernorm_ival(res[t], W[p + "ln_2.weight"],
                                W[p + "ln_2.bias"], LN_EPS)
               for t in range(seq_len)]
        record(f"L{L}.ln_2", ln2)
        hidden = [dot_rows(ln2[t], W[p + "mlp.c_fc.weight"],
                           W[p + "mlp.c_fc.bias"]) for t in range(seq_len)]
        hidden = [[E.gelu_new_ival(u) for u in hidden[t]] for t in range(seq_len)]
        record(f"L{L}.gelu", hidden)
        mlp = [dot_rows(hidden[t], W[p + "mlp.c_proj.weight"],
                        W[p + "mlp.c_proj.bias"]) for t in range(seq_len)]
        res = [[a + b for a, b in zip(res[t], mlp[t])] for t in range(seq_len)]
        record(f"L{L}.resid+mlp  [LAYER {L}]", res)
        _snap(L)

    # --- final LN + selected logits ------------------------------------------
    fin = E.layernorm_ival(res[-1], W["transformer.ln_f.weight"],
                           W["transformer.ln_f.bias"], LN_EPS)
    record("ln_f(last pos)", [fin])

    # logits: float top-10 ids + an even spread across the vocab
    logits_f = forward(sd, ids, torch.float64)[-1]
    top10 = torch.topk(logits_f, 10).indices.tolist()
    if n_logits == -1:
        chosen = list(range(50257))
    else:
        spread = list(range(0, 50257, 50257 // (n_logits - len(top10)) + 1))
        chosen = sorted(set(top10 + spread))[:n_logits]
    Wun = to_pairs_mat(wte[chosen])
    logit_ivs = dot_rows(fin, Wun, None)
    log(f"[{time.time()-t0:6.1f}s] {len(chosen)} logits computed")

    lw = [float(Fraction(iv.hi_i - iv.lo_i, exact._SCALE)) for iv in logit_ivs]
    gap = float(torch.topk(logits_f, 2).values.diff().abs())
    id2iv = dict(zip(chosen, logit_ivs))
    stats["logits"] = {
        "chosen": chosen if len(chosen) <= 220 else f"FULL VOCAB ({len(chosen)})",
        "widths_max": max(lw),
        "widths_med": statistics.median(lw), "gap_top1_top2": gap,
        "ratio_maxw_over_gap": max(lw) / gap,
        "top10": [(i, float(logits_f[i]),
                   float(id2iv[i].lo), float(id2iv[i].hi)) for i in top10],
    }
    # certified argmax check: does top1's lower bound beat every other upper?
    t1 = top10[0]
    sep = all(id2iv[t1].lo_i > iv.hi_i
              for i, iv in id2iv.items() if i != t1)
    stats["logits"]["argmax_certified_among_chosen"] = sep
    stats["logits"]["top1_id"] = t1
    # exact-Fraction certified lower bound on the top1-vs-rest gap (additive:
    # the same lo_i/hi_i comparison `sep` already performs, just also keeping
    # the min margin instead of only the all() boolean — no interval
    # arithmetic changes, PyTorch not on this path).
    stats["logits"]["certified_gap_lower_bound"] = min(
        (Fraction(id2iv[t1].lo_i - iv.hi_i, exact._SCALE)
         for i, iv in id2iv.items() if i != t1),
        default=None,
    )
    stats["runtime_s"] = time.time() - t0
    return stats


if __name__ == "__main__":
    seq = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    prec = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    nlog = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    exact.set_precision(prec)
    E._GELU_C = None   # dead backward-compat shim (Task 0.2: precision-keyed
                       # cache in ival_ext.py; harmless no-op, belt-and-braces)
    print(f"=== interval forward: seq_len={seq}, precision={prec} bits, "
          f"n_logits={nlog} ===")
    st = interval_forward(seq, n_logits=nlog)
    print(json.dumps(st["logits"], indent=1))
    print(f"runtime {st['runtime_s']:.1f}s")
    with open(f"widths_seq{seq}_p{prec}.json", "w") as f:
        json.dump({"sublayer": st["sublayer"], "logits": st["logits"],
                   "guard_thresholds": st["guard_thresholds"],
                   "guard_audit": st["guard_audit"],
                   "runtime_s": st["runtime_s"]}, f, indent=1)
