#!/usr/bin/env python3
"""Independent, torch-free re-derivation of per-sample certificates.

TRUST BOUNDARY (two instruments): this checker re-derives records from the hex
weight export and trusts that export's self-declared checkpoint_sha256; the
binding of hex bytes to the actual checkpoint is certinf/loader.py's one-time,
sha-verified job. check.py alone proves (hex weights => records) consistency,
not weight authenticity — the two instruments together close the boundary.

This module CLOSES THE TORCH TRUST BOUNDARY for the certified path. It reads
the model weights *only* from the sha-pinned hex export produced by
`certinf.loader.export_weights` (`float.fromhex` — never torch, never the
`torch.load` path `certinf.certify`/`certinf.interval_fwd` use to produce a
certificate), the corpus only from its committed token-id JSON, and for every
record re-derives the claimed quantities from those two artifacts alone. It
trusts the certificate for nothing except which inputs it claims to cover.

For each record it re-runs the interval forward pass at the record's own
`precision_P` (same precision => deterministic integer arithmetic) and checks:

  (i)   the weights export's `meta.checkpoint_sha256` and the recomputed
        corpus sha256 agree with the record's `checkpoint_sha256` /
        `corpus_sha256`, and the corpus window at `prompt_index` reproduces
        the record's `token_ids` bit-for-bit;
  (ii)  the certified argmax is RE-DERIVED from the interval endpoints alone
        (the unique token whose lower bound strictly beats every other token's
        upper bound over the FULL vocabulary) — the record's `argmax_token` is
        trusted for nothing — and both it and the exact-Fraction `margin_lo`
        are bit-identical to the record;
  (iii) an ABSTAIN record reproduces: no token separates at `precision_P`
        (an abstain claims nothing, so the check is that nothing CAN be
        certified there — an abstain that hides a certifiable argmax fails).

Completeness (A6). A cert can pass every per-record check yet still be a
forgery *by omission* — silently dropping frozen sample indices. For a HEADLINE
claim (records carry a non-null `prereg_ref`) the checker therefore also
asserts, unless `--sample`, that the multiset of `prompt_index` values is
exactly the frozen `sample-index.json` for the cert's pre-registration
(duplicates counted with multiplicity — the A2 with-replacement design), after
verifying the freeze witness (`certinf.prereg.verify`) and that `prereg_ref`
equals the freeze's `prereg_sha256` and the frozen (model, corpus_sha256,
context_length, P_max) tuple matches every record. A CALIBRATION cert (every
`prereg_ref` null — never a population claim, A2) is re-derived record by
record but makes no coverage claim. Any cert run with `--sample` prints an
honest "sampled" VERIFIED line and asserts no completeness.

Population claim (Task 2.5). Immediately after a full (non-sampled) headline
VERIFIED pass, the checker itself prints the Hoeffding population-claim line
for each pre-registered property (phi1, phi2_joint) via
`certinf.bound_report`: it counts successes over the FULL, just-re-derived
record set and applies each property's frozen `delta_split` share — so the
checker's own re-derivation, not the certificate generator, is the last word
on the number a reader would quote. A `--sample` run never prints this line.

Torch-free note (adaptation, 2026-07-03): the interval ENGINE
(`certinf.interval_fwd`) imports torch at module scope and prepares weights
from a `torch.load`ed state dict, so it cannot be imported here. The audited
interval ARITHMETIC it relies on — `certinf.exact` (Ival, outward rounding)
and `certinf.ival_ext` (layernorm/softmax/gelu_new) — is pure stdlib and is
imported and reused unchanged. Only the weight-preparation (`.tolist()` ->
`float.fromhex`), the residual-stream wiring, and the full-vocab argmax
re-derivation are re-expressed here against the hex blob; the per-term
integer-shift accumulation (`_dot_rows`) is a verbatim port of the engine's,
kept identical so the re-derivation is bit-for-bit comparable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from collections import Counter
from fractions import Fraction

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from certinf import bound_report                                # noqa: E402  (Task 2.5)
from certinf import corpus as corpus_mod                       # noqa: E402
from certinf import exact                                      # noqa: E402  (stdlib)
from certinf import ival_ext as E                              # noqa: E402  (stdlib)
from certinf import prereg as prereg_mod                       # noqa: E402  (public iface only)
from certinf.exact import Ival                                 # noqa: E402

LN_EPS = Fraction(1e-5)          # exact dyadic of the double, matching interval_fwd
_G: dict = {}                    # per-process cache: prepared weights + cfg


# --------------------------------------------------------------------------- #
# torch-free weight preparation (from the hex export, stdlib only)
# --------------------------------------------------------------------------- #
def _pairs_from_hex(rows) -> list[list[tuple[int, int]]]:
    """Hex float rows -> exact dyadic (numerator, shift) pairs: w = n / 2**k.

    `float.fromhex` widens the stored float32 to its exact float64 value, so
    `Fraction(v)` is the exact dyadic rational of the checkpoint weight — the
    identical value the torch engine's `Fraction(tensor_value)` produces."""
    out = []
    for row in rows:
        r = []
        for x in row:
            f = Fraction(float.fromhex(x))
            r.append((f.numerator, f.denominator.bit_length() - 1))
        out.append(r)
    return out


def _fracs_from_hex(xs) -> list[Fraction]:
    return [Fraction(float.fromhex(x)) for x in xs]


def _floats_from_hex(xs) -> list[float]:
    return [float.fromhex(x) for x in xs]


def _prepare(blob: dict) -> dict:
    """Prepare the precision-INDEPENDENT weight tables once (reused across
    every record a worker handles — the dyadic pairs do not depend on P)."""
    sd = blob["state_dict_hex"]
    cfg = blob["cfg"]
    W: dict = {}
    for L in range(cfg["n_layers"]):
        p = f"transformer.h.{L}."
        for name in ("attn.attention.q_proj.weight", "attn.attention.k_proj.weight",
                     "attn.attention.v_proj.weight", "attn.attention.out_proj.weight",
                     "mlp.c_fc.weight", "mlp.c_proj.weight"):
            W[p + name] = _pairs_from_hex(sd[p + name])
        for name in ("attn.attention.out_proj.bias", "mlp.c_fc.bias",
                     "mlp.c_proj.bias", "ln_1.weight", "ln_1.bias",
                     "ln_2.weight", "ln_2.bias"):
            W[p + name] = _fracs_from_hex(sd[p + name])
    for name in ("ln_f.weight", "ln_f.bias"):
        W["transformer." + name] = _fracs_from_hex(sd["transformer." + name])
    return {
        "W": W,
        "cfg": cfg,
        "wte_hex": sd["transformer.wte.weight"],
        "wpe_hex": sd["transformer.wpe.weight"],
        # tied unembedding = wte, as dyadic pairs (the full-vocab logit matrix)
        "Wun": _pairs_from_hex(sd["transformer.wte.weight"]),
    }


# --------------------------------------------------------------------------- #
# interval forward (stdlib; residual-stream wiring re-derived, arithmetic reused)
# --------------------------------------------------------------------------- #
def _dot_rows(x, W, bias):
    """y_j = sum_i x_i * W[j][i] (+ b_j), outward rounding per term.

    Verbatim port of `certinf.interval_fwd.dot_rows` (kept byte-identical so
    the enclosure endpoints are bit-for-bit comparable to the engine's)."""
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


# MIRROR of certinf.interval_fwd.interval_forward — keep in LOCKSTEP with the
# engine's wiring; any engine forward-pass edit must be mirrored here (divergence
# fails loudly as a bit-mismatch, but should be caught by a maintainer first).
def _logit_intervals(prep: dict, ids: list[int]) -> list[Ival]:
    """Full interval forward for TinyStories-1M (GPT-Neo arch), returning the
    full-vocabulary logit intervals for the final position. Mirrors
    `certinf.interval_fwd.interval_forward` (n_logits=-1) exactly, reusing the
    audited `certinf.ival_ext` primitives for LN / softmax / gelu_new."""
    W, cfg = prep["W"], prep["cfg"]
    NL, NH, DM, DH = cfg["n_layers"], cfg["n_heads"], cfg["d_model"], cfg["d_head"]
    wte, wpe = prep["wte_hex"], prep["wpe_hex"]
    seq = len(ids)

    # embeddings (exact rational sum of tok + pos)
    res = []
    for t, tok in enumerate(ids):
        a = _floats_from_hex(wte[tok])
        b = _floats_from_hex(wpe[t])
        res.append([Ival.point(Fraction(av) + Fraction(bv)) for av, bv in zip(a, b)])

    for L in range(NL):
        p = f"transformer.h.{L}."
        E.reset_tracking()
        ln1 = [E.layernorm_ival(res[t], W[p + "ln_1.weight"], W[p + "ln_1.bias"], LN_EPS)
               for t in range(seq)]
        q = [_dot_rows(ln1[t], W[p + "attn.attention.q_proj.weight"], None) for t in range(seq)]
        k = [_dot_rows(ln1[t], W[p + "attn.attention.k_proj.weight"], None) for t in range(seq)]
        v = [_dot_rows(ln1[t], W[p + "attn.attention.v_proj.weight"], None) for t in range(seq)]

        attn_out = [[None] * DM for _ in range(seq)]
        for h in range(NH):
            s0 = h * DH
            for t in range(seq):
                scores = []
                for j in range(t + 1):
                    acc = q[t][s0] * k[j][s0]
                    for c in range(1, DH):
                        acc = acc + q[t][s0 + c] * k[j][s0 + c]
                    scores.append(acc)          # NO 1/sqrt(d_head) (GPT-Neo)
                probs = E.softmax_ival(scores)
                for c in range(DH):
                    acc = probs[0] * v[0][s0 + c]
                    for j in range(1, t + 1):
                        acc = acc + probs[j] * v[j][s0 + c]
                    attn_out[t][s0 + c] = acc

        proj = [_dot_rows(attn_out[t], W[p + "attn.attention.out_proj.weight"],
                          W[p + "attn.attention.out_proj.bias"]) for t in range(seq)]
        res = [[a + b for a, b in zip(res[t], proj[t])] for t in range(seq)]

        ln2 = [E.layernorm_ival(res[t], W[p + "ln_2.weight"], W[p + "ln_2.bias"], LN_EPS)
               for t in range(seq)]
        hidden = [_dot_rows(ln2[t], W[p + "mlp.c_fc.weight"], W[p + "mlp.c_fc.bias"])
                  for t in range(seq)]
        hidden = [[E.gelu_new_ival(u) for u in hidden[t]] for t in range(seq)]
        mlp = [_dot_rows(hidden[t], W[p + "mlp.c_proj.weight"], W[p + "mlp.c_proj.bias"])
               for t in range(seq)]
        res = [[a + b for a, b in zip(res[t], mlp[t])] for t in range(seq)]

    fin = E.layernorm_ival(res[-1], W["transformer.ln_f.weight"],
                           W["transformer.ln_f.bias"], LN_EPS)
    return _dot_rows(fin, prep["Wun"], None)


def _certified_argmax(logit_ivs: list[Ival]):
    """Re-derive the certified argmax from the interval endpoints alone.

    Returns (token_id, margin_lo: Fraction) if a unique token's lower bound
    strictly beats every other token's upper bound over the full vocab, else
    (None, None). The candidate is the token of maximal lower endpoint: if any
    token separates, it must be that one (its lower bound exceeds — hence its
    lower bound is strictly greatest). margin_lo = min_{j != t}(lo_t - hi_j) /
    2**P, exactly the engine's `certified_gap_lower_bound`."""
    cand = max(range(len(logit_ivs)), key=lambda i: logit_ivs[i].lo_i)
    max_other_hi = None
    for j, iv in enumerate(logit_ivs):
        if j == cand:
            continue
        if max_other_hi is None or iv.hi_i > max_other_hi:
            max_other_hi = iv.hi_i
    if max_other_hi is None or logit_ivs[cand].lo_i <= max_other_hi:
        return None, None
    return cand, Fraction(logit_ivs[cand].lo_i - max_other_hi, exact._SCALE)


# --------------------------------------------------------------------------- #
# per-record re-derivation
# --------------------------------------------------------------------------- #
def _init_worker(weights_path: str) -> None:
    with open(weights_path) as f:
        blob = json.load(f)
    _G.clear()
    _G["blob_meta"] = blob["meta"]
    _G["prep"] = _prepare(blob)


def _check_one(rec: dict) -> tuple:
    """Re-derive one record; return (line_no, prompt_index, ok, reason)."""
    line_no = rec["__line__"]
    pi = rec["prompt_index"]
    ids = rec["token_ids"]
    exact.set_precision(rec["precision_P"])
    logit_ivs = _logit_intervals(_G["prep"], ids)
    tok, margin = _certified_argmax(logit_ivs)

    if rec["status"] == "CERTIFIED":
        if tok is None:
            return (line_no, pi, False, "claims CERTIFIED but no token separates "
                    "at precision_P")
        if tok != rec["argmax_token"]:
            return (line_no, pi, False,
                    f"argmax {tok} != recorded {rec['argmax_token']}")
        rec_margin = Fraction(int(rec["margin_lo"][0]), int(rec["margin_lo"][1]))
        if margin != rec_margin:
            return (line_no, pi, False, "margin_lo not bit-identical")
        if margin <= 0:
            return (line_no, pi, False, "re-derived margin_lo is not positive")
        return (line_no, pi, True, "")
    # ABSTAIN: nothing may be certifiable at precision_P (no hidden argmax)
    if tok is not None:
        return (line_no, pi, False,
                f"claims ABSTAIN but token {tok} certifies at precision_P")
    return (line_no, pi, True, "")


# --------------------------------------------------------------------------- #
# integrity + completeness
# --------------------------------------------------------------------------- #
def _fail(msg: str) -> int:
    print(f"FAILED: {msg}")
    return 1


def _load_cert(path: str) -> list[dict]:
    recs = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["__line__"] = i
            recs.append(rec)
    return recs


def _prereg_path(arg: str) -> str:
    """Accept either a prereg.json path or the directory containing it."""
    return os.path.join(arg, "prereg.json") if os.path.isdir(arg) else arg


def _check_completeness(recs, prereg_arg, args, refs) -> int | None:
    """Headline mode: verify the freeze witness, cross-check prereg_ref and the
    frozen tuple, and (unless --sample) assert exact index coverage. Returns
    None on success or an exit code on failure."""
    if len(refs) != 1:
        return _fail(f"headline cert carries multiple distinct prereg_ref values "
                     f"({sorted(refs)[:3]}...); one freeze per cert")
    if prereg_arg is None:
        return _fail("headline cert (non-null prereg_ref) requires --prereg "
                     "(the frozen pre-registration) to cross-check and, unless "
                     "--sample, to assert A6 index completeness")
    pj = _prereg_path(prereg_arg)
    corpus_dir_for_verify = args.corpus
    if not prereg_mod.verify(pj, corpus_dir_for_verify):
        return _fail(f"pre-registration freeze witness failed "
                     f"(prereg.verify({pj!r}, {corpus_dir_for_verify!r}) is False)")
    with open(pj) as f:
        prereg = json.load(f)
    ref = next(iter(refs))
    if ref != prereg.get("prereg_sha256"):
        return _fail(f"record prereg_ref {ref[:12]}... != freeze prereg_sha256 "
                     f"{str(prereg.get('prereg_sha256'))[:12]}...")
    # frozen tuple must match every record
    for rec in recs:
        for field, pfield in (("model", "model"),
                              ("corpus_sha256", "corpus_sha256"),
                              ("context_length", "context_length"),
                              ("P_max", "P_max")):
            if rec[field] != prereg[pfield]:
                return _fail(f"record {field}={rec[field]!r} != frozen "
                             f"{pfield}={prereg[pfield]!r} (line {rec['__line__']})")
    if args.sample:
        return None                     # sampled: no coverage claim
    # A6: exact index coverage, duplicates counted with multiplicity
    idx_path = os.path.join(os.path.dirname(pj) or ".", "sample-index.json")
    with open(idx_path) as f:
        frozen = Counter(json.load(f)["indices"])
    got = Counter(r["prompt_index"] for r in recs)
    if got != frozen:
        missing = sorted((frozen - got).elements())
        extra = sorted((got - frozen).elements())
        return _fail(f"cert does not cover the pre-registered index set "
                     f"(missing {missing[:5]}, unexpected {extra[:5]}; "
                     f"{sum(got.values())} records vs {sum(frozen.values())} frozen)")
    return None


def _report(results, sampled: bool, headline: bool) -> int:
    bad = [(ln, pi, why) for ln, pi, ok, why in results if not ok]
    if bad:
        head = "; ".join(f"line {ln} (prompt {pi}): {why}" for ln, pi, why in bad[:3])
        return _fail(f"{len(bad)} record(s) did not re-derive — {head}")
    claim = "headline" if headline else "calibration (no population claim)"
    mode = f"sampled every {sampled}th" if sampled else "all records"
    print(f"VERIFIED ({len(results)} records re-derived from hex weights, "
          f"{mode}, {claim})")
    return 0


def _print_population_claims(recs: list[dict], prereg_dict: dict) -> None:
    """Print the Hoeffding population-claim line per pre-registered property
    (Task 2.5).

    Runs only after a full (non-sampled) headline VERIFIED pass, so the
    checker's own re-derivation — not the certificate generator — is the last
    word on the number a reader would quote: for each property in the
    frozen `delta_split` (phi1, phi2_joint) it counts successes over the
    FULL, A6-complete record set the checker just re-derived and reports
    `certinf.bound_report`'s Hoeffding lower bound at that property's
    pre-registered delta share.
    """
    n = prereg_dict["n"]
    for prop in ("phi1", "phi2_joint"):
        pair = prereg_dict["delta_split"].get(prop)
        if pair is None:
            continue
        delta = Fraction(int(pair[0]), int(pair[1]))
        k = sum(1 for r in recs if r.get(prop) is True)
        report = bound_report.hoeffding_lower_bound(n=n, k=k, delta=delta)
        print(report.line(property_name=prop))


def main() -> int:
    ap = argparse.ArgumentParser(description="Independent torch-free certificate checker")
    ap.add_argument("--weights", required=True, help="hex weight export (certinf.loader)")
    ap.add_argument("--corpus", required=True, help="committed token-id corpus JSON")
    ap.add_argument("--cert", required=True, help="per-sample certificate JSONL")
    ap.add_argument("--prereg", default=None,
                    help="prereg.json (or its dir) for a headline cert's A6 check")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--sample", type=int, default=None,
                    help="check only every k-th record (smoke mode; says so)")
    args = ap.parse_args()

    recs = _load_cert(args.cert)
    if not recs:
        return _fail(f"certificate {args.cert!r} has no records")

    # --- weights export integrity: its self-declared checkpoint sha must agree
    #     with every record (the re-derivation itself catches any weight tamper).
    with open(args.weights) as f:
        blob_meta = json.load(f)["meta"]
    for rec in recs:
        if rec["checkpoint_sha256"] != blob_meta.get("checkpoint_sha256"):
            return _fail(f"weights export checkpoint_sha256 "
                         f"{str(blob_meta.get('checkpoint_sha256'))[:12]}... != record "
                         f"{rec['checkpoint_sha256'][:12]}... (line {rec['__line__']})")

    # --- corpus integrity: recompute the corpus sha torch-free and bind every
    #     record's token_ids to the corpus window it claims to cover.
    corpus_doc = corpus_mod.load(args.corpus)
    csha = corpus_mod.corpus_sha256(corpus_doc["windows"])
    for rec in recs:
        if rec["corpus_sha256"] != csha:
            return _fail(f"corpus sha256 {csha[:12]}... != record "
                         f"{rec['corpus_sha256'][:12]}... (line {rec['__line__']})")
        wins = corpus_doc["windows"].get(str(rec["context_length"]))
        if wins is None or rec["prompt_index"] >= len(wins):
            return _fail(f"corpus has no window at context_length="
                         f"{rec['context_length']} index {rec['prompt_index']} "
                         f"(line {rec['__line__']})")
        if list(wins[rec["prompt_index"]]) != list(rec["token_ids"]):
            return _fail(f"record token_ids != corpus window at prompt_index "
                         f"{rec['prompt_index']} (line {rec['__line__']})")

    # --- headline vs calibration (A2): prereg_ref all-null => calibration.
    refs = {rec.get("prereg_ref") for rec in recs}
    headline = refs != {None}
    if headline and None in refs:
        return _fail("mixed prereg_ref: a cert must be all-headline (non-null) "
                     "or all-calibration (null) — never mixed (A2)")
    if headline:
        rc = _check_completeness(recs, args.prereg, args, refs)
        if rc is not None:
            return rc
    elif args.prereg is not None:
        return _fail("--prereg given but cert is calibration (all prereg_ref "
                     "null); calibration makes no population claim (A2)")

    # --- re-derive (sampled or full) ---------------------------------------- #
    todo = recs[::args.sample] if args.sample else recs
    if args.jobs <= 1:
        _init_worker(args.weights)
        results = [_check_one(r) for r in todo]
    else:
        with mp.Pool(args.jobs, initializer=_init_worker,
                     initargs=(args.weights,)) as pool:
            results = pool.map(_check_one, todo, chunksize=1)

    rc = _report(results, args.sample, headline)

    # Task 2.5: announce the verified population bound. Guarded to headline
    # mode with full (non-sampled) A6 completeness — a --sample run only
    # re-derives a subset of records, so it must never print a population
    # number.
    if rc == 0 and headline and not args.sample:
        with open(_prereg_path(args.prereg)) as f:
            prereg_dict = json.load(f)
        _print_population_claims(recs, prereg_dict)

    return rc


if __name__ == "__main__":
    sys.exit(main())
