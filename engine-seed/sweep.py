#!/usr/bin/env python3
"""E1 prompt sweep: full-vocab certified argmax on TinyStories-1M at P=192.

28 varied seq-8 prompts (story openers, mid-sentence fragments, degenerate
single-token repeats, rare/OOD token sequences) + the seq-16 baseline prompt
rerun with the FULL vocabulary (closing the sampled-logits gap from the
original run). Per prompt: certified? (top-1 lower bound beats every other
logit's upper bound over all 50,257 logits), max logit width, float
top1-top2 gap, headroom bits = log2(gap / max_width). A non-certified
genuine near-tie is recorded as an ABSTENTION, not a failure.

Output: sweep_results.json + quantile table on stdout.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time

import exact
import ival_ext as E
from float_fwd import PROMPT_IDS_16, load_sd, _HF_CACHE
from interval_fwd import interval_forward, prepare_weights

PRECISION = 192

_VOCAB_CANDIDATES = glob.glob(os.path.join(
    _HF_CACHE, "models--roneneldan--TinyStories-1M", "snapshots", "*", "vocab.json"))
if not _VOCAB_CANDIDATES:
    raise FileNotFoundError(
        f"TinyStories-1M vocab.json not found under {_HF_CACHE!r}. "
        "Fetch it first — see README.md 'Reproduce'.")
VOCAB_PATH = _VOCAB_CANDIDATES[0]
_V = json.load(open(VOCAB_PATH))
_INV = {v: k for k, v in _V.items()}


def tok(s: str) -> int:
    """Exact vocab lookup (Ġ = leading space in GPT-2 byte-level BPE)."""
    if s not in _V:
        raise KeyError(f"token not in vocab: {s!r}")
    return _V[s]


def w(*words: str) -> list[int]:
    """ids for space-prefixed words; a leading '^' means no space prefix."""
    out = []
    for x in words:
        out.append(tok(x[1:]) if x.startswith("^") else tok("Ġ" + x))
    return out


def decode(ids: list[int]) -> str:
    return "".join(_INV[i] for i in ids).replace("Ġ", " ")


# --------------------------------------------------------------------------- #
# the 28 prompts (all seq 8, ids listed explicitly in the artifact JSON)
# --------------------------------------------------------------------------- #
PROMPTS: list[tuple[str, list[int]]] = [
    # -- natural story openers (TinyStories register) --
    ("opener/baseline",  [7454, 2402, 257, 640, 612, 373, 257, 1310]),
    ("opener/one-day",   w("^One", "day", "a", "small", "dog", "went", "to", "the")),
    ("opener/girl-said", w("^The", "little", "girl", "smiled", "and", "said", "I", "want")),
    ("opener/tom-lily",  w("^Tom", "and", "Lily", "went", "to", "the", "park", "to")),
    ("opener/red-ball",  w("^There", "was", "a", "big", "red", "ball", "in", "the")),
    ("opener/mom-said",  w("^Mom", "said", "it", "is", "time", "to", "go", "home")),
    ("opener/sun",       w("^The", "sun", "was", "shining", "and", "the", "birds", "were")),
    ("opener/anna",      w("^Anna", "wanted", "to", "play", "with", "her", "new", "toy")),
    # -- mid-sentence fragments --
    ("frag/and-then",    w("and", "then", "he", "saw", "a", "very", "big", "red")),
    ("frag/because",     w("because", "she", "was", "so", "happy", "to", "see", "her")),
    ("frag/under-tree",  w("under", "the", "tree", "next", "to", "the", "old", "box")),
    ("frag/said-cat",    w("said", "the", "cat", "and", "jumped", "over", "the", "ball")),
    ("frag/but-boy",     w("but", "the", "boy", "did", "not", "want", "to", "stop")),
    ("frag/with-friends", w("with", "his", "friends", "in", "the", "big", "blue", "car")),
    # -- degenerate: one token repeated 8x --
    ("degen/the-x8",     [262] * 8),
    ("degen/comma-x8",   [11] * 8),
    ("degen/eot-x8",     [50256] * 8),
    ("degen/girl-x8",    [2576] * 8),
    ("degen/bang-x8",    [0] * 8),
    # -- rare / OOD token sequences (ids chosen directly, decoded for record) --
    ("rare/mix-1",       [39561, 44832, 27599, 47825, 31235, 48991, 36734, 41984]),
    ("rare/mix-2",       [49731, 40219, 43217, 38442, 33901, 28371, 45677, 46351]),
    ("rare/alt-common",  [262, 39561, 262, 44832, 262, 27599, 262, 47825]),
    ("rare/high-ids",    [50009, 50113, 50221, 50256, 49913, 50052, 50158, 50242]),
    # -- other stress shapes --
    ("misc/digits",      w("1", "2", "3", "4", "5", "6", "7", "8")),
    ("misc/caps",        w("^THE", "END") * 4),
    ("misc/they-found",  w("they", "found", "a", "toy", "car", "in", "the", "water")),
    ("misc/please-help", w("please", "help", "me", "look", "at", "the", "sad", "cat")),
    ("misc/punct-heavy", [tok("Ġstop"), tok("!"), tok("Ġsaid"), tok("Ġmom"),
                          tok("!"), tok("Ġsaid"), tok("Ġdad"), tok("!")]),
]


def quantile(sorted_xs: list[float], q: float) -> float:
    """Nearest-rank-with-interpolation quantile on a pre-sorted list."""
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    i = int(pos)
    frac = pos - i
    if i + 1 < len(sorted_xs):
        return sorted_xs[i] * (1 - frac) + sorted_xs[i + 1] * frac
    return sorted_xs[i]


def main() -> None:
    exact.set_precision(PRECISION)
    E._GELU_C = None
    print(f"=== E1 prompt sweep: {len(PROMPTS)} prompts, seq 8, "
          f"P={PRECISION}, FULL vocab ===")
    sd = load_sd()
    W = prepare_weights(sd)
    silent = lambda *a, **k: None

    results = []
    t_all = time.time()
    for name, ids in PROMPTS:
        assert len(ids) == 8, name
        t0 = time.time()
        st = interval_forward(8, n_logits=-1, log=silent, ids=ids, sd=sd, W=W)
        lg = st["logits"]
        gap = lg["gap_top1_top2"]
        wmax = lg["widths_max"]
        certified = lg["argmax_certified_among_chosen"]
        headroom = math.log2(gap / wmax) if gap > 0 and wmax > 0 else float("-inf")
        top1_id = lg["top10"][0][0]
        guard_hits = sum(g["exp_guard_hits"] + g["tanh_guard_hits"]
                         for g in st["guard_audit"])
        rec = {
            "name": name, "ids": ids, "text": decode(ids),
            "certified": bool(certified),
            "status": "CERTIFIED" if certified else "ABSTAIN",
            "top1_id": top1_id, "top1_tok": _INV[top1_id],
            "gap_top1_top2": gap, "max_logit_width": wmax,
            "headroom_bits": headroom, "guard_hits": guard_hits,
            "runtime_s": round(time.time() - t0, 1),
        }
        results.append(rec)
        print(f"{name:<18} {rec['status']:<9} gap {gap:11.4e}  "
              f"maxw {wmax:.3e}  headroom {headroom:7.2f} bits  "
              f"top1 {rec['top1_tok']!r:<14} ({rec['runtime_s']}s)")

    # seq-16 baseline prompt, FULL vocab (closes the sampled-logits gap)
    print("--- seq-16 baseline prompt, FULL vocab ---")
    t0 = time.time()
    st = interval_forward(16, n_logits=-1, log=silent, ids=PROMPT_IDS_16,
                          sd=sd, W=W)
    lg = st["logits"]
    gap, wmax = lg["gap_top1_top2"], lg["widths_max"]
    certified = lg["argmax_certified_among_chosen"]
    headroom = math.log2(gap / wmax)
    seq16 = {
        "name": "seq16/baseline-fullvocab", "ids": PROMPT_IDS_16,
        "text": decode(PROMPT_IDS_16), "certified": bool(certified),
        "status": "CERTIFIED" if certified else "ABSTAIN",
        "top1_id": lg["top10"][0][0], "top1_tok": _INV[lg["top10"][0][0]],
        "gap_top1_top2": gap, "max_logit_width": wmax,
        "headroom_bits": headroom,
        "guard_hits": sum(g["exp_guard_hits"] + g["tanh_guard_hits"]
                          for g in st["guard_audit"]),
        "runtime_s": round(time.time() - t0, 1),
    }
    print(f"{seq16['name']:<26} {seq16['status']:<9} gap {gap:.4e}  "
          f"maxw {wmax:.3e}  headroom {headroom:.2f} bits")

    # ---- quantile table over the 28 seq-8 prompts ----
    hs = sorted(r["headroom_bits"] for r in results)
    n_cert = sum(r["certified"] for r in results)
    table = {
        "n_prompts": len(results), "n_certified": n_cert,
        "n_abstained": len(results) - n_cert,
        "headroom_bits_min": hs[0],
        "headroom_bits_p10": quantile(hs, 0.10),
        "headroom_bits_median": quantile(hs, 0.50),
        "headroom_bits_p90": quantile(hs, 0.90),
        "headroom_bits_max": hs[-1],
        "max_logit_width_min": min(r["max_logit_width"] for r in results),
        "max_logit_width_max": max(r["max_logit_width"] for r in results),
        "total_guard_hits": sum(r["guard_hits"] for r in results),
    }
    print("\n=== E1 quantile table (28 seq-8 prompts, headroom bits "
          "= log2(gap/max_width)) ===")
    for k, v in table.items():
        print(f"  {k:<24} {v:.3f}" if isinstance(v, float) else f"  {k:<24} {v}")
    print(f"\ntotal sweep time {time.time()-t_all:.0f}s")

    with open("sweep_results.json", "w") as f:
        json.dump({"precision_bits": PRECISION, "prompts": results,
                   "seq16_fullvocab": seq16, "quantiles": table}, f, indent=1)
    print("wrote sweep_results.json")


if __name__ == "__main__":
    main()
