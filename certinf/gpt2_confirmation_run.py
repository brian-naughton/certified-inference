#!/usr/bin/env python3
"""GPT-2-small confirmation-set driver (Task 2.3).

Role (per the 2026-07-03 whole-corpus evaluation, Q3 + SHOULD #4): GPT-2's
place in this project is PRESTIGE CONFIRMATION, not a population claim. This
module certifies a small, deterministically-chosen slice of full-vocabulary
next-token argmaxes on real pretrained GPT-2-small, over the pinned
WikiText-103 confirmation corpus (spec A5) — never a headline, never
pre-registered, never wrapped in a Hoeffding bound.

Every record this driver produces carries `prereg_ref=None` (A2: exploration
and confirmation are never headline samples). There is no per-record "label"
field in the schema (`schema.validate_record` rejects unknown top-level
keys), so the "confirmation set — no population claim" designation lives
here and in the run's `.meta.json` + `docs/PROVENANCE.md`, not inside each
JSON record.

Index selection: the first `n` windows (by list order, i.e. prompt_index =
0..n-1) of the pinned corpus at the chosen context length — a deterministic,
non-cherry-picked slice, exactly like the corpus-build modules' own "first
n_windows" convention. The indices are recorded in the run's `.meta.json`.

Checker status (honest note, not swept under the rug): `certificates/check.py`
re-derives records against `certinf.exact`/`certinf.ival_ext` but its weight
preparation and residual-stream wiring are TinyStories-shaped; it does not
yet have a GPT-2 code path. These records are therefore full-vocabulary
exact-real certificates from the SAME generator instrument validated
elsewhere in this project, but are NOT independently re-derived by the
torch-free checker. Closing that gap is future work (see
docs/PROVENANCE.md's "GPT-2 confirmation set" section).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from certinf import canary, certify, corpus
from certinf.headline_run import _resolve_weights

_DEFAULT_P_GRID = [320, 384]
_DEFAULT_P_MAX = 448
_DEFAULT_CONTEXT_LENGTH = 16
_DEFAULT_N = 8

NOTE = "confirmation set — no population claim (prestige/scaling evidence only)"


def _certify_one(task: tuple) -> dict:
    (weights_path, corpus_path, prompt_index, P_grid, P_max,
     context_length) = task
    # run_canary=False: the widths canary is asserted ONCE PER RUN by run()
    # below (the grid runner's per-cell granularity, certinf/grid.py — both
    # granularities are documented in certinf/certify.py's docstring). A
    # per-sample canary would run TWO extra GPT-2 forwards (P and 2P — the
    # 2P=640-bit forward alone costs more than the certification forward)
    # for every one of the n samples, several-fold multiplying a run that is
    # already ~5 min/sample.
    return certify.certify_sample(
        "gpt2", weights_path, corpus_path, prompt_index=prompt_index,
        P_grid=P_grid, P_max=P_max, prereg_ref=None, run_harness=True,
        context_length=context_length, run_canary=False,
    )


def _assert_canary_once(corpus_path: str, context_length: int,
                        P: int) -> None:
    """Once-per-run widths canary (grid.py's per-cell granularity): assert no
    precision floor on the run's FIRST window at the ladder's first rung
    before any record from this run is trusted. Raises AssertionError on a
    tripped canary — the run must not proceed on unaudited widths."""
    doc = corpus.load(corpus_path)
    ids = doc["windows"][str(context_length)][0]
    print(f"[gpt2-confirmation] widths canary: P={P} vs 2P={2 * P} on "
          f"window 0 (once per run — grid.py granularity) ...", flush=True)
    t0 = time.time()
    canary.assert_no_precision_floor("gpt2", ids, P)
    print(f"[gpt2-confirmation] widths canary PASSED "
          f"({time.time() - t0:.0f}s)", flush=True)


def run(corpus_path: str, out_dir: str, weights_path: str | None = None,
        n: int = _DEFAULT_N, context_length: int = _DEFAULT_CONTEXT_LENGTH,
        P_grid: list[int] | None = None, P_max: int = _DEFAULT_P_MAX,
        jobs: int = 1, run_canary: bool = True) -> dict:
    """Certify the first `n` windows at `context_length` from `corpus_path`.

    Streams records to `<out_dir>/gpt2-confirmation.cert.jsonl` (+
    `.meta.json`) in index order, and additionally writes the first CERTIFIED
    record standalone as `<out_dir>/gpt2-small-fullvocab.cert.json` — the
    brief's "prestige single certificate", extracted from this same run
    rather than a separate pass. Sequential by default (`jobs=1`): each
    full-vocabulary GPT-2 forward is CPU/memory heavy (~5-10 min), so this
    driver does not oversubscribe the machine by default.

    The widths canary runs ONCE PER RUN (before any sample — see
    `_assert_canary_once`); `run_canary=False` is for unit tests only.
    """
    P_grid = list(P_grid) if P_grid else list(_DEFAULT_P_GRID)
    weights_path = weights_path or _resolve_weights("gpt2")
    indices = list(range(n))

    if run_canary:
        _assert_canary_once(corpus_path, context_length, P_grid[0])

    os.makedirs(out_dir, exist_ok=True)
    cert_path = os.path.join(out_dir, "gpt2-confirmation.cert.jsonl")
    meta_path = os.path.join(out_dir, "gpt2-confirmation.cert.meta.json")
    single_path = os.path.join(out_dir, "gpt2-small-fullvocab.cert.json")

    tasks = [(weights_path, corpus_path, idx, P_grid, P_max, context_length)
             for idx in indices]

    print(f"[gpt2-confirmation] n={len(tasks)} ctx={context_length} "
          f"P_grid={P_grid} P_max={P_max} jobs={jobs} note={NOTE!r}",
          flush=True)
    print(f"[gpt2-confirmation] weights={weights_path}", flush=True)
    print(f"[gpt2-confirmation] corpus={corpus_path}", flush=True)
    print(f"[gpt2-confirmation] out={cert_path}", flush=True)

    t0 = time.time()
    k = 0
    k_joint = 0
    taxonomy: dict[str, int] = {}
    done = 0
    first_certified_record: dict | None = None
    with open(cert_path, "w") as out:
        if jobs > 1:
            import multiprocessing as mp
            pool = mp.Pool(jobs)
            results = pool.imap(_certify_one, tasks, chunksize=1)
        else:
            results = (_certify_one(t) for t in tasks)
        for rec in results:
            out.write(json.dumps(rec) + "\n")
            out.flush()
            done += 1
            if rec["status"] == "CERTIFIED":
                k += 1
                if first_certified_record is None:
                    first_certified_record = rec
            if rec.get("phi2_joint") is True:
                k_joint += 1
            if rec["abstain_reason"] is not None:
                taxonomy[rec["abstain_reason"]] = \
                    taxonomy.get(rec["abstain_reason"], 0) + 1
            el = time.time() - t0
            rate = el / done
            eta = rate * (len(tasks) - done)
            print(f"  [{done:>2}/{len(tasks)}] idx={rec['prompt_index']} "
                  f"status={rec['status']} k={k} k_joint={k_joint} "
                  f"runtime_s={rec['runtime_s']:.1f} elapsed={el:.0f}s "
                  f"eta={eta:.0f}s", flush=True)
        if jobs > 1:
            pool.close()
            pool.join()

    runtime_s = time.time() - t0
    if first_certified_record is not None:
        with open(single_path, "w") as f:
            json.dump(first_certified_record, f, indent=1)

    summary = {
        "note": NOTE,
        "model": "gpt2",
        "prereg_ref": None,
        "n": len(tasks),
        "indices": indices,
        "k_phi1": k,
        "k_phi2_joint": k_joint,
        "abstain_taxonomy": taxonomy,
        "P_grid": P_grid,
        "P_max": P_max,
        "context_length": context_length,
        "corpus_path": corpus_path,
        "weights_path": weights_path,
        "cert_path": cert_path,
        "single_cert_path": single_path if first_certified_record else None,
        "jobs": jobs,
        "runtime_s": runtime_s,
        "canary": ("asserted once per run (grid.py per-cell granularity) on "
                   f"window 0 at P={P_grid[0]} vs 2P={2 * P_grid[0]}"
                   if run_canary else "SKIPPED (tests only)"),
        "checker_status": (
            "NOT independently re-derived by certificates/check.py — its "
            "weight preparation and wiring are TinyStories-shaped; a "
            "GPT-2 checker path is future work (see docs/PROVENANCE.md)."
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[gpt2-confirmation] DONE n={len(tasks)} k(phi1)={k} "
          f"k_joint={k_joint} abstains={taxonomy} runtime={runtime_s:.0f}s",
          flush=True)
    print(f"[gpt2-confirmation] wrote {cert_path} + {meta_path}"
          + (f" + {single_path}" if first_certified_record else ""),
          flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python3.11 -m certinf.gpt2_confirmation_run",
        description="GPT-2-small confirmation-set run (Task 2.3, prestige "
                    "confirmation — not a population claim).")
    p.add_argument("--corpus", required=True,
                   help="committed token-id corpus JSON (WikiText-103)")
    p.add_argument("--out", required=True,
                   help="output dir for the cert JSONL + meta + single cert")
    p.add_argument("--weights", default=None,
                   help="pinned pytorch_model.bin (default: resolve from HF cache)")
    p.add_argument("--n", type=int, default=_DEFAULT_N)
    p.add_argument("--context-length", type=int, default=_DEFAULT_CONTEXT_LENGTH)
    p.add_argument("--P-grid", default=None,
                   help="comma-separated escalation ladder, e.g. 320,384")
    p.add_argument("--P-max", type=int, default=_DEFAULT_P_MAX)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--skip-canary", action="store_true",
                   help="skip the once-per-run widths canary (dev/tests only "
                        "— never for a committed artifact)")
    args = p.parse_args(argv)
    P_grid = ([int(x) for x in args.P_grid.split(",")]
              if args.P_grid else None)
    run(args.corpus, args.out, weights_path=args.weights, n=args.n,
        context_length=args.context_length, P_grid=P_grid, P_max=args.P_max,
        jobs=args.jobs, run_canary=not args.skip_canary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
