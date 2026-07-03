#!/usr/bin/env python3
"""Headline certified run driver (Task 2.2).

Loops :func:`certinf.certify.certify_sample` over the FROZEN sample index of a
committed pre-registration (`certificates/prereg/headline/`), producing one
per-sample record per drawn index — duplicates certified with multiplicity (A2
with-replacement design; never deduplicated). Every record carries the freeze's
`prereg_sha256` as its `prereg_ref`, so the independent checker's A6
completeness assertion can bind the cert to exactly this pre-registration.

This is a thin driver, not new certified machinery: it reads the frozen tuple
(model, corpus binding, context length, P_max, escalation ladder) and calls the
existing `certify_sample`. It is committed as CODE *before* the run launches —
it produces no results at import/commit time. The run itself (its records) lands
in a later commit, after the precommitment commit, per the freeze runbook.

The escalation ladder `P_grid` is read from the frozen `escalation_policy`
(falling back to the design default `[128, 160, 192]` capped at the frozen
`P_max`). `run_harness=True` captures φ₂_joint (pinned float32 agreement) in the
SAME frozen run, so φ₁ and φ₂_joint come from one certification pass.

Weights: `certify_sample` runs the interval engine via `torch.load`, so
`--weights` is the pinned `pytorch_model.bin` (NOT the torch-free hex export,
which is the independent checker's input). The record's `checkpoint_sha256` is
that file's sha256; the independent checker later cross-checks it against the
hex export's self-declared checkpoint sha.
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

from certinf import certify

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_P_GRID = [128, 160, 192]


def _resolve_weights(model: str) -> str:
    """Resolve the pinned pytorch_model.bin path from the local HF cache."""
    cache = os.environ.get(
        "LMCERT_HF_CACHE", os.path.join(_REPO_ROOT, "data", "hf_cache"))
    snap = {"tinystories": "models--roneneldan--TinyStories-1M",
            "gpt2": "models--gpt2"}[model]
    hits = glob.glob(os.path.join(cache, snap, "snapshots", "*", "pytorch_model.bin"))
    if not hits:
        raise FileNotFoundError(
            f"{model} weights not found under {cache!r}; see docs/PROVENANCE.md")
    return hits[0]


def _load_prereg(prereg_dir: str) -> tuple[dict, list[int]]:
    with open(os.path.join(prereg_dir, "prereg.json")) as f:
        prereg = json.load(f)
    with open(os.path.join(prereg_dir, "sample-index.json")) as f:
        indices = json.load(f)["indices"]
    return prereg, indices


def _certify_one(task: tuple) -> dict:
    (model, weights_path, corpus_path, prompt_index, P_grid, P_max,
     context_length, prereg_ref) = task
    return certify.certify_sample(
        model, weights_path, corpus_path, prompt_index=prompt_index,
        P_grid=P_grid, P_max=P_max, prereg_ref=prereg_ref, run_harness=True,
        context_length=context_length, run_canary=True,
    )


def run(prereg_dir: str, corpus_path: str, out_dir: str,
        weights_path: str | None = None, jobs: int = 6) -> dict:
    """Certify every frozen index and stream records to
    `<out_dir>/<model>-headline.cert.jsonl` (+ `.meta.json`).

    Returns a summary dict (`n`, `k` = #φ₁, `k_joint` = #φ₂_joint, abstain
    taxonomy). Records are written in frozen-index order as they complete."""
    prereg, indices = _load_prereg(prereg_dir)
    model = prereg["model"]
    P_max = prereg["P_max"]
    context_length = prereg["context_length"]
    prereg_ref = prereg["prereg_sha256"]
    P_grid = list(prereg.get("escalation_policy", {}).get("P_grid", _DEFAULT_P_GRID))
    P_grid = [p for p in P_grid if p <= P_max] or _DEFAULT_P_GRID
    weights_path = weights_path or _resolve_weights(model)

    os.makedirs(out_dir, exist_ok=True)
    cert_path = os.path.join(out_dir, f"{model}-headline.cert.jsonl")
    meta_path = os.path.join(out_dir, f"{model}-headline.cert.meta.json")

    tasks = [(model, weights_path, corpus_path, idx, P_grid, P_max,
              context_length, prereg_ref) for idx in indices]

    print(f"[headline] model={model} n={len(tasks)} ctx={context_length} "
          f"P_grid={P_grid} P_max={P_max} jobs={jobs}", flush=True)
    print(f"[headline] prereg_ref={prereg_ref}", flush=True)
    print(f"[headline] weights={weights_path}", flush=True)
    print(f"[headline] out={cert_path}", flush=True)

    t0 = time.time()
    k = 0
    k_joint = 0
    taxonomy: dict[str, int] = {}
    done = 0
    with open(cert_path, "w") as out:
        if jobs > 1:
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
            if rec.get("phi2_joint") is True:
                k_joint += 1
            if rec["abstain_reason"] is not None:
                taxonomy[rec["abstain_reason"]] = \
                    taxonomy.get(rec["abstain_reason"], 0) + 1
            if done % 25 == 0 or done == len(tasks):
                el = time.time() - t0
                rate = el / done
                eta = rate * (len(tasks) - done)
                print(f"  [{done:>4}/{len(tasks)}] k(phi1)={k} "
                      f"k_joint={k_joint} elapsed={el:.0f}s eta={eta:.0f}s",
                      flush=True)
        if jobs > 1:
            pool.close()
            pool.join()

    runtime_s = time.time() - t0
    summary = {
        "model": model,
        "prereg_ref": prereg_ref,
        "n": len(tasks),
        "k_phi1": k,
        "k_phi2_joint": k_joint,
        "abstain_taxonomy": taxonomy,
        "P_grid": P_grid,
        "P_max": P_max,
        "context_length": context_length,
        "corpus_path": corpus_path,
        "weights_path": weights_path,
        "cert_path": cert_path,
        "jobs": jobs,
        "runtime_s": runtime_s,
    }
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[headline] DONE n={len(tasks)} k(phi1)={k} k_joint={k_joint} "
          f"abstains={taxonomy} runtime={runtime_s:.0f}s", flush=True)
    print(f"[headline] wrote {cert_path} + {meta_path}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python3.11 -m certinf.headline_run",
        description="Headline certified run over a frozen pre-registration (Task 2.2).")
    p.add_argument("--prereg", required=True,
                   help="pre-registration dir (contains prereg.json + sample-index.json)")
    p.add_argument("--corpus", required=True, help="committed token-id corpus JSON")
    p.add_argument("--out", required=True, help="output dir for the cert JSONL + meta")
    p.add_argument("--weights", default=None,
                   help="pinned pytorch_model.bin (default: resolve from HF cache)")
    p.add_argument("--jobs", type=int, default=6)
    args = p.parse_args(argv)
    run(args.prereg, args.corpus, args.out, weights_path=args.weights, jobs=args.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
