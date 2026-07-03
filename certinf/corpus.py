"""Pinned, published prompt corpora as committed token-id lists.

Downstream code trusts only the committed integers — never the tokenizer at
certify time (A4/A2 discipline: "no tokenizer trust downstream"). A corpus
file is `{"meta": {...}, "corpus_sha256": ..., "windows": {ctx_len: [[ids...],
...]}}`; `corpus_sha256` is a canonical-JSON sha256 over the token-id lists
alone, so it is stable regardless of how the meta block is formatted.

Scope note (Task 1 build): this module implements the general build/load/sha
interface. `build_tinystories` fetches the real, licensed TinyStories
validation split (CDLA-Sharing-1.0 — see docs/CORPUS-LICENSING.md) directly as
a parquet file (NOT via the `datasets` library's full builder pipeline, which
regenerates the entire multi-GB train+validation Arrow cache and is
disk-prohibitive in a constrained environment) and tokenises with the
checkpoint's own tokenizer. The GPT-2/WikiText-103 confirmation corpus
(`build_wikitext103`) is Phase 0's Task 0.6 proper and is NOT implemented
here — GPT-2 is confirmation-only (spec A5) and no Task 1 test requires it.
"""
from __future__ import annotations

import hashlib
import json
import os

_LICENSE_TINYSTORIES = "cdla-sharing-1.0"
_TINYSTORIES_VALIDATION_PARQUET = (
    "data/validation-00000-of-00001-869c898b519ad725.parquet"
)


def canonical_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def corpus_sha256(windows: dict) -> str:
    """Canonical-JSON sha256 over the token-id windows alone (the pinned
    identity — no tokenizer trust downstream). `windows` keys may be int or
    str context lengths; normalised to str for a stable digest."""
    normalised = {str(k): [list(w) for w in v] for k, v in windows.items()}
    return hashlib.sha256(canonical_json(normalised).encode()).hexdigest()


def _write(out_path: str, meta: dict, windows: dict) -> dict:
    normalised = {str(k): [list(w) for w in v] for k, v in windows.items()}
    doc = {
        "meta": meta,
        "corpus_sha256": corpus_sha256(normalised),
        "windows": normalised,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=1)
    return doc


def load(path: str) -> dict:
    """Torch-free load: `{"meta", "corpus_sha256", "windows"}`."""
    with open(path) as f:
        return json.load(f)


def build_tinystories(
    context_lengths=(8, 16, 32),
    out_path: str | None = None,
    n_windows: int = 200,
    hf_cache: str = "data/hf_cache",
) -> dict:
    """Build a TinyStories-validation-slice corpus of non-overlapping windows.

    Tokenises the roneneldan/TinyStories-1M checkpoint's own tokenizer
    (GPT-2 byte-level BPE) over the validation split, concatenates the token
    stream per story (each story separately, so no window straddles a story
    boundary), and slides non-overlapping windows of each context length,
    taking the first `n_windows` deterministically. The committed artifact is
    integer token ids only, never raw text.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    parquet_path = hf_hub_download(
        repo_id="roneneldan/TinyStories", repo_type="dataset",
        filename=_TINYSTORIES_VALIDATION_PARQUET, cache_dir=hf_cache,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "roneneldan/TinyStories-1M", cache_dir=hf_cache)
    table = pq.read_table(parquet_path)
    texts = table.column("text").to_pylist()

    windows = {}
    for ctx_len in context_lengths:
        wins = []
        for text in texts:
            if len(wins) >= n_windows:
                break
            ids = tokenizer.encode(text)
            for start in range(0, len(ids) - ctx_len + 1, ctx_len):
                wins.append(ids[start:start + ctx_len])
                if len(wins) >= n_windows:
                    break
        windows[ctx_len] = wins

    meta = {
        "source": "roneneldan/TinyStories validation split",
        "hf_repo": "roneneldan/TinyStories",
        "split": "validation",
        "license": _LICENSE_TINYSTORIES,
        "tokenizer": "roneneldan/TinyStories-1M (GPT-2 byte-level BPE)",
        "build_command": (
            "certinf.corpus.build_tinystories(context_lengths="
            f"{tuple(context_lengths)}, n_windows={n_windows})"
        ),
        "n_windows": {str(k): len(v) for k, v in windows.items()},
    }
    doc = {"meta": meta, "windows": windows}
    doc["corpus_sha256"] = corpus_sha256(windows)
    if out_path:
        _write(out_path, meta, windows)
    return doc


def build_fixed(windows_by_ctx: dict, out_path: str | None = None,
                meta: dict | None = None) -> dict:
    """Write a corpus file directly from given windows (test fixtures / small
    dev corpora — not a licensed production artifact; `meta` should say so)."""
    meta = dict(meta or {})
    meta.setdefault("source", "hand-assembled fixture (not a licensed "
                             "production corpus)")
    meta.setdefault("n_windows",
                    {str(k): len(v) for k, v in windows_by_ctx.items()})
    doc = {"meta": meta, "windows": {str(k): [list(w) for w in v]
                                     for k, v in windows_by_ctx.items()}}
    doc["corpus_sha256"] = corpus_sha256(doc["windows"])
    if out_path:
        _write(out_path, meta, windows_by_ctx)
    return doc
