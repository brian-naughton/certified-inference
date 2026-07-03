"""Pinned, published prompt corpora as committed token-id lists.

Downstream code trusts only the committed integers — never the tokenizer at
certify time (A4/A2 discipline: "no tokenizer trust downstream"). A corpus
file is `{"meta": {...}, "corpus_sha256": ..., "windows": {ctx_len: [[ids...],
...]}}`; `corpus_sha256` is a canonical-JSON sha256 over the token-id lists
alone, so it is stable regardless of how the meta block is formatted.

Scope note: this module implements the general build/load/sha interface.
`build_tinystories` (Task 1 build) fetches the real, licensed TinyStories
validation split (CDLA-Sharing-1.0 — see docs/PROVENANCE.md) directly as
a parquet file (NOT via the `datasets` library's full builder pipeline, which
regenerates the entire multi-GB train+validation Arrow cache and is
disk-prohibitive in a constrained environment) and tokenises with the
checkpoint's own tokenizer. `build_wikitext103` (Task 0.6, GPT-2's
confirmation-only corpus per spec A5/A4) fetches the WikiText-103
(`wikitext-103-raw-v1`) test split the same way and tokenises with GPT-2's
own tokenizer — see docs/PROVENANCE.md for the verified licence text.
"""
from __future__ import annotations

import hashlib
import json
import os

_LICENSE_TINYSTORIES = "cdla-sharing-1.0"
_TINYSTORIES_VALIDATION_PARQUET = (
    "data/validation-00000-of-00001-869c898b519ad725.parquet"
)

# The wikitext HF dataset card is internally inconsistent: its YAML metadata
# tags this config `cc-by-sa-3.0` + `gfdl`, but the card's own prose
# "Licensing Information" section states CC BY-SA 4.0. Both are recorded
# verbatim (see docs/PROVENANCE.md) — either reading is an attribution +
# share-alike licence that permits redistribution of derivatives, so this is
# not an A4 HARD-gate stop condition.
_LICENSE_WIKITEXT = (
    "yaml-tags: cc-by-sa-3.0, gfdl; card-prose: CC BY-SA 4.0 "
    "(https://creativecommons.org/licenses/by-sa/4.0/) — see docs/PROVENANCE.md"
)
_WIKITEXT103_TEST_PARQUET = "wikitext-103-raw-v1/test-00000-of-00001.parquet"


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


def build_wikitext103(
    context_lengths=(8, 16, 32),
    out_path: str | None = None,
    n_windows: int = 200,
    hf_cache: str = "data/hf_cache",
) -> dict:
    """Build the WikiText-103 test-split corpus (GPT-2 confirmation corpus,
    spec amendment A4's redistributable fallback — NOT OpenWebText, whose
    licence is ambiguous).

    Tokenises with GPT-2's own tokenizer (`AutoTokenizer.from_pretrained
    ("gpt2")`, GPT-2 byte-level BPE; the `vocab.json` sha256 is recorded as
    `tokenizer_sha`), and slides non-overlapping windows of each context
    length, taking the first `n_windows` deterministically, mirroring
    `build_tinystories`'s per-row-boundary discipline (no window straddles a
    `text` row; `wikitext-103-raw-v1` rows are single lines — paragraphs,
    blank separators, or `= = Heading = = ` markers — not whole articles).
    Fetched as the single test-split parquet file directly via
    `hf_hub_download` (not the `datasets` library's full builder pipeline,
    which would also pull the ~300 MB train split) — same disk-budget
    rationale as `build_tinystories` (see docs/PROVENANCE.md). The committed
    artifact is integer token ids only, never raw text.
    """
    import hashlib

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    parquet_path = hf_hub_download(
        repo_id="wikitext", repo_type="dataset",
        filename=_WIKITEXT103_TEST_PARQUET, cache_dir=hf_cache,
    )
    vocab_path = hf_hub_download(
        repo_id="gpt2", filename="vocab.json", cache_dir=hf_cache)
    with open(vocab_path, "rb") as f:
        tokenizer_sha = hashlib.sha256(f.read()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=hf_cache)

    table = pq.read_table(parquet_path)
    texts = [t for t in table.column("text").to_pylist() if t.strip()]

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
        "source": "wikitext (config wikitext-103-raw-v1) test split",
        "hf_repo": "wikitext",
        "split": "test",
        "license": _LICENSE_WIKITEXT,
        "tokenizer": "gpt2 (GPT-2 byte-level BPE)",
        "tokenizer_sha": tokenizer_sha,
        "build_command": (
            "certinf.corpus.build_wikitext103(context_lengths="
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
