"""Pre-registration freeze script (spec §8-A2).

A2 protocol, verbatim from the design spec: calibration samples are never
headline samples. Before any headline certified run, FREEZE and commit:
(model, corpus sha, context length, P_max, n, delta and its split, seed +
sample-index file, phi definitions, escalation policy). delta split
explicit: delta/2 per property per claim, further divided across multiple
headline claims. Seeds/index files committed BEFORE certification (precommit
record). Duplicate draws counted as duplicate trials (with-replacement
design) — no deduplication. No adaptive stopping, no n-extension, no
post-hoc property/corpus/model promotion; exploration and confirmation
strictly separated.

RNG discipline: sample indices are drawn with
`random.Random(seed).choices(range(n_windows), k=n)` — Python's stdlib
Mersenne Twister, pinned by a plain integer seed. The `random` module's
seeding + `choices` algorithm is part of its documented, stable contract,
so this is deterministic and reproducible from the committed seed alone
without pulling in numpy/torch RNG as a freeze dependency.

The actual headline freeze (a real spec + real pinned corpus, committed
before certification begins) happens in Task 2.1 — this module is only the
machinery: `freeze` draws and commits the sample index + prereg tuple,
`verify` re-draws from the committed seed and asserts the index file
reproduces bit-identically (the freeze witness).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass

from certinf import corpus as corpus_mod

_REQUIRED_SPEC_FIELDS = (
    "model", "checkpoint_sha256", "context_length", "P_max", "n", "delta",
    "delta_split", "seed", "phi_definitions", "escalation_policy",
)


def canonical_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class PreRegistration:
    """The frozen A2 pre-registration tuple — everything a headline run must
    commit before drawing a single certified sample."""

    model: str
    checkpoint_sha256: str
    corpus_sha256: str
    context_length: int
    P_max: int
    n: int
    delta: float
    delta_split: dict
    seed: int
    phi_definitions: dict
    escalation_policy: dict
    sample_index_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def _draw_sample_indices(seed: int, n_windows: int, n: int) -> list[int]:
    """The A2 with-replacement draw. Duplicates are kept as duplicate
    trials — never deduplicated (A2)."""
    rng = random.Random(seed)
    return rng.choices(range(n_windows), k=n)


def freeze(spec: dict, corpus_path: str, out_dir: str) -> dict:
    """Freeze a headline pre-registration (A2).

    Draws the `n` with-replacement sample indices from the pinned corpus's
    window list at `spec["context_length"]` using `random.Random(spec[
    "seed"])`, writes `sample-index.json` (the frozen index list — duplicates
    kept) and `prereg.json` (the `PreRegistration` tuple plus
    `prereg_sha256`, the sha256 of the canonical tuple) under `out_dir`, and
    returns the `prereg.json` dict. `prereg_sha256` is the value every
    headline certificate produced under this freeze must carry as
    `prereg_ref` (see `certinf.schema.validate_headline_record`).

    `spec` must supply: model, checkpoint_sha256, context_length, P_max, n,
    delta, delta_split, seed, phi_definitions, escalation_policy.
    `delta_split` must sum to `delta` (A2: "delta split explicit") — e.g.
    `{"phi1": 0.025, "phi2_joint": 0.025}` for `delta=0.05`.

    No adaptive stopping, no n-extension, no post-hoc promotion: once
    written, `prereg.json` and `sample-index.json` are the precommit record
    and must be committed to version control before any certification run
    reads them.
    """
    missing = [f for f in _REQUIRED_SPEC_FIELDS if f not in spec]
    if missing:
        raise ValueError(f"spec missing required field(s): {missing!r}")

    delta = spec["delta"]
    delta_split = spec["delta_split"]
    split_sum = sum(delta_split.values())
    if abs(split_sum - delta) > 1e-12:
        raise ValueError(f"delta_split must sum to delta: sum({delta_split!r})"
                         f"={split_sum!r} != delta={delta!r}")

    corpus_doc = corpus_mod.load(corpus_path)
    corpus_sha256 = corpus_doc["corpus_sha256"]
    windows = corpus_doc["windows"].get(str(spec["context_length"]))
    if windows is None:
        raise ValueError(f"corpus {corpus_path!r} has no windows at "
                         f"context_length={spec['context_length']}")

    n = spec["n"]
    seed = spec["seed"]
    indices = _draw_sample_indices(seed, len(windows), n)

    os.makedirs(out_dir, exist_ok=True)
    index_doc = {
        "seed": seed,
        "corpus_sha256": corpus_sha256,
        "context_length": spec["context_length"],
        "n": n,
        "indices": indices,
    }
    index_path = os.path.join(out_dir, "sample-index.json")
    with open(index_path, "w") as f:
        json.dump(index_doc, f, indent=1)
    sample_index_sha256 = hashlib.sha256(
        canonical_json(index_doc).encode()).hexdigest()

    prereg = PreRegistration(
        model=spec["model"],
        checkpoint_sha256=spec["checkpoint_sha256"],
        corpus_sha256=corpus_sha256,
        context_length=spec["context_length"],
        P_max=spec["P_max"],
        n=n,
        delta=delta,
        delta_split=delta_split,
        seed=seed,
        phi_definitions=spec["phi_definitions"],
        escalation_policy=spec["escalation_policy"],
        sample_index_sha256=sample_index_sha256,
    )
    prereg_dict = prereg.to_dict()
    prereg_dict["prereg_sha256"] = hashlib.sha256(
        canonical_json(prereg_dict).encode()).hexdigest()

    prereg_path = os.path.join(out_dir, "prereg.json")
    with open(prereg_path, "w") as f:
        json.dump(prereg_dict, f, indent=1)

    return prereg_dict


def verify(prereg_path: str, corpus_path: str) -> bool:
    """Re-draw from the committed `(seed, corpus_sha)` and assert the
    committed `sample-index.json` reproduces bit-identically — the freeze
    witness (A2 precommit). Returns False (never raises) on any mismatch:
    wrong corpus sha, missing/tampered index file, non-reproducing draw, or
    a tampered `prereg.json` (its own `prereg_sha256` fails to recompute).
    """
    try:
        with open(prereg_path) as f:
            prereg_dict = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    stored_prereg_sha = prereg_dict.get("prereg_sha256")
    body = {k: v for k, v in prereg_dict.items() if k != "prereg_sha256"}
    if stored_prereg_sha != hashlib.sha256(
        canonical_json(body).encode()).hexdigest():
        return False

    try:
        corpus_doc = corpus_mod.load(corpus_path)
    except (OSError, json.JSONDecodeError):
        return False
    if corpus_doc["corpus_sha256"] != prereg_dict["corpus_sha256"]:
        return False

    windows = corpus_doc["windows"].get(str(prereg_dict["context_length"]))
    if windows is None:
        return False

    index_path = os.path.join(os.path.dirname(prereg_path), "sample-index.json")
    try:
        with open(index_path) as f:
            index_doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    recomputed_index_sha = hashlib.sha256(
        canonical_json(index_doc).encode()).hexdigest()
    if recomputed_index_sha != prereg_dict["sample_index_sha256"]:
        return False

    redrawn = _draw_sample_indices(prereg_dict["seed"], len(windows),
                                   prereg_dict["n"])
    return redrawn == index_doc["indices"]
