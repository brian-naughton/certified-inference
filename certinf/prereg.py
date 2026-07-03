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
Mersenne Twister. `random.Random(seed)`'s seeding is part of the documented,
stable stdlib contract. The unweighted `choices()` draw is stable *in
practice* across the CPython versions this module has been exercised
against, but — unlike `Random.seed`/`random()` — it is not called out by the
language reference as a permanent cross-version guarantee. Two things keep
this from being a silent hazard: (1) the drawn indices are recorded verbatim
in `sample-index.json`, so if a future interpreter's `choices()` ever drifted
the *recorded* indices — not a re-derivation of them — remain the sample of
record; drift would only ever be visible as a `verify()` witness failure
(re-draw mismatch), never as a silently-different accepted sample. (2) the
Python version used at freeze time is recorded in both artifacts (see
`_python_version`) precisely so `verify()` can tell benign interpreter drift
apart from tampering (see `verify`'s docstring).

WITNESS SEMANTICS (read before trusting a `verify() == True`): `verify()`
proves only that the committed artifacts (`prereg.json` + `sample-index.json`)
are the deterministic, arithmetically-consistent output of the committed
`seed` against the committed corpus — i.e. that nobody has silently edited
the sample indices, the corpus binding, or the delta budget out from under
the recorded hashes. It does NOT prove that the seed was chosen before the
results were known: a `prereg.json` can be regenerated (edited fields +
recomputed `prereg_sha256`) after the fact and still verify True, because
`verify()` has no way to see wall-clock history. Pre-commitment — the actual
A2 guarantee — is established externally, by publishing `prereg_sha256` as
`prereg_ref` in a certificate and by the artifact's own commit/publication
timestamp in version control, not by anything `verify()` can check from the
files alone.

The actual headline freeze (a real spec + real pinned corpus, committed
before certification begins) happens in Task 2.1 — this module is only the
machinery: `freeze` draws and commits the sample index + prereg tuple,
`verify` re-draws from the committed seed and asserts the index file
reproduces bit-identically (the freeze witness), re-validates the delta-split
budget, and cross-checks the two artifacts' bindings against each other.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import warnings
from dataclasses import asdict, dataclass

from certinf import corpus as corpus_mod

_REQUIRED_SPEC_FIELDS = (
    "model", "checkpoint_sha256", "context_length", "P_max", "n", "delta",
    "delta_split", "seed", "phi_definitions", "escalation_policy",
)


def canonical_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def _python_version() -> list[int]:
    """`[major, minor, micro]` of the running interpreter — recorded in both
    artifacts at freeze time as provenance only (M2). It is not part of the
    RNG determinism contract: `verify()` treats a mismatch here as benign
    drift, not tamper, as long as the verbatim indices still check out."""
    v = sys.version_info
    return [v.major, v.minor, v.micro]


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
    `prereg_ref` (see `certinf.schema.validate_headline_record`). Both
    artifacts also record `python_version` (`[major, minor, micro]` of the
    interpreter that drew them — M2), purely as provenance for `verify()`
    to distinguish benign interpreter drift from tamper.

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
    python_version = _python_version()

    os.makedirs(out_dir, exist_ok=True)
    index_doc = {
        "seed": seed,
        "corpus_sha256": corpus_sha256,
        "context_length": spec["context_length"],
        "n": n,
        "indices": indices,
        "python_version": python_version,
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
    prereg_dict["python_version"] = python_version
    prereg_dict["prereg_sha256"] = hashlib.sha256(
        canonical_json(prereg_dict).encode()).hexdigest()

    prereg_path = os.path.join(out_dir, "prereg.json")
    with open(prereg_path, "w") as f:
        json.dump(prereg_dict, f, indent=1)

    return prereg_dict


def verify(prereg_path: str, corpus_path: str) -> bool:
    """Re-draw from the committed `(seed, corpus_sha)` and assert the
    committed `sample-index.json` reproduces bit-identically — the freeze
    witness (A2 precommit). Also re-validates the delta-split invariant
    (`abs(sum(delta_split.values()) - delta) <= 1e-12`) and cross-checks
    `sample-index.json`'s own embedded `seed`/`corpus_sha256`/
    `context_length` against `prereg.json`'s, so a self-consistently
    re-hashed artifact whose budget or binding has silently drifted still
    fails. Returns False (never raises) on any mismatch: wrong corpus sha,
    missing/tampered index file, non-reproducing draw, a broken delta split,
    a binding mismatch between the two artifacts, or a tampered `prereg.json`
    (its own `prereg_sha256` fails to recompute).

    IMPORTANT — what `True` does and doesn't mean: a `True` result proves
    the artifacts are the deterministic, arithmetically-consistent output of
    the committed seed against the committed corpus. It does NOT prove the
    seed was chosen before the results were seen — `prereg.json` can be
    edited and `prereg_sha256` recomputed after the fact, and a purely
    internal check like this one cannot detect that (see the module
    docstring's WITNESS SEMANTICS section, and the tamper tests in
    `tests/test_prereg.py` that pin this deliberately). Pre-commitment is
    established externally: by publishing `prereg_sha256` as a certificate's
    `prereg_ref`, and by the artifact's own commit/publication timestamp in
    version control.

    A recorded `python_version` mismatch (M2) between either artifact and
    the running interpreter triggers a `UserWarning` but does not by itself
    fail verification — it is treated as benign interpreter drift, not
    tamper, as long as the verbatim indices and re-draw still check out.
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

    # I1: re-validate the delta-split invariant. freeze() enforces this at
    # write time, but verify() must not simply trust a re-frozen artifact —
    # a self-consistently re-hashed prereg.json with a broken delta budget
    # must fail the witness.
    delta = prereg_dict.get("delta")
    delta_split = prereg_dict.get("delta_split")
    if not isinstance(delta_split, dict) or not isinstance(delta, (int, float)):
        return False
    try:
        split_ok = abs(sum(delta_split.values()) - delta) <= 1e-12
    except TypeError:
        return False
    if not split_ok:
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

    # M4: cross-check sample-index.json's own embedded binding against
    # prereg.json's. Without this, an index file whose *metadata* has been
    # edited (but whose sha and indices were left alone, or vice versa) can
    # silently disagree with the prereg tuple it is supposed to belong to.
    if index_doc.get("seed") != prereg_dict.get("seed"):
        return False
    if index_doc.get("corpus_sha256") != prereg_dict.get("corpus_sha256"):
        return False
    if index_doc.get("context_length") != prereg_dict.get("context_length"):
        return False

    # M2: python-version drift is provenance, not a tamper signal — warn,
    # don't fail. The determinism claim being witnessed here is the redraw
    # below, not interpreter identity.
    current_version = _python_version()
    for label, doc in (("prereg.json", prereg_dict), ("sample-index.json", index_doc)):
        recorded_version = doc.get("python_version")
        if recorded_version is not None and recorded_version != current_version:
            warnings.warn(
                f"{label} was frozen under Python "
                f"{'.'.join(str(x) for x in recorded_version)}; verifying "
                f"under {'.'.join(str(x) for x in current_version)}. This is "
                "benign interpreter drift (random.Random seeding/choices() "
                "is documented-stable / stable-in-practice — see module "
                "docstring), not a tamper signal.",
                stacklevel=2,
            )

    redrawn = _draw_sample_indices(prereg_dict["seed"], len(windows),
                                   prereg_dict["n"])
    return redrawn == index_doc["indices"]
