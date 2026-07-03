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
from decimal import ROUND_CEILING, Decimal, localcontext
from fractions import Fraction

from certinf import corpus as corpus_mod

_REQUIRED_SPEC_FIELDS = (
    "model", "checkpoint_sha256", "context_length", "P_max", "n", "delta",
    "delta_split", "seed", "phi_definitions", "escalation_policy",
)

# M1 — prereg artifact format version. Format 1 (implicit, unversioned) carried
# the delta budget as binary float32/float64 (`delta: 0.05`, `delta_split:
# {..: 0.025}`) and accepted a 1e-12 tolerance in the sum-check. Format 2
# carries delta and every delta_split entry as an EXACT ["num","den"] Fraction
# pair, the sum-check is exact rational equality (no tolerance anywhere in the
# certification path), and a Hoeffding block records the (n, delta) inputs
# exactly plus a conservatively display-rounded epsilon string.
PREREG_FORMAT_VERSION = 2
_KNOWN_FORMAT_VERSIONS = {2}

# Decimal places in the ADVISORY Hoeffding-epsilon display string.
_EPSILON_DISPLAY_DP = 6


def canonical_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def _to_fraction(v) -> Fraction:
    """Parse an EXACT delta value into a Fraction.

    Accepts an int, a decimal- or ratio-string (``"0.05"``, ``"1/20"`` — both
    exact), or a ``[num, den]`` pair. A bare ``float`` is REJECTED: the point
    of M1 is that no binary-float imprecision ever enters the delta budget or
    the certification path. ``Fraction("0.05")`` is exactly ``1/20``, whereas
    ``Fraction(0.05)`` (the float) is ``3602879701896397/72057594037927936``.

    Args:
        v: An int, a decimal/ratio string, or a ``[num, den]`` pair.

    Returns:
        The exact Fraction the value denotes.

    Raises:
        ValueError: If ``v`` is a float or is otherwise unparseable.
    """
    if isinstance(v, bool):
        raise ValueError(f"delta value must be exact, not bool: {v!r}")
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, str):
        return Fraction(v)
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise ValueError(f"delta [num, den] pair must have length 2: {v!r}")
        return Fraction(int(v[0]), int(v[1]))
    if isinstance(v, float):
        raise ValueError(
            "delta value must be given exactly (a decimal string like "
            f"'0.05', a ratio '1/20', or a [num, den] pair) — never a float: {v!r}")
    raise ValueError(f"cannot parse delta value as an exact Fraction: {v!r}")


def _frac_to_pair(f: Fraction) -> list[str]:
    """Serialise a Fraction as ``[num, den]`` decimal strings (arbitrary
    precision; always fully reduced, so the pair is canonical)."""
    return [str(f.numerator), str(f.denominator)]


def _hoeffding_epsilon_display(n: int, delta: Fraction,
                               dp: int = _EPSILON_DISPLAY_DP) -> str:
    """Advisory decimal display of the Hoeffding half-width
    ``epsilon(n, delta) = sqrt(ln(1/delta) / (2n))``, rounded UP at ``dp``
    decimal places.

    Rounding epsilon UP is deliberate. The published population claim is a
    LOWER bound ``k/n - epsilon``, so a larger epsilon yields a smaller (more
    conservative) displayed lower bound: rounding epsilon up therefore rounds
    any displayed lower bound DOWN, so the display can only ever understate —
    never overstate — the certified rate. The authoritative record is the
    exact ``(n, delta)`` pair in the Hoeffding block; this string is cosmetic.

    Computed deterministically at 60 significant digits via :mod:`decimal`
    (ln + sqrt), far below the ``dp``-place rounding boundary, so the ceiling
    is stable across interpreters.

    Args:
        n: Sample size (positive).
        delta: Failure probability, a Fraction in the open interval (0, 1).
        dp: Decimal places in the returned string.

    Returns:
        The epsilon value as a decimal string rounded up to ``dp`` places.

    Raises:
        ValueError: If ``n <= 0`` or ``delta`` is not in (0, 1).
    """
    if n <= 0:
        raise ValueError(f"Hoeffding n must be positive: {n!r}")
    if not (0 < delta < 1):
        raise ValueError(f"Hoeffding delta must lie in (0, 1): {delta!r}")
    with localcontext() as ctx:
        ctx.prec = 60
        delta_dec = Decimal(delta.numerator) / Decimal(delta.denominator)
        # ln(1/delta) = -ln(delta) (delta < 1 => ln(delta) < 0 => positive)
        inner = (-delta_dec.ln()) / (Decimal(2) * Decimal(n))
        eps = inner.sqrt()
        quantum = Decimal(1).scaleb(-dp)
        return str(eps.quantize(quantum, rounding=ROUND_CEILING))


def _hoeffding_block(n: int, delta: Fraction) -> dict:
    """The frozen Hoeffding accounting: exact ``(n, delta)`` inputs plus a
    conservatively display-rounded epsilon (see
    :func:`_hoeffding_epsilon_display`)."""
    return {
        "formula": "epsilon = sqrt(ln(1/delta) / (2*n))",
        "n": n,
        "delta": _frac_to_pair(delta),
        "epsilon_display": _hoeffding_epsilon_display(n, delta),
        "epsilon_display_dp": _EPSILON_DISPLAY_DP,
        "epsilon_display_rounding": "up",
        "display_note": (
            "epsilon is rounded UP so any displayed population lower bound "
            "k/n - epsilon rounds DOWN — the display never overstates the "
            "certified rate. The exact record is (n, delta); the string is "
            "advisory."
        ),
    }


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
    commit before drawing a single certified sample.

    M1 (format 2): ``delta`` is an exact ``["num","den"]`` Fraction pair and
    ``delta_split`` maps each property to such a pair; ``hoeffding`` records
    the exact ``(n, delta)`` epsilon inputs plus an advisory display string.
    No float appears anywhere in the delta budget."""

    prereg_format_version: int
    model: str
    checkpoint_sha256: str
    corpus_sha256: str
    context_length: int
    P_max: int
    n: int
    delta: list                       # ["num","den"] exact Fraction pair
    delta_split: dict                 # {property: ["num","den"]}
    hoeffding: dict
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
    `delta` and every `delta_split` value are given EXACTLY — as a decimal
    string (`"0.05"`), a ratio (`"1/20"`), or a `[num, den]` pair — never a
    float (M1). `delta_split` must sum to `delta` under EXACT rational
    equality, with no tolerance (A2: "delta split explicit") — e.g.
    `{"phi1": "0.025", "phi2_joint": "0.025"}` for `delta="0.05"`. The written
    artifact stores them as canonical `["num","den"]` pairs and records a
    Hoeffding block (exact `(n, delta)` plus a conservatively rounded epsilon
    display).

    No adaptive stopping, no n-extension, no post-hoc promotion: once
    written, `prereg.json` and `sample-index.json` are the precommit record
    and must be committed to version control before any certification run
    reads them.
    """
    missing = [f for f in _REQUIRED_SPEC_FIELDS if f not in spec]
    if missing:
        raise ValueError(f"spec missing required field(s): {missing!r}")

    # M1: parse delta + every split entry as EXACT Fractions; the sum-check is
    # exact rational equality — NO tolerance in the certification path.
    delta_frac = _to_fraction(spec["delta"])
    split_fracs = {k: _to_fraction(v) for k, v in spec["delta_split"].items()}
    split_sum = sum(split_fracs.values(), Fraction(0))
    if split_sum != delta_frac:
        raise ValueError(
            f"delta_split must sum to delta EXACTLY: "
            f"sum={_frac_to_pair(split_sum)} != delta={_frac_to_pair(delta_frac)}")

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
        prereg_format_version=PREREG_FORMAT_VERSION,
        model=spec["model"],
        checkpoint_sha256=spec["checkpoint_sha256"],
        corpus_sha256=corpus_sha256,
        context_length=spec["context_length"],
        P_max=spec["P_max"],
        n=n,
        delta=_frac_to_pair(delta_frac),
        delta_split={k: _frac_to_pair(v) for k, v in split_fracs.items()},
        hoeffding=_hoeffding_block(n, delta_frac),
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
    witness (A2 precommit). Also re-validates the delta-split invariant under
    EXACT rational equality (`sum(Fraction(delta_split[k])) == Fraction(delta)`
    — no tolerance, M1), re-checks the Hoeffding block's epsilon display
    against its own exact `(n, delta)`, and cross-checks
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

    # M1: only known format versions are witnessable. Format 1 (unversioned,
    # float delta) is deliberately not accepted here.
    if prereg_dict.get("prereg_format_version") not in _KNOWN_FORMAT_VERSIONS:
        return False

    # I1 + M1: re-validate the delta-split invariant under EXACT rational
    # equality. freeze() enforces this at write time, but verify() must not
    # simply trust a re-frozen artifact — a self-consistently re-hashed
    # prereg.json with a broken delta budget must fail the witness. NO
    # tolerance: a split summing to delta +/- 1e-13 is REJECTED.
    delta = prereg_dict.get("delta")
    delta_split = prereg_dict.get("delta_split")
    if not isinstance(delta_split, dict):
        return False
    try:
        delta_frac = _to_fraction(delta)
        split_sum = sum((_to_fraction(v) for v in delta_split.values()),
                        Fraction(0))
    except (ValueError, TypeError, ZeroDivisionError):
        return False
    if split_sum != delta_frac:
        return False

    # M1: the Hoeffding block must be self-consistent — its own (n, delta) must
    # match the tuple's n/delta and its epsilon display must recompute exactly
    # (a tampered display or a drifted n/delta fails the witness).
    hoeffding = prereg_dict.get("hoeffding")
    if not isinstance(hoeffding, dict):
        return False
    try:
        h_delta = _to_fraction(hoeffding.get("delta"))
    except (ValueError, TypeError, ZeroDivisionError):
        return False
    if hoeffding.get("n") != prereg_dict.get("n") or h_delta != delta_frac:
        return False
    try:
        expected_eps = _hoeffding_epsilon_display(hoeffding["n"], h_delta)
    except (ValueError, TypeError, KeyError):
        return False
    if hoeffding.get("epsilon_display") != expected_eps:
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
