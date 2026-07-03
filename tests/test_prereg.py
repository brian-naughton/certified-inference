import dataclasses
import hashlib
import json
import sys

import pytest

from certinf import corpus, prereg


def _tiny_corpus(tmp_path, n_windows=6, ctx_len=8):
    windows = {ctx_len: [[i] * ctx_len for i in range(n_windows)]}
    out = tmp_path / "tiny.ids.json"
    corpus.build_fixed(windows, out_path=str(out))
    return str(out), n_windows


def _spec(**overrides):
    spec = dict(
        model="tinystories",
        checkpoint_sha256="a" * 64,
        context_length=8,
        P_max=256,
        n=20,
        delta="0.05",                                       # exact (M1)
        delta_split={"phi1": "0.025", "phi2_joint": "0.025"},
        seed=1234,
        phi_definitions={
            "phi1": "argmax certified at P <= P_max",
            "phi2_joint": "pinned float32 top-1 agrees with certified exact-real top-1",
        },
        escalation_policy={"P_grid": [128, 160, 192, 224, 256], "P_max": 256},
    )
    spec.update(overrides)
    return spec


def test_freeze_produces_n_indices_in_range_with_duplicates_possible(tmp_path):
    corpus_path, n_windows = _tiny_corpus(tmp_path)
    out_dir = tmp_path / "freeze1"
    doc = prereg.freeze(_spec(n=50), corpus_path, str(out_dir))

    index_doc = json.loads((out_dir / "sample-index.json").read_text())
    indices = index_doc["indices"]
    assert len(indices) == 50
    assert all(isinstance(i, int) and 0 <= i < n_windows for i in indices)
    assert len(set(indices)) < len(indices)  # duplicates expected with n=50 > n_windows=6

    assert doc["n"] == 50
    sha = doc["sample_index_sha256"]
    assert isinstance(sha, str) and len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)


def test_freeze_deterministic_same_seed_reproduces_sha(tmp_path):
    corpus_path, _ = _tiny_corpus(tmp_path)
    doc1 = prereg.freeze(_spec(seed=42), corpus_path, str(tmp_path / "run1"))
    doc2 = prereg.freeze(_spec(seed=42), corpus_path, str(tmp_path / "run2"))
    assert doc1["sample_index_sha256"] == doc2["sample_index_sha256"]
    assert doc1["prereg_sha256"] == doc2["prereg_sha256"]


def test_freeze_different_seed_changes_sha(tmp_path):
    corpus_path, _ = _tiny_corpus(tmp_path)
    doc1 = prereg.freeze(_spec(seed=1), corpus_path, str(tmp_path / "run1"))
    doc2 = prereg.freeze(_spec(seed=2), corpus_path, str(tmp_path / "run2"))
    assert doc1["sample_index_sha256"] != doc2["sample_index_sha256"]


def test_verify_true_for_matching_pair(tmp_path):
    corpus_path, _ = _tiny_corpus(tmp_path)
    out_dir = tmp_path / "run1"
    prereg.freeze(_spec(seed=7), corpus_path, str(out_dir))
    assert prereg.verify(str(out_dir / "prereg.json"), corpus_path) is True


def test_verify_false_if_corpus_sha_differs(tmp_path):
    corpus_path, _ = _tiny_corpus(tmp_path)
    out_dir = tmp_path / "run1"
    prereg.freeze(_spec(seed=7), corpus_path, str(out_dir))

    other_corpus_path, _ = _tiny_corpus(tmp_path / "other", n_windows=9)
    assert prereg.verify(str(out_dir / "prereg.json"), other_corpus_path) is False


def test_delta_split_sums_to_delta():
    from fractions import Fraction
    spec = _spec()
    # M1: exact rational equality, no float tolerance.
    total = sum((Fraction(v) for v in spec["delta_split"].values()), Fraction(0))
    assert total == Fraction(spec["delta"])
    assert spec["delta_split"] == {"phi1": "0.025", "phi2_joint": "0.025"}


def test_freeze_rejects_inconsistent_delta_split(tmp_path):
    corpus_path, _ = _tiny_corpus(tmp_path)
    bad_spec = _spec(delta="0.05", delta_split={"phi1": "0.01", "phi2_joint": "0.01"})
    with pytest.raises(ValueError, match="delta_split"):
        prereg.freeze(bad_spec, corpus_path, str(tmp_path / "bad"))


def test_freeze_stores_delta_as_exact_pairs_not_float(tmp_path):
    """M1: the written artifact carries delta / delta_split as ["num","den"]
    Fraction pairs — no float anywhere in the delta budget."""
    from fractions import Fraction
    corpus_path, _ = _tiny_corpus(tmp_path)
    doc = prereg.freeze(_spec(seed=7), corpus_path, str(tmp_path / "run1"))
    assert doc["delta"] == ["1", "20"]
    assert doc["delta_split"] == {"phi1": ["1", "40"], "phi2_joint": ["1", "40"]}
    assert Fraction(int(doc["delta"][0]), int(doc["delta"][1])) == Fraction(1, 20)
    assert doc["prereg_format_version"] == prereg.PREREG_FORMAT_VERSION


def test_freeze_rejects_float_delta(tmp_path):
    """M1: a bare float delta is rejected — exactness is not optional."""
    corpus_path, _ = _tiny_corpus(tmp_path)
    with pytest.raises(ValueError, match="never a float"):
        prereg.freeze(_spec(delta=0.05), corpus_path, str(tmp_path / "badf"))


def test_freeze_accepts_ratio_and_pair_delta_forms(tmp_path):
    """M1: '1/20' (ratio string) and [1, 20] (pair) are equivalent exact
    spellings of the same delta and freeze identically."""
    corpus_path, _ = _tiny_corpus(tmp_path)
    d_ratio = prereg.freeze(
        _spec(seed=7, delta="1/20",
              delta_split={"phi1": "1/40", "phi2_joint": "1/40"}),
        corpus_path, str(tmp_path / "ratio"))
    d_pair = prereg.freeze(
        _spec(seed=7, delta=["1", "20"],
              delta_split={"phi1": [1, 40], "phi2_joint": [1, 40]}),
        corpus_path, str(tmp_path / "pair"))
    assert d_ratio["delta"] == d_pair["delta"] == ["1", "20"]
    assert d_ratio["prereg_sha256"] == d_pair["prereg_sha256"]


def test_freeze_rejects_tolerance_near_miss_split(tmp_path):
    """M1: a delta split summing to delta +/- 1e-13 — which the old 1e-12
    float tolerance accepted — is now REJECTED under exact rational equality."""
    corpus_path, _ = _tiny_corpus(tmp_path)
    near = _spec(delta="0.05",
                 delta_split={"phi1": "0.025", "phi2_joint": "0.0250000000001"})
    with pytest.raises(ValueError, match="EXACTLY"):
        prereg.freeze(near, corpus_path, str(tmp_path / "nearmiss"))


def test_hoeffding_block_records_exact_inputs_and_conservative_epsilon(tmp_path):
    """M1: the Hoeffding block records exact (n, delta) and an epsilon display
    rounded UP (so any k/n - epsilon lower bound displayed rounds DOWN)."""
    from decimal import Decimal
    corpus_path, _ = _tiny_corpus(tmp_path)
    doc = prereg.freeze(_spec(seed=7, n=500), corpus_path, str(tmp_path / "h"))
    h = doc["hoeffding"]
    assert h["n"] == 500
    assert h["delta"] == ["1", "20"]
    assert h["epsilon_display_rounding"] == "up"
    # epsilon(500, 1/20) = sqrt(ln(20)/1000) ~= 0.0547337; rounded UP at 6 dp.
    assert h["epsilon_display"] == "0.054734"
    # rounded-UP display is >= the true value (never understates epsilon).
    assert Decimal(h["epsilon_display"]) >= Decimal("0.0547337")


def test_freeze_rejects_missing_spec_field(tmp_path):
    corpus_path, _ = _tiny_corpus(tmp_path)
    spec = _spec()
    del spec["seed"]
    with pytest.raises(ValueError, match="seed"):
        prereg.freeze(spec, corpus_path, str(tmp_path / "bad2"))


def test_prereg_dataclass_is_frozen():
    p = prereg.PreRegistration(
        prereg_format_version=prereg.PREREG_FORMAT_VERSION,
        model="tinystories", checkpoint_sha256="a" * 64, corpus_sha256="b" * 64,
        context_length=8, P_max=256, n=20, delta=["1", "20"],
        delta_split={"phi1": ["1", "40"], "phi2_joint": ["1", "40"]},
        hoeffding={}, seed=1,
        phi_definitions={}, escalation_policy={}, sample_index_sha256="c" * 64,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.seed = 2


def test_verify_false_on_tampered_index_file(tmp_path):
    corpus_path, _ = _tiny_corpus(tmp_path)
    out_dir = tmp_path / "run1"
    prereg.freeze(_spec(seed=7), corpus_path, str(out_dir))

    index_path = out_dir / "sample-index.json"
    index_doc = json.loads(index_path.read_text())
    index_doc["indices"][0] = (index_doc["indices"][0] + 1) % 6
    index_path.write_text(json.dumps(index_doc))

    assert prereg.verify(str(out_dir / "prereg.json"), corpus_path) is False


def _resign_prereg(prereg_path):
    """Recompute prereg_sha256 over the current file contents (a
    self-consistent re-freeze, as a hostile-but-careful editor would do) and
    write it back."""
    prereg_dict = json.loads(prereg_path.read_text())
    body = {k: v for k, v in prereg_dict.items() if k != "prereg_sha256"}
    prereg_dict["prereg_sha256"] = hashlib.sha256(
        prereg.canonical_json(body).encode()).hexdigest()
    prereg_path.write_text(json.dumps(prereg_dict))
    return prereg_dict


def test_verify_false_on_edited_field_without_resigning(tmp_path):
    """I2(a): editing a prereg.json field (n or seed) WITHOUT recomputing
    prereg_sha256 must fail — the cheapest possible tamper."""
    corpus_path, _ = _tiny_corpus(tmp_path)
    out_dir = tmp_path / "run1"
    prereg.freeze(_spec(seed=7), corpus_path, str(out_dir))

    prereg_path = out_dir / "prereg.json"
    prereg_dict = json.loads(prereg_path.read_text())
    prereg_dict["n"] = prereg_dict["n"] + 1
    prereg_path.write_text(json.dumps(prereg_dict))

    assert prereg.verify(str(prereg_path), corpus_path) is False


def test_verify_true_on_self_consistent_reedit_of_non_binding_field(tmp_path):
    """I2(b): editing prereg.json AND recomputing prereg_sha256 (a
    self-consistent re-freeze of a field that doesn't feed the corpus/index
    binding, e.g. `model`) is EXPECTED to verify True.

    This is not a bug: `verify()` can only witness internal arithmetic
    consistency (the sha recomputes, the delta split still balances, the
    index still re-draws bit-identically from the seed). It has no way to
    see wall-clock history, so it cannot by itself distinguish "this was the
    original frozen tuple" from "this was silently edited and re-signed
    after the fact". Pre-commitment against exactly this kind of edit is
    established externally — by publishing `prereg_sha256` as a
    certificate's `prereg_ref` and by `prereg.json`'s own commit/publication
    timestamp in git history — not by anything `verify()` reads from disk.
    This test pins that semantic on purpose (see module docstring's WITNESS
    SEMANTICS section and `verify`'s docstring).
    """
    corpus_path, _ = _tiny_corpus(tmp_path)
    out_dir = tmp_path / "run1"
    prereg.freeze(_spec(seed=7), corpus_path, str(out_dir))

    prereg_path = out_dir / "prereg.json"
    prereg_dict = json.loads(prereg_path.read_text())
    prereg_dict["model"] = "a-completely-different-model"
    prereg_path.write_text(json.dumps(prereg_dict))
    _resign_prereg(prereg_path)

    assert prereg.verify(str(prereg_path), corpus_path) is True


def test_verify_false_on_tampered_index_with_recomputed_sha(tmp_path):
    """I2(c): tampering sample-index.json's indices AND recomputing its own
    sha (propagated into a re-signed prereg.json) still fails — caught by
    the re-draw, not by either sha check, because the indices themselves are
    not derivable from anything except the seed."""
    corpus_path, n_windows = _tiny_corpus(tmp_path)
    out_dir = tmp_path / "run1"
    prereg.freeze(_spec(seed=7), corpus_path, str(out_dir))

    index_path = out_dir / "sample-index.json"
    index_doc = json.loads(index_path.read_text())
    index_doc["indices"][0] = (index_doc["indices"][0] + 1) % n_windows
    new_index_sha = hashlib.sha256(
        prereg.canonical_json(index_doc).encode()).hexdigest()
    index_path.write_text(json.dumps(index_doc))

    prereg_path = out_dir / "prereg.json"
    prereg_dict = json.loads(prereg_path.read_text())
    prereg_dict["sample_index_sha256"] = new_index_sha
    prereg_path.write_text(json.dumps(prereg_dict))
    _resign_prereg(prereg_path)

    # Both shas now check out internally; only the seed-redraw catches it.
    assert prereg.verify(str(prereg_path), corpus_path) is False


def test_verify_false_on_missing_index_file(tmp_path):
    """I2(d): a missing sample-index.json must return False, not raise."""
    corpus_path, _ = _tiny_corpus(tmp_path)
    out_dir = tmp_path / "run1"
    prereg.freeze(_spec(seed=7), corpus_path, str(out_dir))

    (out_dir / "sample-index.json").unlink()

    assert prereg.verify(str(out_dir / "prereg.json"), corpus_path) is False


def test_verify_false_on_broken_delta_split_even_when_resigned(tmp_path):
    """I1: verify() must itself re-check the delta-split invariant — a
    re-frozen prereg.json with a broken delta budget must not pass just
    because its own sha recomputes."""
    corpus_path, _ = _tiny_corpus(tmp_path)
    out_dir = tmp_path / "run1"
    prereg.freeze(_spec(seed=7), corpus_path, str(out_dir))

    prereg_path = out_dir / "prereg.json"
    prereg_dict = json.loads(prereg_path.read_text())
    # sums to 1/50 (0.02), not 1/20 (0.05)
    prereg_dict["delta_split"] = {"phi1": ["1", "100"], "phi2_joint": ["1", "100"]}
    prereg_path.write_text(json.dumps(prereg_dict))
    _resign_prereg(prereg_path)

    assert prereg.verify(str(prereg_path), corpus_path) is False


def test_verify_false_on_tolerance_near_miss_split_resigned(tmp_path):
    """M1: a re-signed prereg.json whose split sums to delta +/- 1e-13 (an
    old-tolerance near-miss) must fail verify() under exact rational equality."""
    corpus_path, _ = _tiny_corpus(tmp_path)
    out_dir = tmp_path / "run1"
    prereg.freeze(_spec(seed=7), corpus_path, str(out_dir))

    prereg_path = out_dir / "prereg.json"
    prereg_dict = json.loads(prereg_path.read_text())
    # 1/40 + (1/40 + 1e-13) = 1/20 + 1e-13: within the retired 1e-12 tolerance,
    # but not exactly 1/20. Spell the perturbed half as an exact decimal.
    prereg_dict["delta_split"] = {
        "phi1": ["1", "40"],
        "phi2_joint": ["250000000001", "10000000000000"],  # 0.0250000000001
    }
    prereg_path.write_text(json.dumps(prereg_dict))
    _resign_prereg(prereg_path)

    assert prereg.verify(str(prereg_path), corpus_path) is False


def test_verify_false_on_tampered_hoeffding_epsilon_display(tmp_path):
    """M1: a tampered epsilon display (inconsistent with the exact (n, delta))
    fails the witness even when the prereg is re-signed."""
    corpus_path, _ = _tiny_corpus(tmp_path)
    out_dir = tmp_path / "run1"
    prereg.freeze(_spec(seed=7), corpus_path, str(out_dir))

    prereg_path = out_dir / "prereg.json"
    prereg_dict = json.loads(prereg_path.read_text())
    prereg_dict["hoeffding"]["epsilon_display"] = "0.000001"  # absurdly small
    prereg_path.write_text(json.dumps(prereg_dict))
    _resign_prereg(prereg_path)

    assert prereg.verify(str(prereg_path), corpus_path) is False


def test_verify_false_on_unknown_format_version(tmp_path):
    """M1: an unrecognised prereg_format_version fails the witness (format 1's
    unversioned float budget is deliberately not accepted)."""
    corpus_path, _ = _tiny_corpus(tmp_path)
    out_dir = tmp_path / "run1"
    prereg.freeze(_spec(seed=7), corpus_path, str(out_dir))

    prereg_path = out_dir / "prereg.json"
    prereg_dict = json.loads(prereg_path.read_text())
    prereg_dict["prereg_format_version"] = 1
    prereg_path.write_text(json.dumps(prereg_dict))
    _resign_prereg(prereg_path)

    assert prereg.verify(str(prereg_path), corpus_path) is False


def test_verify_false_on_index_doc_seed_mismatch_with_prereg(tmp_path):
    """M4: sample-index.json's own embedded seed must match prereg.json's.
    Tampering only the embedded metadata (not the indices themselves) would
    otherwise sail through the sha + re-draw checks, since the re-draw uses
    prereg.json's seed, not the index doc's — this is exactly the gap M4
    closes."""
    corpus_path, _ = _tiny_corpus(tmp_path)
    out_dir = tmp_path / "run1"
    prereg.freeze(_spec(seed=7), corpus_path, str(out_dir))

    index_path = out_dir / "sample-index.json"
    index_doc = json.loads(index_path.read_text())
    index_doc["seed"] = index_doc["seed"] + 1  # indices left untouched
    new_index_sha = hashlib.sha256(
        prereg.canonical_json(index_doc).encode()).hexdigest()
    index_path.write_text(json.dumps(index_doc))

    prereg_path = out_dir / "prereg.json"
    prereg_dict = json.loads(prereg_path.read_text())
    prereg_dict["sample_index_sha256"] = new_index_sha
    prereg_path.write_text(json.dumps(prereg_dict))
    _resign_prereg(prereg_path)

    assert prereg.verify(str(prereg_path), corpus_path) is False


def test_freeze_records_python_version_in_both_artifacts(tmp_path):
    """M2: both artifacts record [major, minor, micro] of the interpreter
    that froze them."""
    corpus_path, _ = _tiny_corpus(tmp_path)
    out_dir = tmp_path / "run1"
    doc = prereg.freeze(_spec(seed=7), corpus_path, str(out_dir))

    index_doc = json.loads((out_dir / "sample-index.json").read_text())
    expected = [sys.version_info.major, sys.version_info.minor, sys.version_info.micro]
    assert doc["python_version"] == expected
    assert index_doc["python_version"] == expected


def test_verify_warns_not_fails_on_python_version_drift(tmp_path):
    """M2: a recorded python_version that disagrees with the running
    interpreter is benign drift (warn) as long as the verbatim index sha and
    re-draw still check out — it must not fail verification by itself."""
    corpus_path, _ = _tiny_corpus(tmp_path)
    out_dir = tmp_path / "run1"
    prereg.freeze(_spec(seed=7), corpus_path, str(out_dir))

    index_path = out_dir / "sample-index.json"
    index_doc = json.loads(index_path.read_text())
    index_doc["python_version"] = [3, 1, 0]  # obviously-fake old version
    new_index_sha = hashlib.sha256(
        prereg.canonical_json(index_doc).encode()).hexdigest()
    index_path.write_text(json.dumps(index_doc))

    prereg_path = out_dir / "prereg.json"
    prereg_dict = json.loads(prereg_path.read_text())
    prereg_dict["sample_index_sha256"] = new_index_sha
    prereg_dict["python_version"] = [3, 1, 0]
    prereg_path.write_text(json.dumps(prereg_dict))
    _resign_prereg(prereg_path)

    with pytest.warns(UserWarning, match="Python"):
        result = prereg.verify(str(prereg_path), corpus_path)
    assert result is True
