import dataclasses
import json

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
        delta=0.05,
        delta_split={"phi1": 0.025, "phi2_joint": 0.025},
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
    spec = _spec()
    assert sum(spec["delta_split"].values()) == pytest.approx(spec["delta"])
    assert spec["delta_split"] == {"phi1": 0.025, "phi2_joint": 0.025}


def test_freeze_rejects_inconsistent_delta_split(tmp_path):
    corpus_path, _ = _tiny_corpus(tmp_path)
    bad_spec = _spec(delta=0.05, delta_split={"phi1": 0.01, "phi2_joint": 0.01})
    with pytest.raises(ValueError, match="delta_split"):
        prereg.freeze(bad_spec, corpus_path, str(tmp_path / "bad"))


def test_freeze_rejects_missing_spec_field(tmp_path):
    corpus_path, _ = _tiny_corpus(tmp_path)
    spec = _spec()
    del spec["seed"]
    with pytest.raises(ValueError, match="seed"):
        prereg.freeze(spec, corpus_path, str(tmp_path / "bad2"))


def test_prereg_dataclass_is_frozen():
    p = prereg.PreRegistration(
        model="tinystories", checkpoint_sha256="a" * 64, corpus_sha256="b" * 64,
        context_length=8, P_max=256, n=20, delta=0.05,
        delta_split={"phi1": 0.025, "phi2_joint": 0.025}, seed=1,
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
