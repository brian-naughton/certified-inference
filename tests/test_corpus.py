import json

import pytest

from certinf import corpus


def test_load_and_sha_roundtrip(tmp_path):
    windows = {8: [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]}
    out = tmp_path / "fixture.ids.json"
    doc = corpus.build_fixed(windows, out_path=str(out))
    assert doc["corpus_sha256"] == corpus.corpus_sha256(windows)

    loaded = corpus.load(str(out))
    assert loaded["corpus_sha256"] == doc["corpus_sha256"]
    assert loaded["windows"]["8"] == windows[8]

    # stable across reload
    reloaded_sha = corpus.corpus_sha256(loaded["windows"])
    assert reloaded_sha == doc["corpus_sha256"]


def test_every_window_has_exact_ctx_len_and_valid_ids(tmp_path):
    windows = {8: [[7454, 2402, 257, 640, 612, 373, 257, 1310]],
              16: [list(range(100, 116))]}
    out = tmp_path / "fixture2.ids.json"
    corpus.build_fixed(windows, out_path=str(out))
    loaded = corpus.load(str(out))
    for ctx_len_str, wins in loaded["windows"].items():
        ctx_len = int(ctx_len_str)
        for w in wins:
            assert len(w) == ctx_len
            assert all(isinstance(i, int) and 0 <= i < 50257 for i in w)


def test_fixed_corpus_meta_has_source_note(tmp_path):
    out = tmp_path / "fixture3.ids.json"
    corpus.build_fixed({8: [[0] * 8]}, out_path=str(out))
    loaded = corpus.load(str(out))
    assert loaded["meta"]["source"]


@pytest.mark.torch
def test_build_tinystories_real_slice(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("pyarrow")
    out = tmp_path / "tinystories-val.ids.json"
    doc = corpus.build_tinystories(context_lengths=(8,), out_path=str(out),
                                   n_windows=5)
    assert len(doc["windows"][8]) == 5
    for w in doc["windows"][8]:
        assert len(w) == 8
        assert all(isinstance(i, int) and 0 <= i < 50257 for i in w)
    assert doc["meta"]["license"] == "cdla-sharing-1.0"
    loaded = corpus.load(str(out))
    assert loaded["corpus_sha256"] == doc["corpus_sha256"]
