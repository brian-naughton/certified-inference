import json
import os

import pytest

torch = pytest.importorskip("torch")

from certinf import loader  # noqa: E402


def _flatten_hex(node, out):
    if isinstance(node, str):
        out.append(node)
    else:
        for child in node:
            _flatten_hex(child, out)


def _assert_roundtrips_exactly(node, tensor, name):
    """float.fromhex(node) == tensor[idx].item() for every element, flattened
    (equivalent to the nested-index comparison but avoids per-leaf tensor
    indexing overhead across GPT-2's 124M elements)."""
    flat_hex: list = []
    _flatten_hex(node, flat_hex)
    flat_vals = tensor.flatten().tolist()
    assert len(flat_hex) == len(flat_vals), name
    for h, v in zip(flat_hex, flat_vals):
        assert float.fromhex(h) == v, name


def _check_blob(blob, model_key: str, expected_cfg: dict, tensor_names: list):
    sd = torch.load(loader._locate_checkpoint(model_key),
                    map_location="cpu", weights_only=True)
    assert blob["meta"]["checkpoint_sha256"] == loader.CHECKPOINT_SHA256[model_key]
    assert blob["meta"]["model"] == model_key
    assert blob["meta"]["hf_repo"] == loader.HF_REPO[model_key]
    assert blob["meta"]["dtype"] == "float32"
    assert blob["cfg"] == expected_cfg
    assert set(blob["state_dict_hex"]) == set(tensor_names)
    for name in tensor_names:
        _assert_roundtrips_exactly(blob["state_dict_hex"][name],
                                   sd[name].float(), name)


@pytest.mark.torch
def test_export_tinystories_roundtrips_exactly(tmp_path):
    out = tmp_path / "tinystories-1M.weights.json"
    blob = loader.export_weights("tinystories-1M", str(out))
    # the file IS the returned blob (JSON round-trip of the serialised form)
    with open(out) as f:
        assert json.load(f) == blob
    _check_blob(blob, "tinystories-1M", loader.TS_CFG, loader.TS_TENSOR_NAMES)


@pytest.mark.torch
@pytest.mark.slow
def test_export_gpt2_roundtrips_exactly():
    # The serialised GPT-2 export is ~2 GB; writing it to a pytest tmp dir
    # filled the dev disk (see docs/PROVENANCE.md). The JSON file<->blob
    # round-trip is model-independent and covered by the TinyStories test, so
    # here the full 124M-element hex round-trip is asserted on the returned
    # blob with the file write discarded.
    blob = loader.export_weights("gpt2-small", os.devnull)
    _check_blob(blob, "gpt2-small", loader.GPT2_CFG, loader.GPT2_TENSOR_NAMES)


@pytest.mark.torch
def test_hex_str_strips_losslessly():
    cases = [0.0, -0.0, 1.0, -1.0, 0.5, 3.14159265, -2.5e-8, 65504.0,
             float(torch.tensor(0.1, dtype=torch.float32))]
    for v in cases:
        s = loader._hex_str(v)
        assert float.fromhex(s) == v, (v, s)
        assert "0000000p" not in s, s


@pytest.mark.torch
def test_unknown_model_key_raises(tmp_path):
    with pytest.raises(ValueError):
        loader.export_weights("not-a-model", str(tmp_path / "x.json"))


@pytest.mark.torch
def test_sha256_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setitem(loader.CHECKPOINT_SHA256, "tinystories-1M", "0" * 64)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        loader.export_weights("tinystories-1M", str(tmp_path / "x.json"))
