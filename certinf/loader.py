#!/usr/bin/env python3
"""Sha-pinned, torch-free-downstream hex export of pretrained checkpoints.

torch is used HERE ONLY (house rule: the certified path — exact.py,
ival_ext.py, interval_fwd.py, gpt2_interval.py, certify.py — stays stdlib).
This module loads the pinned `pytorch_model.bin` for TinyStories-1M or
GPT-2-small, verifies its sha256 against `CHECKPOINT_SHA256`, and dumps every
named weight tensor as nested `float(v).hex()` strings.

Exact-real caveat: every stored weight is float32 (<=24-bit mantissa), and
Python's `float` is a float64 (53-bit mantissa), so widening float32 -> float64
is exact (no rounding) and `float.hex()` is a lossless round-trip literal —
`float.fromhex(node) == original_value` bit-for-bit. Downstream stdlib code
(certinf.interval_fwd, certinf.gpt2_interval) already treats every weight as
an exact dyadic rational (`Fraction(v)` / frexp-based common-denominator
conversion); this export introduces no additional rounding beyond that
existing float32->real assumption.

Size note (deviation from certgrok's raw `float.hex()` output, NOT from its
exactness discipline): redundant trailing zeros are stripped from the hex
mantissa ("0x1.921fb60000000p+1" -> "0x1.921fb6p+1"). `float.fromhex` parses
the short form back to the identical float64 — the round-trip test asserts
this per element — and it shrinks the artifacts ~30% (float32-origin values
carry at most 6 significant hex digits vs float.hex()'s fixed 13). The
GPT-2-small export is still ~2 GB (124M weights x ~16 bytes/element is a
floor for any hex-text encoding) — see docs/PROVENANCE.md for why that
artifact is regenerated on demand rather than committed.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pinned checkpoint provenance (Task 0.4 Step 1) — verified independently via
# `shasum -a 256` against the downloaded blobs; see docs/PROVENANCE.md.
CHECKPOINT_SHA256 = {
    "tinystories-1M": "07f9609ea882b8163ff3b23d40e2b82cb715d409631beb15c84b164f3877dae7",
    "gpt2-small": "7c5d3f4b8b76583b422fcb9189ad6c89d5d97a094541ce8932dce3ecabde1421",
}

HF_REPO = {
    "tinystories-1M": "roneneldan/TinyStories-1M",
    "gpt2-small": "gpt2",
}

# HF hub cache snapshot dirnames, per model (mirrors the glob-based cache
# resolution in certinf/float_fwd.py and certinf/gpt2_float.py: a local,
# gitignored `data/hf_cache` populated by `huggingface-cli download`,
# overridable with the LMCERT_HF_CACHE env var). Resolved lazily (only when
# export_weights actually needs the file) so importing this module — e.g. to
# read CHECKPOINT_SHA256 — never requires the cache to be populated.
_CACHE_SNAPSHOT_DIR = {
    "tinystories-1M": "models--roneneldan--TinyStories-1M",
    "gpt2-small": "models--gpt2",
}

# TinyStories-1M (GPT-Neo arch, 8 layers) — names as consumed by
# certinf.interval_fwd.prepare_weights / certinf.float_fwd.forward.
_TS_N_LAYERS = 8
TS_TENSOR_NAMES = ["transformer.wte.weight", "transformer.wpe.weight"]
for _L in range(_TS_N_LAYERS):
    _p = f"transformer.h.{_L}."
    TS_TENSOR_NAMES += [
        _p + "ln_1.weight", _p + "ln_1.bias",
        _p + "attn.attention.q_proj.weight",
        _p + "attn.attention.k_proj.weight",
        _p + "attn.attention.v_proj.weight",
        _p + "attn.attention.out_proj.weight",
        _p + "attn.attention.out_proj.bias",
        _p + "ln_2.weight", _p + "ln_2.bias",
        _p + "mlp.c_fc.weight", _p + "mlp.c_fc.bias",
        _p + "mlp.c_proj.weight", _p + "mlp.c_proj.bias",
    ]
TS_TENSOR_NAMES += ["transformer.ln_f.weight", "transformer.ln_f.bias"]

TS_CFG = dict(n_layers=8, n_heads=16, d_model=64, d_head=4, d_mlp=256,
              vocab_size=50257, n_ctx=2048)

# GPT-2-small (12 layers) — names as consumed by
# certinf.gpt2_interval.interval_forward_gpt2 / certinf.gpt2_float.forward.
# wte.weight is TIED (also the unembedding); attn.bias / attn.masked_bias are
# non-parameter causal-mask buffers and are deliberately excluded.
_GPT2_N_LAYERS = 12
GPT2_TENSOR_NAMES = ["wte.weight", "wpe.weight"]
for _L in range(_GPT2_N_LAYERS):
    _p = f"h.{_L}."
    GPT2_TENSOR_NAMES += [
        _p + "ln_1.weight", _p + "ln_1.bias",
        _p + "attn.c_attn.weight", _p + "attn.c_attn.bias",
        _p + "attn.c_proj.weight", _p + "attn.c_proj.bias",
        _p + "ln_2.weight", _p + "ln_2.bias",
        _p + "mlp.c_fc.weight", _p + "mlp.c_fc.bias",
        _p + "mlp.c_proj.weight", _p + "mlp.c_proj.bias",
    ]
GPT2_TENSOR_NAMES += ["ln_f.weight", "ln_f.bias"]

GPT2_CFG = dict(n_layers=12, n_heads=12, d_model=768, d_head=64, d_mlp=3072,
                 vocab_size=50257, n_ctx=1024)

_TENSOR_NAMES = {"tinystories-1M": TS_TENSOR_NAMES, "gpt2-small": GPT2_TENSOR_NAMES}
_CFG = {"tinystories-1M": TS_CFG, "gpt2-small": GPT2_CFG}


def _hf_cache_dir() -> str:
    return os.environ.get("LMCERT_HF_CACHE", os.path.join(_REPO_ROOT, "data", "hf_cache"))


def _locate_checkpoint(model_key: str) -> str:
    """Resolve the pinned `pytorch_model.bin` path for `model_key`.

    Raises:
        FileNotFoundError: no snapshot found under the resolved HF cache dir.
    """
    cache = _hf_cache_dir()
    pattern = os.path.join(cache, _CACHE_SNAPSHOT_DIR[model_key],
                            "snapshots", "*", "pytorch_model.bin")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"{model_key} weights not found under {cache!r} (pattern "
            f"{pattern!r}). Fetch them first — see docs/PROVENANCE.md.")
    return candidates[0]


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hex_str(v: float) -> str:
    """Lossless short hex literal: float.hex() minus redundant mantissa zeros.

    "0x1.921fb60000000p+1" -> "0x1.921fb6p+1"; "0x0.0p+0" -> "0x0p+0".
    float.fromhex(result) == v exactly (see module docstring's size note).
    """
    m, _, e = float(v).hex().partition("p")
    if "." in m:
        m = m.rstrip("0").rstrip(".")
    return f"{m}p{e}"


def _to_hex(node):
    """Nested lists of floats (from Tensor.tolist()) -> nested hex strings."""
    if isinstance(node, float):
        return _hex_str(node)
    return [_to_hex(child) for child in node]


def export_weights(model_key: str, out_path: str) -> dict:
    """Sha-verify the pinned checkpoint and write a torch-free hex JSON export.

    Args:
        model_key: "tinystories-1M" or "gpt2-small".
        out_path: destination JSON path.

    Returns:
        The exported blob: `{"meta": {...}, "cfg": {...}, "state_dict_hex": {...}}`.

    Raises:
        ValueError: model_key not recognised, or the checkpoint's sha256
            doesn't match the pinned `CHECKPOINT_SHA256[model_key]`.
        FileNotFoundError: checkpoint not present under the resolved HF cache.
    """
    if model_key not in CHECKPOINT_SHA256:
        raise ValueError(
            f"unknown model_key {model_key!r}; expected one of "
            f"{sorted(CHECKPOINT_SHA256)}")
    bin_path = _locate_checkpoint(model_key)
    got_sha = _sha256_of(bin_path)
    expected_sha = CHECKPOINT_SHA256[model_key]
    if got_sha != expected_sha:
        raise ValueError(
            f"checkpoint sha256 mismatch for {model_key}: got {got_sha}, "
            f"expected {expected_sha}")

    sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    blob = {
        "meta": {
            "model": model_key,
            "hf_repo": HF_REPO[model_key],
            "checkpoint_sha256": got_sha,
            "dtype": "float32",
        },
        "cfg": _CFG[model_key],
        "state_dict_hex": {
            name: _to_hex(sd[name].float().tolist())
            for name in _TENSOR_NAMES[model_key]
        },
    }
    with open(out_path, "w") as f:
        json.dump(blob, f)
    return blob


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else "tinystories-1M"
    out = sys.argv[2] if len(sys.argv) > 2 else f"certificates/{key}.weights.json"
    export_weights(key, out)
    print(f"wrote {out}")
