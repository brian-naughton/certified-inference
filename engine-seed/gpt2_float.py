#!/usr/bin/env python3
"""Manual float reference forward for HuggingFace `gpt2` (GPT-2 small, 124M).

Reference semantics (HF GPT2LMHeadModel):
  - pre-LN blocks: x += attn(ln_1(x)); x += mlp(ln_2(x)); final ln_f.
  - Conv1D layout: weights stored (in, out) — y = x @ W + b (NOT nn.Linear's
    x @ W.T). c_attn is fused qkv 768->2304 (split q|k|v), c_proj 768->768,
    mlp c_fc 768->3072, c_proj 3072->768; ALL with biases.
  - attention: 12 heads x d_head 64; scale 1/sqrt(64) = 1/8 (EXACT rational);
    causal mask; softmax.
  - gelu_new, LN eps 1e-5, learned wpe, TIED unembedding (wte.T).
"""
from __future__ import annotations

import glob
import math
import os

import torch

# Local, gitignored cache populated by `huggingface-cli download` — see
# README.md "Reproduce" for the exact fetch command. Override with the
# LMCERT_HF_CACHE env var to point at an existing HF cache instead.
_HF_CACHE = os.environ.get(
    "LMCERT_HF_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hf_cache"),
)
_GPT2_BIN_CANDIDATES = glob.glob(os.path.join(
    _HF_CACHE, "models--gpt2", "snapshots", "*", "pytorch_model.bin"))
if not _GPT2_BIN_CANDIDATES:
    raise FileNotFoundError(
        f"GPT-2-small weights not found under {_HF_CACHE!r}. "
        "Fetch them first — see README.md 'Reproduce'.")
GPT2_BIN = _GPT2_BIN_CANDIDATES[0]
N_LAYERS = 12
N_HEADS = 12
D_MODEL = 768
D_HEAD = 64
LN_EPS = 1e-5
GELU_C = math.sqrt(2.0 / math.pi)

# "The quick brown fox jumps over the lazy" — ids verified against the repo's
# vocab.json: The=464, Ġquick=2068, Ġbrown=7586, Ġfox=21831, Ġjumps=18045,
# Ġover=625, Ġthe=262, Ġlazy=16931.  Expected continuation ' dog' (3290).
PROMPT_IDS = [464, 2068, 7586, 21831, 18045, 625, 262, 16931]


def load_sd() -> dict:
    return torch.load(GPT2_BIN, map_location="cpu", weights_only=True)


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(GELU_C * (x + 0.044715 * x**3)))


def layer_norm(x, g, b):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + LN_EPS) * g + b


def forward(sd: dict, ids: list[int], dtype=torch.float32) -> torch.Tensor:
    """Manual forward; returns logits [T, vocab]."""
    # skip only the causal-mask buffers h.L.attn.bias / h.L.attn.masked_bias
    # (".c_attn.bias" does NOT match ".attn.bias" — the underscore differs)
    W = {k: v.to(dtype) for k, v in sd.items()
         if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias")}
    T = len(ids)
    x = W["wte.weight"][ids] + W["wpe.weight"][:T]
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    for L in range(N_LAYERS):
        p = f"h.{L}."
        h = layer_norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        qkv = h @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]
        q, k, v = qkv.split(D_MODEL, dim=-1)
        q = q.view(T, N_HEADS, D_HEAD).transpose(0, 1)
        k = k.view(T, N_HEADS, D_HEAD).transpose(0, 1)
        v = v.view(T, N_HEADS, D_HEAD).transpose(0, 1)
        att = (q @ k.transpose(-1, -2)) / 8.0          # 1/sqrt(64) exact
        att = torch.where(mask, att, torch.tensor(float("-inf"), dtype=dtype))
        att = torch.softmax(att, dim=-1)
        o = (att @ v).transpose(0, 1).reshape(T, D_MODEL)
        o = o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        x = x + o
        h = layer_norm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        h = h @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"]
        h = gelu_new(h)
        h = h @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
        x = x + h
    x = layer_norm(x, W["ln_f.weight"], W["ln_f.bias"])
    return x @ W["wte.weight"].T


if __name__ == "__main__":
    sd = load_sd()
    l32 = forward(sd, PROMPT_IDS, torch.float32)[-1]
    l64 = forward(sd, PROMPT_IDS, torch.float64)[-1]
    top = torch.topk(l32, 5)
    print("prompt ids :", PROMPT_IDS)
    print("top-5 ids  :", top.indices.tolist())
    print("top-5 logit:", [round(v, 4) for v in top.values.tolist()])
    print("f32 vs f64 max |diff| (last pos):",
          (l32.double() - l64).abs().max().item())
    print("argmax f32:", int(l32.argmax()), " f64:", int(l64.argmax()))
    g = torch.topk(l64, 2).values
    print("top1-top2 gap (f64):", (g[0] - g[1]).item())
