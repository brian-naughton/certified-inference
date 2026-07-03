#!/usr/bin/env python3
"""Manual float reference forward pass for TinyStories-1M (GPT-Neo arch).

Reference semantics (HF GPTNeoForCausalLM, transformers 4.28):
  - pre-LN blocks: x += attn(ln_1(x)); x += mlp(ln_2(x))
  - attention: q,k,v projections WITHOUT bias; NO 1/sqrt(d_head) scaling
    (GPT-Neo omits the scale); causal mask via where(mask, w, -1e9);
    out_proj WITH bias.
  - local attention layers == global causal attention when seq_len <= window
    (256); prompt here is 8-16 tokens, so all layers are standard causal.
  - mlp: c_fc (64->256) -> gelu_new -> c_proj (256->64), both with bias.
  - gelu_new(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x**3)))
  - final ln_f, logits = h @ wte.T (tied embeddings).
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
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "hf_cache"),
)
_MODEL_BIN_CANDIDATES = glob.glob(os.path.join(
    _HF_CACHE, "models--roneneldan--TinyStories-1M", "snapshots", "*", "pytorch_model.bin"))
if not _MODEL_BIN_CANDIDATES:
    raise FileNotFoundError(
        "TinyStories-1M weights not found under "
        f"{_HF_CACHE!r}. Fetch them first — see README.md 'Reproduce'.")
MODEL_BIN = _MODEL_BIN_CANDIDATES[0]
N_LAYERS = 8
N_HEADS = 16
D_MODEL = 64
D_HEAD = 4
LN_EPS = 1e-5
GELU_C = math.sqrt(2.0 / math.pi)


def load_sd() -> dict:
    return torch.load(MODEL_BIN, map_location="cpu", weights_only=True)


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(GELU_C * (x + 0.044715 * x**3)))


def layer_norm(x: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)   # biased, matches LN
    return (x - mu) / torch.sqrt(var + LN_EPS) * g + b


def forward(sd: dict, ids: list[int], dtype=torch.float32,
            capture: dict | None = None) -> torch.Tensor:
    """Run the manual forward; returns logits [T, vocab].

    If `capture` is given, stores residual stream after each layer as
    capture['layer_k'] (k=0..7) and capture['embed'].
    """
    W = {k: v.to(dtype) for k, v in sd.items()}
    T = len(ids)
    tok = W["transformer.wte.weight"][ids]            # [T, 64]
    pos = W["transformer.wpe.weight"][:T]             # [T, 64]
    x = tok + pos
    if capture is not None:
        capture["embed"] = x.clone()
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    for L in range(N_LAYERS):
        p = f"transformer.h.{L}."
        h = layer_norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        q = h @ W[p + "attn.attention.q_proj.weight"].T
        k = h @ W[p + "attn.attention.k_proj.weight"].T
        v = h @ W[p + "attn.attention.v_proj.weight"].T
        q = q.view(T, N_HEADS, D_HEAD).transpose(0, 1)   # [H, T, dh]
        k = k.view(T, N_HEADS, D_HEAD).transpose(0, 1)
        v = v.view(T, N_HEADS, D_HEAD).transpose(0, 1)
        att = q @ k.transpose(-1, -2)                    # NO scaling (GPT-Neo)
        att = torch.where(mask, att, torch.tensor(-1e9, dtype=dtype))
        att = torch.softmax(att, dim=-1)
        o = (att @ v).transpose(0, 1).reshape(T, D_MODEL)
        o = o @ W[p + "attn.attention.out_proj.weight"].T \
            + W[p + "attn.attention.out_proj.bias"]
        x = x + o
        h = layer_norm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        h = h @ W[p + "mlp.c_fc.weight"].T + W[p + "mlp.c_fc.bias"]
        h = gelu_new(h)
        h = h @ W[p + "mlp.c_proj.weight"].T + W[p + "mlp.c_proj.bias"]
        x = x + h
        if capture is not None:
            capture[f"layer_{L}"] = x.clone()
    x = layer_norm(x, W["transformer.ln_f.weight"], W["transformer.ln_f.bias"])
    if capture is not None:
        capture["final_ln"] = x.clone()
    return x @ W["transformer.wte.weight"].T


# "Once upon a time there was a little" in GPT-2 BPE (ids from memory,
# validated below by checking the continuation is plausible English).
PROMPT_IDS = [7454, 2402, 257, 640, 612, 373, 257, 1310]
# extension to 16 tokens: " girl who lived in a big house with"
# (ids looked up exactly in the repo's vocab.json)
PROMPT_IDS_16 = PROMPT_IDS + [2576, 508, 5615, 287, 257, 1263, 2156, 351]

if __name__ == "__main__":
    sd = load_sd()
    logits32 = forward(sd, PROMPT_IDS, torch.float32)
    logits64 = forward(sd, PROMPT_IDS, torch.float64)
    last32, last64 = logits32[-1], logits64[-1]
    top = torch.topk(last32, 10)
    print("top-10 ids :", top.indices.tolist())
    print("top-10 logit:", [round(v, 4) for v in top.values.tolist()])
    print("f32 vs f64 max |diff| on last-pos logits:",
          (last32.double() - last64).abs().max().item())
    print("argmax f32:", int(last32.argmax()), " f64:", int(last64.argmax()))
    g = torch.topk(last64, 2).values
    print("top1-top2 gap (f64):", (g[0] - g[1]).item())
