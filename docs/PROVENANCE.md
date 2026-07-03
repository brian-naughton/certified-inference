# Provenance

Internal build record for this session's Task 1 (Phase 1) work. This file
tracks checkpoint/corpus hashes, environment pins, and pilot-run evidence —
not itself a public claim document.

## Checkpoints (sha256 of `pytorch_model.bin`)

| model | HF repo | sha256 |
|---|---|---|
| TinyStories-1M | `roneneldan/TinyStories-1M` | `07f9609ea882b8163ff3b23d40e2b82cb715d409631beb15c84b164f3877dae7` |
| GPT-2-small | `gpt2` | `7c5d3f4b8b76583b422fcb9189ad6c89d5d97a094541ce8932dce3ecabde1421` |

Fetched via `huggingface_hub.snapshot_download` (TinyStories-1M) and a
pattern-filtered `snapshot_download` for GPT-2 (pytorch_model.bin +
tokenizer/config only — an unfiltered `snapshot_download` pulls every stored
format — TF, safetensors, ONNX — and bloated the local cache to >5 GB on a
disk-constrained development machine; the filtered fetch is ~500 MB, matching
the actual model size).

## Weight hex export (Task 0.4)

`certinf.loader.export_weights(model_key, out_path)` sha-verifies the pinned
`pytorch_model.bin` above (re-hashing on every call — never trusts the cache
path alone), then writes every named weight tensor
(`certinf.loader.TS_TENSOR_NAMES` / `GPT2_TENSOR_NAMES` — exactly the tensor
names `certinf.interval_fwd.prepare_weights` / `certinf.gpt2_interval`
consume) as nested `float(v).hex()` strings, torch-free downstream.

**Exact-real caveat**: every stored weight is float32 (<=24-bit mantissa);
Python's `float` is float64 (53-bit mantissa), so the float32->float64
widening is exact (no rounding), and `float.hex()`/`float.fromhex()` round
trip that float64 value bit-for-bit. `certinf.interval_fwd` and
`certinf.gpt2_interval` already treat every weight as an exact dyadic
rational (`Fraction(v)`, or a frexp-based common-denominator conversion) —
this export introduces no additional rounding beyond that pre-existing
float32-as-exact-real assumption. Redundant trailing mantissa zeros are
stripped from each hex literal (`0x1.921fb60000000p+1` -> `0x1.921fb6p+1`;
float32-origin values carry at most 6 significant hex digits) — a ~30% size
cut with `float.fromhex` still parsing back to the identical float64,
asserted per element by `tests/test_loader.py`.

**Artifact policy (measured sizes, 2026-07-03):**

| artifact | size | policy |
|---|---|---|
| `certificates/tinystories-1M.weights.json` | ~64 MB | committed |
| `certificates/gpt2-small.weights.json` | ~2.1 GB | gitignored — regenerate on demand |

The task plan estimated "TS ~a few MB; GPT-2 ~tens of MB"; the real sizes
are ~30x / ~100x that (124.4M weights x >=16 bytes/element is a hard floor
for hex text). The GPT-2 export exceeds GitHub's 100 MB per-file hard limit
~20x and (during this task) filled the development disk mid-write, so it is
NOT committed. Its sha-pinned provenance keeps it fully deterministic:
regenerating from the pinned checkpoint reproduces the identical file.

Regenerate either artifact with:

```bash
python3.11 -c "from certinf.loader import export_weights as e; \
e('tinystories-1M', 'certificates/tinystories-1M.weights.json')"
# GPT-2 (~2.1 GB on disk — check free space first):
python3.11 -c "from certinf.loader import export_weights as e; \
e('gpt2-small', 'certificates/gpt2-small.weights.json')"
```

## Corpus

`certificates/corpora/tinystories-val.ids.json`: 200 non-overlapping windows
each at context lengths {8, 16, 32}, built by `certinf.corpus.build_tinystories`
from the `roneneldan/TinyStories` validation split (licence `cdla-sharing-1.0`,
verified via `HfApi.dataset_info`), tokenised with the TinyStories-1M
checkpoint's own tokenizer. Fetched as the single validation parquet file
directly via `hf_hub_download` rather than through the `datasets` library's
full builder pipeline — the latter regenerates a multi-GB train+validation
Arrow cache (2.1M examples) and filled the development disk to 0 bytes free
on first attempt; the parquet-only path is ~10 MB.
`corpus_sha256 = de3579e9051d85980ff6154b1be980fc5bd3f7a433945a61c53c06313c6b42bd`.

`tests/fixtures/tinystories-dev.ids.json`: a small, explicitly non-production
dev fixture assembled from the twice-adversarially-reviewed foothold prompt
set (`engine-seed/sweep.py`), used by Task 1's own unit tests, which need a
known-certifiable baseline at a fixed index. Index 0 = the foothold seq-8
"Once upon a time there was a little" prompt
(`[7454, 2402, 257, 640, 612, 373, 257, 1310]`).

**Not built in this pass:** `build_wikitext103` (the GPT-2 confirmation
corpus, Task 0.6 proper) — GPT-2 is confirmation-only per the design spec
(A5) and no Task 1 test requires it.

## Environment (A1 transcript, `certinf.harness.transcript()`)

- Python 3.11.14 (`python3.11`, Homebrew)
- PyTorch 2.11.0, CPU only
- `torch.backends.cuda.matmul.allow_tf32 = False`,
  `torch.use_deterministic_algorithms(True, warn_only=True)`, `eval()`/`no_grad()`
  mode, float32 throughout
- macOS (Darwin), arm64

## φ₂ pilot (Task 1.2, Step 5)

20 samples, `certify_sample(..., run_harness=True)`, TinyStories-1M,
`certificates/corpora/tinystories-val.ids.json` prompt indices 0–19 (context
length 8), `P_grid=[128,160,192,224,256]`, `P_max=256`. Widths canary asserted
once (index 0 only, for pilot speed — already exercised independently by
`tests/test_canary.py`).

- 20/20 CERTIFIED (`phi1=True` for every sample)
- Escalation: every sample abstained (width) at P=128 and certified at
  P=160 — `escalation_trace == [128, 160]` uniformly across this 20-sample
  slice
- `phi2_joint`: **20/20 True** — the pinned float32 harness's top-1 agreed
  with the certified exact-real top-1 on every certified sample
- `harness.determinism_check("tinystories", MODEL_BIN, <foothold prompt>,
  repeats=5)` → **True** — CPU float32 is deterministic in this pinned
  environment; **no A1 CI-D downgrade triggered**
- Wall time: ~180 s / 20 samples ≈ 9 s/sample (canary only on sample 0)

This is a pilot, not the headline run (A2: calibration/pilot samples are
never headline samples; no `prereg_ref` was set for any of these). It
confirms the harness+certifier wiring end-to-end and that `phi2_joint` fills
correctly before Task 1.3's calibration grid and Phase 2's pre-registered
headline runs.
