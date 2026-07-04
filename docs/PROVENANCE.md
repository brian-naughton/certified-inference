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

`certificates/corpora/wikitext103-test.ids.json` (Task 0.6): 200
non-overlapping windows each at context lengths {8, 16, 32}, built by
`certinf.corpus.build_wikitext103` from the `wikitext` dataset's
`wikitext-103-raw-v1` **test** split (GPT-2's confirmation-only corpus per
spec A5, WikiText-103 chosen over OpenWebText per amendment A4 because
OpenWebText's licence is ambiguous — see "Licensing gate (A4)" below),
tokenised with GPT-2's own tokenizer. Fetched as the single test-split
parquet file directly via `hf_hub_download`
(`wikitext-103-raw-v1/test-00000-of-00001.parquet`, 733 KB) rather than
through the `datasets` library's full builder pipeline, which would also
pull the ~300 MB train split — same disk-budget rationale as the TinyStories
corpus above (disk had 5 GB free at build time). Windows respect `text`-row
boundaries (never straddle a row): `wikitext-103-raw-v1` rows are single
lines — paragraphs, blank separators, or `= = Heading = = ` markers, not
whole articles — so this mirrors `build_tinystories`'s per-unit windowing
discipline rather than a single global concatenated stream.
`corpus_sha256 = ff5d4fe62f865b34aa2ce53a79a0123ecbfe90f37d1d27dcc10a1717a0c1d374`
(reproduced bit-for-bit on a second build from the pinned parquet + `gpt2`
tokenizer — see `tests/test_corpus.py::test_pinned_wikitext103_corpus_matches_committed_sha`).
`tokenizer_sha` (sha256 of GPT-2's `vocab.json`) =
`196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783`.

### Coverage note (WikiText-103 corpus — added post-review, 2026-07-03)
The pinned windows are drawn deterministically from the START of the test split (~0.55% of its non-blank rows — effectively the first few contiguous articles), with strong inter-window correlation, inheriting the TinyStories corpus convention. This is statistically sound for our population claims, which quantify ONLY over the pinned finite corpus C (Hoeffding requires i.i.d. draws from C, nothing more) — but do not read "WikiText-103 test" as a broad or representative slice of that dataset. At ctx=8, 7/200 windows are heading-only markup rows (raw WikiText texture, expected). Follow-ups tracked: automated rebuild-determinism test; HF revision pin.

## Licensing gate (A4 HARD gate)

Per corpus: source, licence (verbatim from the HF dataset card), what is
committed, and why that is compliant. The committed artifact for both
corpora is **token IDs only** (JSON integers) — never raw text, never the
tokenizer's vocab/merge files.

### TinyStories (validation split)

- **Source**: `roneneldan/TinyStories`, HF dataset, `validation` split.
- **Licence** (verified via `HfApi.dataset_info("roneneldan/TinyStories").cardData`,
  Task 1): `cdla-sharing-1.0` (Community Data License Agreement – Sharing,
  Version 1.0) — a permissive data-sharing licence that explicitly permits
  redistributing the dataset and derivative/adapted data, including in
  modified form, provided downstream recipients receive the data under the
  same licence terms.
- **Committed**: `certificates/corpora/tinystories-val.ids.json` — GPT-2
  byte-level-BPE token-id windows only.
- **Why compliant**: CDLA-Sharing-1.0 defines "Results" (§1.11) as
  computational outputs carrying "no more than a *de minimis* portion of
  the Data" (§3.5, unrestricted) — a full GPT-2-BPE token-id window is a
  lossless, reversible re-encoding of the source text, so it does **not**
  qualify as de minimis Results; it is treated conservatively as
  "Enhanced Data" (modified/derived Data) instead. §2.1 grants the right
  to Publish Data, including in modified form; §3.3 permits publishing
  Enhanced Data provided it stays "Published under this Agreement" (same
  licence downstream — hence `meta.license = "cdla-sharing-1.0"` on the
  committed file itself) with "no further restrictions" added; §3.1
  requires preserving attribution/notice to the Data Provider, which this
  document (and the corpus file's `meta.source`/`meta.hf_repo`) supplies.
  All three conditions are met — TinyStories redistribution is compliant.

### WikiText-103 (GPT-2 confirmation corpus, test split)

- **Source**: `wikitext` HF dataset, config `wikitext-103-raw-v1`, `test`
  split. Chosen over OpenWebText (the GPT-2 training-data analogue) per
  spec amendment A4, because OpenWebText's own licence status is
  ambiguous (it is a third-party Reddit-URL scrape with no clear licence
  grant from the underlying page owners); WikiText-103 is redistributable
  and is the standard confirmation/held-out corpus for GPT-2-family
  evaluation in the literature.
- **Licence** (verified by fetching the dataset card directly — both the
  YAML front-matter and prose body, `data/hf_cache/datasets--wikitext/.../README.md`,
  fetched via `huggingface_hub.hf_hub_download("wikitext", "README.md",
  repo_type="dataset")`): the card is **internally inconsistent**.
  - YAML `license:` tags (machine-readable metadata, `HfApi.dataset_info`
    `cardData`): `cc-by-sa-3.0`, `gfdl`.
  - Prose "Licensing Information" section (verbatim): *"The dataset is
    available under the [Creative Commons Attribution-ShareAlike License
    (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)."*
  - Both readings are recorded verbatim in the corpus file's
    `meta.license` field rather than picked arbitrarily. The discrepancy
    does not change the compliance conclusion below: both CC BY-SA 3.0
    and CC BY-SA 4.0 are attribution + share-alike licences that permit
    redistributing derivative works, provided attribution is preserved
    and the derivative carries a compatible share-alike licence forward.
    (The `gfdl` YAML co-tag — GNU Free Documentation License, the
    dual-licence Wikipedia itself uses for older content — was not
    independently re-verified for this compliance conclusion; it is not
    relied on, since the CC BY-SA reading alone is sufficient regardless
    of which CC BY-SA version controls.) This is not an A4 HARD-gate stop
    condition; the owner decision gate was not triggered.
- **Committed**: `certificates/corpora/wikitext103-test.ids.json` — GPT-2
  byte-level-BPE token-id windows only.
- **Why compliant**: same reasoning as TinyStories — the committed
  artifact is integer token ids, a computed derivative of the licensed
  text, not the text itself; this document (and the corpus file's own
  `meta.license` field) carries attribution back to the `wikitext` dataset
  and its CC BY-SA licence, and the corpus file's licence itself doubles
  as the share-alike notice for anyone redistributing it further. No
  verbatim article text, titles, or the tokenizer's vocabulary file are
  committed.

Build/regenerate either corpus with:

```bash
python3.11 -c "from certinf.corpus import build_tinystories as b; \
b(out_path='certificates/corpora/tinystories-val.ids.json')"
python3.11 -c "from certinf.corpus import build_wikitext103 as b; \
b(out_path='certificates/corpora/wikitext103-test.ids.json')"
```

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

## Sampling bridge — AUDITED, not kernel-checked (Hoeffding wrapper hypotheses)
The Lean theorem `hoeffding_lower_confidence` assumes: (1) mutual independence (`iIndepFun`) — discharged because the frozen design draws indices i.i.d. uniformly WITH REPLACEMENT from the pinned corpus C (without replacement would be hypergeometric-dependent and break this hypothesis; the freeze forbids deduplication); (2) common mean μ[Bᵢ] = p — discharged because each Bᵢ is a deterministic {0,1} predicate of the drawn index, so its expectation under a uniform draw equals the population φ-rate on C, identical for every draw; (3) values in [0,1] and measurability — trivial for indicator functions on a finite space. These discharges are audited provenance (this document + the committed prereg artifacts + the seeded re-draw witness), not formalised in Lean; the kernel-checked object is the statistical inequality itself.

## GPT-2 confirmation set (Task 2.3)

**Role: prestige confirmation, NOT a population claim.** Per the
2026-07-03 whole-corpus evaluation (Q3 + SHOULD #4): "Give GPT-2 a cleaner
role: one or a few full-vocab per-sample certificates plus an honest note
that the checker path is not yet GPT-2-complete. Do not let GPT-2 drive the
statistical headline." This supersedes the earlier task brief's `n≈30`
figure; `n=8` full-vocabulary certificates were generated instead — still
hours of compute, but explicitly scoped as confirmation/scaling evidence,
never a Hoeffding-bound population claim.

- **What was certified**: `certinf.certify.certify_sample("gpt2", ...)` —
  the SAME hardened full-vocabulary dispatch path validated by
  `tests/test_certify.py` (commit `548e42e`'s `require_full=True` guard:
  the certified competitor set is forced to the full 50,257-token
  vocabulary and the call raises rather than silently downgrading to
  top-200) — over the first `n=8` windows (`prompt_index=0..7`, a
  deterministic, non-cherry-picked prefix slice, never a random draw) of the
  pinned WikiText-103-test corpus (`certificates/corpora/wikitext103-test.ids.json`,
  `corpus_sha256=ff5d4fe62f865b34aa2ce53a79a0123ecbfe90f37d1d27dcc10a1717a0c1d374`)
  at context length 16. Escalation ladder `P_grid=[320,384]`, `P_max=448`.
  The widths canary was asserted ONCE PER RUN (window 0, P=320 vs 2P=640 —
  `certinf/grid.py`'s documented per-cell granularity; a per-sample canary
  would add two extra GPT-2 forwards, one at 640 bits, to every ~5-minute
  sample and several-fold multiply the run's cost).
- **Every record carries `prereg_ref=None`.** There is no per-record
  "label" field in the certificate schema (`schema.validate_record` rejects
  unrecognised top-level keys), so the "confirmation set — no population
  claim" designation is carried in the run's own
  `certificates/gpt2-confirmation/gpt2-confirmation.cert.meta.json` (`note`
  field) and here, not inside each JSON record.
- **Checker status (honest, not glossed over)**: `certificates/check.py`
  independently re-derives records torch-free from the sha-pinned hex weight
  export, but its weight preparation and residual-stream wiring are
  TinyStories-shaped; it does not yet have a GPT-2 code path (the GPT-2 hex
  export itself, `certificates/gpt2-small.weights.json`, is ~2.1 GB and
  gitignored — see the Task 0.4 section above). These GPT-2 records are
  therefore full-vocabulary exact-real certificates produced by the SAME
  generator instrument (`certinf.exact` / `certinf.ival_ext` / the widths
  canary) validated extensively elsewhere in this project, but they are
  **NOT yet independently re-derived by a second, torch-free instrument**.
  Closing that gap (a GPT-2 path in `check.py`) is future work, tracked
  alongside the trilogy's other open items.
- **Results (run completed 2026-07-04, log
  `docs/logs/gpt2-confirmation.log`)**: **8/8 CERTIFIED, 8/8 φ₂_joint**
  (pinned float32 harness top-1 agreed with the certified exact-real top-1
  on every sample), zero abstentions. Every sample certified at the FIRST
  ladder rung P=320 (`escalation_trace=[320]` uniformly — neither 384 nor
  the 448 cap was ever needed). Exact-Fraction `margin_lo` range
  **0.001064 – 1.809204** (the 0.001064 sample, index 5, still clears its
  max logit-interval width 2.37e-07 by ~12.1 bits); per-sample headroom
  (`log2(margin_lo / logit_width_max)`) 12.1–27.6 bits. Runtime: widths
  canary 1647 s (the 2P=640-bit forward dominates), then 8 samples in
  4147 s (per-sample 432–878 s; first sample slower — cold torch load);
  total wall ~5794 s ≈ 97 min, peak RSS ~2.4 GB, jobs=1. The single
  prestige certificate `gpt2-small-fullvocab.cert.json` is byte-identical
  to record 0 of the JSONL. Records spot-validated post-run:
  schema-valid, `prompt_index` 0–7 with `token_ids` bit-identical to the
  pinned corpus windows, checkpoint/corpus sha256 match the pins above,
  `prereg_ref=null` throughout. Note the full-vocabulary competitor set is
  not a per-record schema field: `require_full` enforcement is the tested
  code path (`certinf/certify.py` GPT-2 dispatch + commit `548e42e`'s
  raise-not-degrade guard + `tests/test_certify.py`), under which a
  partial competitor set aborts rather than certifies.
