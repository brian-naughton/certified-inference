# Pre-registration dry run + frozen-condition determinism evidence

This document is a **dry run**: an end-to-end exercise of the pre-registration,
certification, and independent-checking pipeline on a small subset of the
pinned TinyStories validation corpus. It demonstrates that the machinery works —
including the failure paths that must reject a tampered or incomplete
certificate — and records a determinism check of the pinned inference harness at
the frozen conditions the headline run will use.

It is **not** a headline population claim. The headline freeze (a full-size
pinned corpus, committed before certification begins) is a separate, later
artifact. The freeze artifacts used here live under
`certificates/prereg/dryrun/` and are labelled `DRY RUN NOT HEADLINE`.

All commands were run with Python 3.11 / PyTorch 2.11.0 on macOS
(arm64). The δ budget is carried as exact rational arithmetic throughout —
`δ = 1/20`, split `1/40 + 1/40`, with an exact sum-check (no floating-point
tolerance anywhere in the certification path).

---

## 1. Freeze

A 12-window subset (context length 16) of the pinned TinyStories validation
corpus is written as token-ids, and a pre-registration is frozen over it. The
sample indices are drawn with replacement from a committed seed; the drawn
multiset contains duplicates, which are kept and counted with multiplicity (no
deduplication). The freeze also records the exact Hoeffding inputs `(n, δ)` and
a conservatively rounded ε display — ε is rounded **up**, so any displayed
lower bound `k/n − ε` rounds **down** and can never overstate the certified
rate.

```
corpus_sha256: a64cbe64269d4620e8a9313f369634e84b6809037cfe96273a3a419891dea1ad
prereg_format_version: 2
prereg_sha256: 3fa32a573203654e77646430888d52e61d3ef706f638e2c8d9c72e4cefaeb396
delta: ['1', '20'] delta_split: {'phi1': ['1', '40'], 'phi2_joint': ['1', '40']}
hoeffding: {"formula": "epsilon = sqrt(ln(1/delta) / (2*n))", "n": 12, "delta": ["1", "20"], "epsilon_display": "0.353302", "epsilon_display_dp": 6, "epsilon_display_rounding": "up", "display_note": "epsilon is rounded UP so any displayed population lower bound k/n - epsilon rounds DOWN — the display never overstates the certified rate. The exact record is (n, delta); the string is advisory."}
frozen indices: [1, 10, 9, 3, 5, 5, 7, 9, 1, 0, 10, 5]
multiset: {0: 1, 1: 2, 3: 1, 5: 3, 7: 1, 9: 2, 10: 2} -> duplicates at [1, 5, 9, 10]
freeze witness verify(): True
```

The frozen index multiset contains four duplicated indices (1, 5, 9, 10) —
index 5 is drawn three times — so this dry run exercises the with-replacement
multiplicity accounting directly.

The written pre-registration (`certificates/prereg/dryrun/prereg.json`) carries
the δ budget as exact `["numerator", "denominator"]` pairs:

```json
 "delta": ["1", "20"],
 "delta_split": {
  "phi1": ["1", "40"],
  "phi2_joint": ["1", "40"]
 },
 "hoeffding": {
  "formula": "epsilon = sqrt(ln(1/delta) / (2*n))",
  "n": 12,
  "delta": ["1", "20"],
  "epsilon_display": "0.353302",
  "epsilon_display_dp": 6,
  "epsilon_display_rounding": "up"
 }
```

## 2. Certify

Each drawn index is certified with real interval forward passes, escalating
precision over the ladder `P ∈ [128, 160, 192]` up to `P_max = 256`. Every
record carries this freeze's `prereg_sha256` as its `prereg_ref`. Duplicated
indices are certified with multiplicity.

```
  [ 1/12] prompt_index= 1 status=CERTIFIED P=160 27.3s
  [ 2/12] prompt_index=10 status=CERTIFIED P=160 28.0s
  [ 3/12] prompt_index= 9 status=CERTIFIED P=160 26.6s
  [ 4/12] prompt_index= 3 status=CERTIFIED P=160 26.7s
  [ 5/12] prompt_index= 5 status=CERTIFIED P=160 26.5s
  [ 6/12] prompt_index= 5 status=CERTIFIED P=160 27.1s
  [ 7/12] prompt_index= 7 status=CERTIFIED P=160 27.4s
  [ 8/12] prompt_index= 9 status=CERTIFIED P=160 23.9s
  [ 9/12] prompt_index= 1 status=CERTIFIED P=160 23.6s
  [10/12] prompt_index= 0 status=CERTIFIED P=160 22.6s
  [11/12] prompt_index=10 status=CERTIFIED P=160 22.5s
  [12/12] prompt_index= 5 status=CERTIFIED P=160 22.2s
wrote certificates/prereg/dryrun/records.jsonl (12 records)
```

All twelve samples certify (each at precision `P = 160`).

## 3. Independent check

The independent, torch-free checker re-derives every record from the pinned hex
weight export and the committed corpus, verifies the freeze witness, and — for a
headline (pre-registered) certificate — asserts that the multiset of covered
prompt indices exactly matches the frozen sample index (duplicates counted with
multiplicity).

### 3.1 Success: verified with completeness

```
$ python3.11 certificates/check.py --weights certificates/tinystories-1M.weights.json --corpus certificates/prereg/dryrun/tinystories-dryrun.ids.json --cert certificates/prereg/dryrun/records.jsonl --prereg certificates/prereg/dryrun --jobs 4
VERIFIED (12 records re-derived from hex weights, all records, headline)
[exit code: 0]
```

### 3.2 Failure: a dropped record

Removing a single record (here, one of the three copies of index 5) fails the
completeness assertion. Because indices are counted with multiplicity, dropping
one copy of a triplicated index still registers as a missing `5`:

```
$ python3.11 certificates/check.py --weights certificates/tinystories-1M.weights.json --corpus certificates/prereg/dryrun/tinystories-dryrun.ids.json --cert /tmp/dryrun_dropped.jsonl --prereg certificates/prereg/dryrun --jobs 4
FAILED: cert does not cover the pre-registered index set (missing [5], unexpected []; 11 records vs 12 frozen)
[exit code: 1]
```

### 3.3 Failure: an extra (over-covering) record

Adding a record beyond the frozen set — even a genuine, re-derivable one — fails:
the covered multiset no longer equals the frozen multiset.

```
$ python3.11 certificates/check.py --weights certificates/tinystories-1M.weights.json --corpus certificates/prereg/dryrun/tinystories-dryrun.ids.json --cert /tmp/dryrun_extra.jsonl --prereg certificates/prereg/dryrun --jobs 4
FAILED: cert does not cover the pre-registered index set (missing [], unexpected [0]; 13 records vs 12 frozen)
[exit code: 1]
```

### 3.4 Failure: mixed pre-registration references

A certificate must be entirely headline (every record carries a
pre-registration reference) or entirely calibration (none do) — never mixed.
Nulling one record's reference is rejected:

```
$ python3.11 certificates/check.py --weights certificates/tinystories-1M.weights.json --corpus certificates/prereg/dryrun/tinystories-dryrun.ids.json --cert /tmp/dryrun_mixed.jsonl --prereg certificates/prereg/dryrun --jobs 4
FAILED: mixed prereg_ref: a cert must be all-headline (non-null) or all-calibration (null) — never mixed (A2)
[exit code: 1]
```

---

## 4. Harness determinism at frozen conditions

The deployment-gap property depends on the pinned float32 inference harness
being deterministic in the pinned environment. If it were not, the joint
deployment-gap claim would have to downgrade to an implementation-specific
observation. This section re-runs the determinism gate at the frozen context
length (16), with the exact command shape the headline harness invocation
takes, over five repetitions on seven distinct prompts from the pinned corpus.

### 4.1 Runnable CLI invocation (authoritative)

The determinism gate is a genuinely-runnable CLI (`certinf.harness` has a real
`argparse __main__`). The command below was executed verbatim against the pinned
headline corpus `certificates/corpora/tinystories-val.ids.json` at context
length 16 (its first 12 ctx-16 windows are bit-identical to the dry-run subset,
so prompts `[0, 1, 3, 5, 7, 9, 10]` are the same token windows as before):

```
$ BIN=data/hf_cache/models--roneneldan--TinyStories-1M/snapshots/77f1b168e219585646439073245fe87e56b3023e/pytorch_model.bin
$ python3.11 -m certinf.harness --model tinystories --weights "$BIN" \
    --corpus certificates/corpora/tinystories-val.ids.json --context-length 16 \
    --prompt-index 0 --prompt-index 1 --prompt-index 3 --prompt-index 5 \
    --prompt-index 7 --prompt-index 9 --prompt-index 10 --reps 5
### implementation transcript
{
 "checkpoint_sha256": "07f9609ea882b8163ff3b23d40e2b82cb715d409631beb15c84b164f3877dae7",
 "command_line": "python3.11 -m certinf.harness --model tinystories --weights pytorch_model.bin --corpus tinystories-val.ids.json --context-length 16",
 "deterministic_flags": {
  "cuda_matmul_allow_tf32": false,
  "use_deterministic_algorithms": true
 },
 "eval_mode": true,
 "os": "Darwin 22.6.0",
 "platform": "macOS-13.7.8-arm64-arm-64bit",
 "python": "3.11.14 (main, Oct 12 2025, 19:18:13) [Clang 14.0.3 (clang-1403.0.22.14.1)]",
 "tf32": false,
 "torch_version": "2.11.0"
}
transcript_sha256: b4e2fa62eb3a242a4993ede2fde10be2d39b1aa166da06ebeb6d790f085f0358

=== determinism gate: 5 repetitions x 7 distinct prompts (ctx=16) ===
  prompt_index= 0  top1 x5 = [1097, 1097, 1097, 1097, 1097]  -> DETERMINISTIC
  prompt_index= 1  top1 x5 = [11254, 11254, 11254, 11254, 11254]  -> DETERMINISTIC
  prompt_index= 3  top1 x5 = [2227, 2227, 2227, 2227, 2227]  -> DETERMINISTIC
  prompt_index= 5  top1 x5 = [27498, 27498, 27498, 27498, 27498]  -> DETERMINISTIC
  prompt_index= 7  top1 x5 = [1239, 1239, 1239, 1239, 1239]  -> DETERMINISTIC
  prompt_index= 9  top1 x5 = [340, 340, 340, 340, 340]  -> DETERMINISTIC
  prompt_index=10  top1 x5 = [13, 13, 13, 13, 13]  -> DETERMINISTIC

VERDICT: DETERMINISTIC across all prompts (A1 CI-D gate PASSES)
[exit code: 0]
```

The `command_line` embedded in the transcript is canonicalised to basenames
(environment-independent, so the transcript sha does not depend on absolute
cache/checkout paths); the shell line above is the concrete, runnable form. The
recomputed `transcript_sha256` is
`b4e2fa62eb3a242a4993ede2fde10be2d39b1aa166da06ebeb6d790f085f0358` and the
per-prompt top-1 tokens are bit-identical to the superseded hand-run below —
independent confirmation that the pinned harness is deterministic at the frozen
conditions and reproducible from the committed CLI.

### 4.2 Superseded — original hand-run transcript (pre-CLI)

*The entry below is retained for history and is **superseded** by §4.1. It was
produced by an ad-hoc harness call before `certinf.harness` had a runnable
`__main__`, so its recorded `command_line` was not directly executable at the
time. §4.1 re-records the identical transcript sha and results under the genuine
CLI.*

```
{
 "checkpoint_sha256": "07f9609ea882b8163ff3b23d40e2b82cb715d409631beb15c84b164f3877dae7",
 "command_line": "python3.11 -m certinf.harness --model tinystories --weights pytorch_model.bin --corpus tinystories-val.ids.json --context-length 16",
 "deterministic_flags": {
  "cuda_matmul_allow_tf32": false,
  "use_deterministic_algorithms": true
 },
 "eval_mode": true,
 "os": "Darwin 22.6.0",
 "platform": "macOS-13.7.8-arm64-arm-64bit",
 "python": "3.11.14 (main, Oct 12 2025, 19:18:13) [Clang 14.0.3 (clang-1403.0.22.14.1)]",
 "tf32": false,
 "torch_version": "2.11.0"
}
transcript_sha256: b4e2fa62eb3a242a4993ede2fde10be2d39b1aa166da06ebeb6d790f085f0358
```

```
=== determinism gate: 5 repetitions x 7 distinct prompts (ctx=16) ===
  prompt_index= 0  top1 x5 = [1097, 1097, 1097, 1097, 1097]  -> DETERMINISTIC
  prompt_index= 1  top1 x5 = [11254, 11254, 11254, 11254, 11254]  -> DETERMINISTIC
  prompt_index= 3  top1 x5 = [2227, 2227, 2227, 2227, 2227]  -> DETERMINISTIC
  prompt_index= 5  top1 x5 = [27498, 27498, 27498, 27498, 27498]  -> DETERMINISTIC
  prompt_index= 7  top1 x5 = [1239, 1239, 1239, 1239, 1239]  -> DETERMINISTIC
  prompt_index= 9  top1 x5 = [340, 340, 340, 340, 340]  -> DETERMINISTIC
  prompt_index=10  top1 x5 = [13, 13, 13, 13, 13]  -> DETERMINISTIC

VERDICT: DETERMINISTIC across all prompts (A1 CI-D gate PASSES)
```

**Determinism verdict: deterministic.** Every prompt produced an identical
top-1 token across all five repetitions in the pinned environment. No
nondeterminism was observed, so no downgrade of the deployment-gap property is
triggered.

---

## Summary

| Check | Result |
| --- | --- |
| Freeze witness (re-draw + exact δ budget) | verified |
| Certify 12 pre-registered indices (real interval runs) | 12/12 certified |
| Independent re-derivation + completeness | VERIFIED |
| Dropped record | rejected |
| Extra record | rejected |
| Mixed pre-registration references | rejected |
| Harness determinism (5 × 7 prompts, ctx 16) | deterministic |
