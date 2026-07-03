# Results detail

This is the technical backing for the claims in `README.md`: setup, the per-layer width measurements, the engine bugs found and fixed along the way, and the two experiments (the 28-prompt sweep and the GPT-2-small prediction test). It replaces the original lab notebook, which mixed this content with day-to-day planning notes not relevant to a reader checking the results.

## Setup

- **TinyStories-1M** (`roneneldan/TinyStories-1M`): a real, pretrained GPT-Neo language model — 8 layers, d_model 64, 16 heads (d_head 4), MLP width 256, pre-LayerNorm (eps 1e-5), `gelu_new` (tanh approximation), learned positional embeddings, final LayerNorm, tied unembedding, vocabulary 50,257. GPT-Neo-specific details handled: q/k/v projections have no bias, out_proj has bias, attention has **no** `1/sqrt(d_head)` scaling. At the sequence lengths used here (8–16 tokens), GPT-Neo's alternating local/global attention pattern is equivalent to standard causal attention (`window_size` 256 in the model config), verified directly from `config.json`.
- **GPT-2-small** (`gpt2`, 124M parameters): 12 layers, d_model 768, 12 heads (d_head 64), pre-LayerNorm (eps 1e-5), `gelu_new`, learned positional embeddings, tied unembedding, vocabulary 50,257. Architecture deltas from the GPT-Neo harness: `Conv1D` weight layout (`(in, out)`, so `y = x @ W + b`, not `nn.Linear`'s transposed convention), fused qkv projection, and an exact `1/8` attention scale (`1/sqrt(64)`, a rational number, unlike TinyStories' irrational `1/sqrt(32)`).
- **Prompts.** The primary TinyStories prompt is "Once upon a time there was a little" (8 tokens) extended with " girl who lived in a big house with" (16 tokens); ids confirmed against the model's own `vocab.json`. The 28-prompt sweep (below) covers story openers, mid-sentence fragments, degenerate single-token repeats, rare/out-of-distribution token sequences, and assorted stress shapes. The GPT-2-small prompt is "The quick brown fox jumps over the lazy" (8 tokens). Every float forward pass was validated against a from-scratch, dependency-free reference implementation (`float_fwd.py`, `gpt2_float.py`) before any certified run.
- **Engine.** The certified-grokking interval core (dyadic fixed-point endpoints, outward rounding throughout — `exact.py`), extended (`ival_ext.py`) with rigorous interval LayerNorm (interval mean/variance, `1/sqrt` over an interval), tanh/`gelu_new` via a rigorous `exp` enclosure, and max-shifted interval softmax. Weights are treated as exact rationals derived from the float32 bit pattern, not approximations of it.

## Per-layer width table (TinyStories-1M, residual stream, P=192)

| after   | max width | median width | per-layer factor |
|---------|-----------|---------------|-------------------|
| layer 0 | 1.38e-50  | 3.65e-54      | (representation floor) |
| layer 1 | 2.00e-46  | 8.45e-50      | ×1.5e4 |
| layer 2 | 7.61e-41  | 3.53e-44      | ×3.8e5 |
| layer 3 | 2.36e-35  | 1.16e-38      | ×3.1e5 |
| layer 4 | 5.10e-30  | 3.07e-33      | ×2.2e5 |
| layer 5 | 8.45e-25  | 6.56e-28      | ×1.7e5 |
| layer 6 | 5.17e-19  | 3.95e-22      | ×6.1e5 |
| layer 7 | 4.29e-13  | 4.40e-16      | ×8.3e5 |
| final LN | 1.33e-10 | 6.32e-11     | ×3.1e2 |

Geometric-mean per-layer factor (layers 0→7, linear regime): ×2.3e5 ≈ 2^17.8 per layer at P=192, later tightened to **2^17.3** once the exp-tail floor (below) was fixed. This factor is precision-*independent*: P=96 shows the same slope until widths reach O(1) around layer 5 and the nonlinear saturation regime distorts it. Within a layer, the two LayerNorms dominate the compounding (roughly ×200–300 each, from the variance division's dependency loss); softmax contributes a further ×20; GELU and the matmuls are roughly width-neutral.

At P=96, widths cross O(1) during layer 5 and the chain explodes by the final layer norm — a clean illustration of the wall. At P=192, the full 8-layer chain survives with a final-logit width of order 1e-10 to 1e-12 against decision gaps of order 1, giving 30–40 bits of headroom depending on the prompt.

## Engine bugs found and fixed

Four bugs were found across two rounds of adversarial review, all in the interval extensions rather than the audited core inherited from certified-grokking, and all fixed as strict tightenings (never loosening an existing sound bound) with regression tests added to `test_ext.py`.

**1 — `exp` underflow fast path, inward above P≈101.** The softmax `exp` enclosure had a fast path returning the one-ulp interval `(0, 2^-P)` whenever the argument was very negative, using a fixed threshold calibrated for P≈96. At P=192, `exp(-70) ≈ 4e-31`, which is far larger than `2^-192` — the returned upper bound was roughly 27 orders of magnitude too small, i.e. unsound. Fixed with a precision-aware threshold, `x ≤ -(P+2)·ln2`, derived from a rational *upper* bound on ln2 so the guard is conservative at every precision.

**2 — tanh saturation fast path, inward above P≈137.** The same shape of bug in the tanh saturation shortcut: a fixed threshold (`|x| ≥ 48`) sound only up to roughly P=137, fixed the same way with a precision-aware threshold derived from the same rational ln2 bound.

Both original P=192 certificates turned out to be *accidentally* sound — per-layer artifact logging (added as part of the fix, and present in every width JSON in `evidence/`) shows neither fast path ever fired on either prompt at P=192, so the buggy thresholds were simply never on the decision path. The fix makes the engine sound as a function of precision rather than sound by luck, and the rerun numbers are bit-for-bit identical to the pre-fix numbers.

**3 — exp truncation-tail floor, precision-independent.** The `exp` enclosure's Taylor-series term count was a fixed function of the argument only (`N = floor(x)+60`), leaving a truncation tail that does not shrink as precision increases — found because one sweep prompt (a deliberately out-of-distribution, near-vocabulary-end token sequence) abstained with a width that was bit-identical at P=192 and P=256, which is exactly the symptom a genuinely precision-parametric enclosure should never show. Fixed by making the term count precision-aware (`N = floor(x)+60+P`) plus a rigorous early tail cut; both changes only tighten the bound. Post-fix, the previously-abstaining prompt certifies at P=192 with 37.3 bits of headroom.

**4 — a 61-digit pi constant, precision-independent.** `gelu_new`'s `sqrt(2/pi)` term was built from a hard-coded 61-digit decimal string for pi, giving it an enclosure width of roughly `2^-200` regardless of precision — invisible below P≈192, but it capped every GELU evaluation at around 200 effective bits at P=384, which aborted the first GPT-2-small run partway through layer 0. Fixed with a rigorous rational enclosure of pi via Machin's identity (`16·arctan(1/5) − 4·arctan(1/239)`) computed in exact arithmetic with a precision-scaled target width, so every constant in the engine is now precision-parametric.

## Experiment 1 — the 28-prompt sweep

28 varied seq-8 prompts, full 50,257-token vocabulary, P=192: 8 story openers, 6 mid-sentence fragments, 5 degenerate single-token repeats (including `<|endoftext|>` × 8), 4 rare/out-of-distribution token sequences, and 5 miscellaneous stress shapes (digits, capitals, punctuation-heavy). All token ids are recorded in `evidence/sweep_results.json`.

| statistic | value |
|-----------|-------|
| certified / abstained | 28 / 0 |
| headroom bits, min | 36.11 |
| headroom bits, p10 | 37.14 |
| headroom bits, median | 39.14 |
| headroom bits, p90 | 40.50 |
| headroom bits, max | 41.30 |

Max logit width varies only from 1.7e-13 to 1.1e-11 across all 28 prompts — post-fix, width is nearly prompt-independent, and headroom spread is mostly gap spread (the smallest float top-1/top-2 gap across the sweep was 0.116). The 16-token baseline prompt was also rerun at full vocabulary (the original run had sampled 200 logits): it certifies, width 7.41e-12, headroom 34.1 bits. No heavy tail; guard-fast-path hit count across the whole sweep: 0.

## Experiment 2 — the GPT-2-small prediction test

The TinyStories measurement (≈2^17.3 per layer) predicts that GPT-2-small (12 layers, versus TinyStories' 8) needs on the order of a few hundred bits of precision to certify. At P=384, the result: **certified**, full 50,257-token vocabulary, in 275.7 seconds of single-thread pure-Python integer arithmetic. Prompt: "The quick brown fox jumps over the lazy" (float top-1 continuation: ','). Architecture deltas handled: `Conv1D` weight layout, fused qkv, exact `1/8` attention scale, 12 layers × 12 heads × d_head 64, `gelu_new`, eps 1e-5, learned positional embeddings, tied unembedding.

- Max width over all 50,257 logits: 1.46e-27, against a float top-1/top-2 gap of 0.27036 — **87.3 bits of headroom**. Certified statement: the top-1 logit exceeds every competitor by at least 0.2703614814005327 in the exact-real semantics of the float32 weights.
- The full-vocabulary competitor set (rather than just the float top-200) was affordable because a runtime projection from layer 0's timing put the total run comfortably inside budget; both the projection and the actual wall time are in `evidence/gpt2_run.log`.
- Implied minimal certifying precision ≈ 384 − 87 ≈ 297 bits. This is the lower edge of the predicted band read off the observed headroom, **not** a measured minimum — no P≈297 run was performed.
- Measured slope: 2^23.6 per layer (geometric mean, layers 1→11) — model-dependent (TinyStories: 2^17.3), but the same linear-in-precision mechanism.
- First genuine guard-fast-path activations: the tanh saturation guard fired 24 times (GPT-2-small's outlier dimensions push GELU inner arguments to around 8821 at layer 2); the exp guard fired 0 times. Full per-layer widths and the guard audit are in `evidence/widths_gpt2_p384.json`.

## Adversarial review disposition

A standing second-opinion pass (Codex / GPT-5.5) reviewed the interval extensions and the P=192 TinyStories certificate, and a second pass reviewed the confirmation-stage fixes and the GPT-2-small run. Both are reflected in the bug list above and in the caveats in `README.md`. Two points worth stating explicitly, as the second review round did:

- The extrapolated ≈297-bit GPT-2-small threshold is an implication of the observed headroom, not a measured minimum.
- The per-layer slope is model-dependent (≈17.3 bits/layer on TinyStories, ≈23.6 on GPT-2-small); the stable claim is the *mechanism* — LayerNorm-dominated multiplicative growth, linear precision cost in depth — not the specific constant, which needs calibration per model family.

## Next steps

1. **A calibration grid** — many more prompts, multiple context lengths, several precisions, across more than two models — reporting per-layer amplification as a distribution (median, 90th/99th percentile) rather than a point estimate, before the exchange rate is used for planning.
2. **A third model at a different depth/width** to check whether the slope is closer to TinyStories' or GPT-2-small's, or moves further.
3. **Affine arithmetic or zonotopes at the two LayerNorms** — the dependency-loss sublayers responsible for most of the compounding — as the targeted fix if the calibration grid shows a heavy precision tail.
