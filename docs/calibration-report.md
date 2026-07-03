# Calibration report (Phase 1, non-headline)

Every record in this report has `prereg_ref=None` (A2: calibration samples are never headline samples). Abstention is never hidden — the taxonomy table below reports every abstain reason observed, including zero counts.

## Per-cell quantile tables

| model | ctx | P (cell ceiling) | n | n_cert | required-P p95 (95% CI) | required-P p99 (95% CI) | float-gap p95 | abstain taxonomy |
|---|---|---|---|---|---|---|---|---|
| tinystories | 8 | 256 | 100 | 100 | 160.0 [160.00, 160.00] | 160.0 [160.00, 160.00] | 4.07 | none |
| tinystories | 16 | 256 | 100 | 100 | 160.0 [160.00, 160.00] | 160.3 [160.00, 192.00] | 4.09 | none |
| tinystories | 32 | 256 | 100 | 100 | 160.0 [160.00, 160.00] | 160.0 [160.00, 160.00] | 4.94 | none |

## Float-gap distributions (top1-vs-top2, diagnostic)

- **tinystories ctx8 P256**: n=50 sampled gaps, min=0.0511 max=0.935
- **tinystories ctx16 P256**: n=50 sampled gaps, min=0.0125 max=1.17
- **tinystories ctx32 P256**: n=50 sampled gaps, min=0.053 max=1.28

## Runtime / memory quantiles

| model | ctx | P | runtime_s p95 | peak_rss_mb p95 |
|---|---|---|---|---|
| tinystories | 8 | 256 | 30.97 | 1011.2 |
| tinystories | 16 | 256 | 63.95 | 970.2 |
| tinystories | 32 | 256 | 106.05 | 984.8 |

## Fitted precision-depth curve (CI-B)

**Single-family grid: no cross-family slope is estimable from this summary alone.** The fit below degenerates to the one family's representative point (slope 0 by construction); the only cross-family depth measurement to date remains the foothold's two-model observation (~17.3 vs ~23.6 bits/layer, engine-seed/RESULTS.md), which this grid neither confirms nor extends across families.

`slope_bits_per_layer = 0.00`, `intercept = 160.00`, `r2 = 1.0000`

Per-family points used in the fit:

| family | depth (layers) | representative p95 required-P |
|---|---|---|
| tinystories | 8 | 160.0 |

> This is the first measured precision-depth curve for these models on this engine version — family-dependent slope, point inputs only; NOT a universal law.

## Decision point (Task 1.4 Step 4)

- **No §6 pivot triggered by this grid.** p99 tracks p95 at every context
  length (a single ctx-16 sample of 300 required 192 bits; every other sample
  certified at 160). Abstention rate is 0/300 — far below the ~20% KILL
  threshold. Required-P is not heavy-tailed and not visibly prompt-dependent
  at these context lengths.
- The brief's expectation "TS-1M required-P p95 ≈ 192 ± margin at ctx 8/16"
  is met on the tight side: measured p95 = 160 at ctx 8, 16 AND 32 (the
  foothold's P=192 sweet spot carried ~36 bits of headroom, consistent with
  certifying one escalation rung lower).
- **The GPT-2 KILL check (abstention at P=384 well under 20%) was NOT
  evaluated by this grid** — this run was TinyStories-only (GPT-2 is
  confirmation-only per A5, ~5 min/sample; a GPT-2 cell was out of this
  session's budget). It remains open for the Phase 1 gate proper.
- TinyStories-1M as headline: **confirmed** by this grid's P profile and
  zero-abstention behaviour.
