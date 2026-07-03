# Claim freeze — the pre-registered public claim and the freeze runbook

*This document is the source of truth for the headline wording. It freezes the exact
public claim text **before** any headline certificate is generated, states the Lean
status switch, fixes the language rules, and gives the commit choreography for the
freeze. Its own commit timestamp — with the parameters bound but no results present —
is part of the pre-commitment record (see [Freeze runbook](#freeze-runbook-m7)).*

The wording here is adopted from the whole-corpus evaluation
(`docs/reviews/2026-07-03-codex-whole-corpus-evaluation.md`, Q3) and the design-spec §8
amendments (`docs/superpowers/specs/2026-07-03-certified-inference-design.md`). Nothing
in the headline may drift from the frozen text below without a fresh, timestamped edit
to this file.

---

## Parameter slots (bound at freeze time)

The claim text is frozen verbatim with the slots below left as named placeholders. They
are bound exactly once, at the freeze commit, from `prereg.json` and the completed
headline run — and never edited afterwards.

| Slot | Meaning | Source |
|---|---|---|
| `C` | the pinned finite TinyStories-1M token-id corpus at the headline context length | `prereg.json` `corpus_sha256` |
| `n` | pre-registered sample size (with replacement) | `prereg.json` `n` |
| `k` | number of the `n` sampled prompts that certified (φ₁) | headline records |
| `k₂` | number of the `n` sampled prompts on which the harness agreed (φ₂_joint) | headline records |
| `P_max` | precision ceiling of the escalation policy (headline candidate: 256) | `prereg.json` `P_max` |
| `δ` | total confidence budget (headline candidate: 0.05) | `prereg.json` `delta` |
| `δ₁` | φ₁'s share of the budget (headline candidate: 0.025) | `prereg.json` `delta_split["phi1"]` |
| `δ₂` | φ₂_joint's share of the budget (headline candidate: 0.025) | `prereg.json` `delta_split["phi2_joint"]` |
| `L₁` | Hoeffding lower bound on the φ₁ population rate over `C` at confidence `1 − δ₁` | computed from `k`, `n`, `δ₁` |
| `L₂` | Hoeffding lower bound on the φ₂_joint population rate over `C` at confidence `1 − δ₂` | computed from `k₂`, `n`, `δ₂` |

The one-sided Hoeffding lower bound is `L = k/n − sqrt(ln(1/δ) / (2n))`, clipped to
`[0, 1]`, computed from exact `(k, n, δ)` inputs with stable decimal rendering.

---

## The headline claim (frozen verbatim)

### φ₁ — the title claim

> For a pre-registered with-replacement sample of `n` prompts from the pinned finite
> TinyStories-1M token-id corpus `C` at context length 16, every sampled prompt [or:
> `k` of `n`] had a unique full-vocabulary exact-real next-token argmax certified at
> `P ≤ 256`. By the stated Hoeffding bound, with confidence `1 − δ₁`, at least `L₁` of
> `C` is certifiable under this engine and escalation policy.

Use the **"every sampled prompt"** wording only if `k = n` (all-success run); otherwise
use the **"`k` of `n`"** wording. φ₁ is the scientific core — exact-real,
full-vocabulary argmax uniqueness — and it carries the title.

The strongest clean one-line form, for the eventual README headline, is:

> Pre-registered exact-real full-vocabulary certificates for TinyStories-1M: `k`/`n`
> sampled ctx-16 prompts certified at `P ≤ 256` on a pinned finite corpus, yielding an
> `L₁` Hoeffding lower bound at `δ₁ = 0.025`; pinned float32 harness agreement reported
> as a separate joint-event bound.

### φ₂_joint — the subtitle / table claim

> In the same frozen run, the pinned float32 harness agreed with the certified
> exact-real argmax on `k₂`/`n` samples, giving a separate lower bound `L₂` on the joint
> event ("certified **and** harness-agrees"), with confidence `1 − δ₂`.

φ₂_joint belongs in the subtitle or the status table — never the title. It is
**pinned-environment conformance**, not a broad deployment-gap statement, and its
wording is explicitly conditional on the determinism gate (below).

### φ₂_joint determinism gate (M6) — the switch

φ₂_joint may be stated as a certified population bound **only if** the pinned inference
harness is deterministic across repeated runs under the exact frozen command line and
context length, with the transcript SHA and environment recorded in the run manifest.

- **If the determinism gate passes:** publish φ₂_joint as above — a certified lower
  bound `L₂` on the joint event.
- **If determinism wobbles (M6 KILL):** φ₂_joint **demotes to measured conformance**.
  Drop it from the headline entirely and report it descriptively as an
  implementation-specific observation — "the pinned harness agreed on `k₂`/`n` samples
  in this environment" — with no Hoeffding bound and no population language.

φ₁ stands regardless of the φ₂_joint gate outcome.

---

## The Lean status switch (M4)

The Lean wrapper theorem has a one-week timebox. Which of the two wordings below ships is
decided by whether the theorem lands in that timebox — and the decision is made **before**
the headline is stated, not negotiated afterwards. "Kernel-checked" **never** appears
before the theorem exists and its numeric instantiation path is tested.

- **If the statistical lower-bound theorem lands in its timebox, use verbatim:**

  > The statistical lower-bound theorem is kernel-checked in Lean; per-sample facts are
  > interval-certified and torch-free re-checked; sampling/harness provenance is audited.

- **If it does not land, use verbatim:**

  > Per-sample certificates and checker are complete; statistical wrapper is the standard
  > Hoeffding bound with a paper proof and numeric audit; Lean wrapper deferred.

Both are publishable. A false "kernel-checked" label would be far more damaging than a
deferred Lean wrapper — so the deferred wording is the default, adopted unless and until
the theorem is genuinely kernel-checked.

---

## Language rules (non-negotiable)

These are the overclaim guards from the evaluation. They apply to every public
surface — README, technical note, social card, commit messages, this file.

| Always say | Never say | Why |
|---|---|---|
| "pinned finite corpus `C`" (name the population) | "TinyStories-1M certifiability" (bare) | The claim quantifies over `C` only — not TinyStories validation, not story prompts generally, not any deployment distribution. |
| "torch-free checker sharing an audited arithmetic core" | "independent implementations" / "two independent checkers" | The checker is torch-free and independent of the generator's weight-loading path, but shares `certinf.exact` / `certinf.ival_ext`. A shared arithmetic bug fools both instruments. |
| "calibration table" / "calibration evidence" / "precision table" | "calibration law" / "precision-depth law" | The Phase 1 grid degenerates to a single-family point; no cross-family law is estimated. |
| GPT-2 as "prestige confirmation" / "scaling evidence" (one full-vocab certificate) | any GPT-2 population or statistical-headline language | GPT-2 lacks the checker/completeness path and the sample budget; it must not carry population language. |

Additional standing wording (from the spec §8 amendments):

- "pinned float32 execution" is an **implementation transcript** (hardware/OS/Python/
  PyTorch versions, deterministic flags, TF32 off, eval mode, token-id corpus, checkpoint
  hash, command line) — empirical conformance, **never** "binary32 semantics".
- The wrapper is a "kernel-checked statistical wrapper over audited certificate records",
  **never** "certified end-to-end"; enumerate the trust strata (kernel / interval-certified
  records / provenance-audited sampling + harness) wherever the wrapper is claimed.
- Argmax claims state **uniqueness** explicitly ("a unique full-vocabulary … argmax").

---

## Freeze runbook (M7)

The pre-commitment is established externally — by the commit choreography and its
public timestamps — not by anything `verify()` can check from the files alone (see
`certinf/prereg.py`, WITNESS SEMANTICS). The choreography below is mandatory.

### Commit choreography

**Commit 1 — the precommitment (no results).** In a single commit, with the parameters
bound but **no certificate records present**:

- `prereg.json` — the frozen `PreRegistration` tuple (model, `checkpoint_sha256`,
  `corpus_sha256`, `context_length`, `P_max`, `n`, `delta`, `delta_split`, `seed`,
  `phi_definitions`, `escalation_policy`, `sample_index_sha256`, `prereg_sha256`).
- `sample-index.json` — the frozen with-replacement draw (duplicates kept, never
  deduplicated).
- this `docs/claim-freeze.md`, with every parameter slot bound to its frozen value.

The **timestamp of commit 1 is the precommitment**. It must contain no headline
certificate results. This is what publishing `prereg_sha256` (as each certificate's
`prereg_ref`) and the version-control timestamp together buy: evidence that the seed and
budget were fixed before the results were seen.

**Commit 2 and later — the certificate records.** Only after commit 1 is committed
(and, for the public record, pushed) does certification begin. Each headline record
carries `prereg_ref = prereg_sha256` and lands in commit 2 or later.

**Gate — no claim before the checker passes.** The torch-free checker must pass with
`--prereg` — re-deriving every record from the hex weights and the committed token-id
corpus, re-validating the delta split, asserting full-corpus completeness (no silent
omission), and confirming every record's `prereg_ref` — **before any headline number is
stated anywhere**. A claim stated ahead of a green `--prereg` pass is not permitted.

### `n` decision rule

The Hoeffding lower bound on an all-success run (`k = n`) at `δ₁ = 0.025` is
`1 − sqrt(ln(1/0.025) / (2n))`:

| `n` | Role | All-success (`k = n`) lower bound `L₁` at `δ₁ = 0.025` |
|---|---|---|
| 1000 | **Primary** | ≈ 0.9571 → headline "at least 95.7%" |
| 750 | **Fallback** | ≈ 0.9504 → headline "at least 95.04%" (smallest clean "above 95%") |
| 500 | (reference only) | ≈ 0.9393 — below the clean-headline threshold; not used |

- **Primary:** `n = 1000`. If the run is `1000/1000`, the headline is **"at least
  95.7%"** at `δ₁ = 0.025`.
- **Fallback:** if the compute budget forces it, `n = 750`. If `750/750`, the headline is
  **"at least 95.04%"** — the smallest `n` that still clears a clean "above 95%"
  all-success headline at `δ₁ = 0.025`.
- No adaptive stopping, no `n`-extension, no post-hoc promotion: `n` is fixed at
  commit 1 and the run is carried to that `n` regardless of interim observations.
