/-
Task 2.4 (spec §8-A5 timebox target): generic Hoeffding lower-confidence bound.

For `n` i.i.d. `{0,1}`-valued (Bernoulli) random variables `B i` with common mean
`p`, with probability ≥ 1 − δ over the sampling,

    p ≥ (∑ i, B i) / n − √(ln(1/δ) / (2n)),

so any certified success count `k ≤ ∑ i, B i` yields `p ≥ k/n − √(ln(1/δ)/(2n))`.

Route (per the task brief): each centred variable `B i − p` lies in an interval of
length 1, so Hoeffding's lemma (`hasSubgaussianMGF_of_mem_Icc`, Mathlib v4.30.0)
makes it sub-Gaussian with parameter `c = (b − a)²/4 = 1/4`; the Chernoff bound for
sums of independent sub-Gaussian variables
(`HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun`) then gives
`P[∑ (B i − p) ≥ nε] ≤ exp(−(nε)²/(2·n·(1/4))) = exp(−2nε²)`, which equals δ at
`ε = √(ln(1/δ)/(2n))`.

TAIL-DIRECTION NOTE: the failure mode of the lower confidence bound
`p ≥ k/n − ε` is the empirical mean OVERSHOOTING `p` (i.e. `∑(B i − p) > nε`), so
the relevant tail is the upper tail of `∑ (B i − p)` — not, as the task brief's
sketch had it, the upper tail of `∑ (p − B i)` (that tail controls the upper
confidence bound `p ≤ k'/n + ε`). Both tails obey the same `exp(−2nε²)` bound by
symmetry; the formalisation below proves the direction the headline claim
actually needs, and the Lean type-checker is what surfaced the flip.

Numeric cross-check against the claim-freeze numbers (docs/claim-freeze.md):
headline `n = 1000`, φ₁ budget share `δ₁ = 1/40 = 0.025` (per the runbook split
`δ = 0.05 = 0.025 + 0.025` — δ₁ is 1/40, NOT 1/20):
    ε = √(ln 40 / 2000) = √(3.68888…/2000) = √0.00184444… ≈ 0.042947 ≈ 0.0430,
so on an all-success run (k = n = 1000, empirical mean 1) the certified lower
confidence bound is `L₁ = 1 − 0.0430 = 0.9570` at confidence `1 − δ₁ = 0.975`.

SCOPE NOTE (A5): this theorem is generic over ANY i.i.d. Bernoulli family on an
abstract probability space. The finite-corpus sampling bridge (that seeded
with-replacement draws from the pinned corpus instantiate these hypotheses) is
deliberately NOT formalised — it is handled by audited provenance
(docs/PROVENANCE.md), per the task brief and spec §8-A5.
-/
import Mathlib.Probability.Moments.SubGaussian

open MeasureTheory ProbabilityTheory Real
open scoped NNReal

/-- **Generic Hoeffding lower-confidence bound** (one-sided, i.i.d. Bernoulli).

If `B 0, …, B (n−1)` are independent random variables taking values in `[0,1]`
(in particular `{0,1}`-valued indicators) with common mean `p`, then with
probability at least `1 − δ` the true mean `p` is at least the empirical mean
`(∑ i, B i) / n` minus `√(ln(1/δ)/(2n))`.

Deltas from the task-brief statement (documented):
* a `Measurable (B i)` hypothesis is added — `iIndepFun` alone does not supply the
  measurability needed for the tail bound and the complement argument;
* the conclusion is stated via `μ.real` (the real-valued probability) rather than
  the raw `ℝ≥0∞`-valued measure, matching the Chernoff-bound API in
  `Mathlib.Probability.Moments.SubGaussian`; for a probability measure the two
  forms carry the same content;
* the event is `{(∑ B)/n − ε ≤ p}`, the tail direction the claim `p ≥ k/n − ε`
  actually requires (see the tail-direction note in the file header). -/
theorem hoeffding_lower_confidence
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (n : ℕ) (hn : 0 < n) (δ : ℝ) (hδ : 0 < δ) (hδ1 : δ < 1)
    (B : Fin n → Ω → ℝ) (hBm : ∀ i, Measurable (B i))
    (hB01 : ∀ i, ∀ ω, B i ω ∈ Set.Icc (0 : ℝ) 1)
    (hindep : iIndepFun B μ)
    (p : ℝ) (hp : ∀ i, μ[B i] = p) :
    μ.real {ω | (∑ i, B i ω) / n - Real.sqrt (Real.log (1 / δ) / (2 * n)) ≤ p}
      ≥ 1 - δ := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  set ε : ℝ := Real.sqrt (Real.log (1 / δ) / (2 * n)) with hε_def
  have hε0 : 0 ≤ ε := Real.sqrt_nonneg _
  -- The radicand is nonnegative: 1/δ ≥ 1, so log (1/δ) ≥ 0.
  have hlog0 : 0 ≤ Real.log (1 / δ) := by
    apply Real.log_nonneg
    rw [le_div_iff₀ hδ]
    linarith
  have hrad0 : 0 ≤ Real.log (1 / δ) / (2 * n) := by positivity
  have hε_sq : ε ^ 2 = Real.log (1 / δ) / (2 * n) := Real.sq_sqrt hrad0
  -- Centred variables: Y i = B i − p, mean 0, values in [−p, 1 − p].
  set Y : Fin n → Ω → ℝ := fun i ω => B i ω - p with hY_def
  have hYindep : iIndepFun Y μ :=
    hindep.comp (fun _ x => x - p) fun _ => measurable_id.sub measurable_const
  -- Each Y i is sub-Gaussian with parameter 1/4 (Hoeffding's lemma, c = (b−a)²/4).
  have hYsub : ∀ i, HasSubgaussianMGF (Y i) (1 / 4 : ℝ≥0) μ := by
    intro i
    have h := hasSubgaussianMGF_of_mem_Icc (μ := μ) (X := B i) (a := 0) (b := 1)
      (hBm i).aemeasurable (Filter.Eventually.of_forall (hB01 i))
    have hc : ((‖(1 : ℝ) - 0‖₊ / 2) ^ 2 : ℝ≥0) = 1 / 4 := by
      rw [sub_zero, nnnorm_one]
      ext
      norm_num
    rw [hc, hp i] at h
    exact h
  -- Chernoff/Hoeffding tail bound for the sum of the Y i at threshold nε.
  have key := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun (s := Finset.univ)
    (c := fun _ => (1 / 4 : ℝ≥0)) hYindep (fun i _ => hYsub i)
    (ε := n * ε) (by positivity)
  -- The exponential bound evaluates to exactly δ.
  have hsum_c : ((∑ _i : Fin n, (1 / 4 : ℝ≥0) : ℝ≥0) : ℝ) = n / 4 := by
    push_cast [Finset.sum_const, Finset.card_univ, Fintype.card_fin]
    ring
  have hexp : Real.exp (-(n * ε) ^ 2 / (2 * ((∑ _i : Fin n, (1 / 4 : ℝ≥0) : ℝ≥0) : ℝ)))
      = δ := by
    rw [hsum_c]
    have harg : -((n : ℝ) * ε) ^ 2 / (2 * ((n : ℝ) / 4)) = Real.log δ := by
      have h1 : -((n : ℝ) * ε) ^ 2 / (2 * ((n : ℝ) / 4)) = -(2 * n * ε ^ 2) := by
        field_simp
        ring
      rw [h1, hε_sq]
      have h2 : 2 * (n : ℝ) * (Real.log (1 / δ) / (2 * n)) = Real.log (1 / δ) := by
        field_simp
      rw [h2, Real.log_div one_ne_zero hδ.ne', Real.log_one]
      ring
    rw [harg, Real.exp_log hδ]
  have key' : μ.real {ω | n * ε ≤ ∑ i, Y i ω} ≤ δ := by
    calc μ.real {ω | n * ε ≤ ∑ i, Y i ω}
        ≤ Real.exp (-(n * ε) ^ 2 / (2 * ((∑ _i : Fin n, (1 / 4 : ℝ≥0) : ℝ≥0) : ℝ))) := key
      _ = δ := hexp
  -- Complement argument: the bad event {(∑ B)/n − ε > p} sits inside the tail event.
  have hAmeas : MeasurableSet {ω | (∑ i, B i ω) / n - ε ≤ p} := by
    apply measurableSet_le _ measurable_const
    exact ((Finset.measurable_sum Finset.univ fun i _ => hBm i).div_const _).sub
      measurable_const
  have hsubset : {ω | (∑ i, B i ω) / n - ε ≤ p}ᶜ ⊆ {ω | n * ε ≤ ∑ i, Y i ω} := by
    intro ω hω
    simp only [Set.mem_compl_iff, Set.mem_setOf_eq, not_le] at hω
    simp only [Set.mem_setOf_eq, hY_def]
    have hsumY : ∑ i, (B i ω - p) = (∑ i, B i ω) - n * p := by
      rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul]
    -- From p < (∑ B)/n − ε and n > 0: n·ε ≤ (∑ B) − n·p.
    have hlt : (p + ε) * n < ∑ i, B i ω := by
      have := (lt_div_iff₀ hn').mp (by linarith : p + ε < (∑ i, B i ω) / n)
      linarith
    rw [hsumY]
    nlinarith
  have hbad : μ.real {ω | (∑ i, B i ω) / n - ε ≤ p}ᶜ ≤ δ :=
    le_trans (measureReal_mono hsubset (measure_ne_top μ _)) key'
  rw [probReal_compl_eq_one_sub hAmeas] at hbad
  linarith

/-- Count form matching the claim-freeze wording: on the same ≥ 1 − δ event, ANY
certified success count `k` with `k ≤ ∑ i, B i` yields the lower confidence bound
`p ≥ k/n − √(ln(1/δ)/(2n))`. (For `{0,1}`-valued `B i`, take `k` = the number of
certified successes; `k/n ≤ (∑ B)/n` then transfers the bound.) -/
theorem hoeffding_lower_confidence_count
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (n : ℕ) (hn : 0 < n) (δ : ℝ) (hδ : 0 < δ) (hδ1 : δ < 1)
    (B : Fin n → Ω → ℝ) (hBm : ∀ i, Measurable (B i))
    (hB01 : ∀ i, ∀ ω, B i ω ∈ Set.Icc (0 : ℝ) 1)
    (hindep : iIndepFun B μ)
    (p : ℝ) (hp : ∀ i, μ[B i] = p) :
    μ.real {ω | ∀ k : ℕ, (k : ℝ) ≤ ∑ i, B i ω →
        (k : ℝ) / n - Real.sqrt (Real.log (1 / δ) / (2 * n)) ≤ p}
      ≥ 1 - δ := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  refine le_trans (hoeffding_lower_confidence μ n hn δ hδ hδ1 B hBm hB01 hindep p hp)
    (measureReal_mono ?_ (measure_ne_top μ _))
  intro ω hω k hk
  simp only [Set.mem_setOf_eq] at hω ⊢
  have hkn : (k : ℝ) / n ≤ (∑ i, B i ω) / n := by gcongr
  linarith
