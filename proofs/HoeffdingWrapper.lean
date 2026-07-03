import HoeffdingWrapper.Basic

-- Axiom audit (Task 2.4 acceptance): expected exactly
-- [propext, Classical.choice, Quot.sound] — Mathlib's classical base.
-- Must NOT contain sorryAx or any native_decide axiom (Lean.ofReduceBool).
#print axioms hoeffding_lower_confidence
#print axioms hoeffding_lower_confidence_count
