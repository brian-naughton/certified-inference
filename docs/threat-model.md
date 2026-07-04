# Ways this could still be wrong

*The one-page threat model for the eventual public README. Same register as the sister
project [certified-grokking](https://github.com/brian-naughton/certified-grokking)'s
"Ways this could still be wrong" section: we would rather name the failure modes than let
a reviewer discover them. Each is a real way the guarantee could break, with the
mitigation stated honestly beside it.*

- **Shared interval-arithmetic core (single point of failure).** The torch-free checker
  is independent of PyTorch and of the generator's weight-loading path, and it mirrors the
  transformer wiring separately — but it imports the **same** `certinf.exact` and
  `certinf.ival_ext` as the generator. A soundness bug in outward rounding, the `exp`
  enclosure, LayerNorm intervalisation, GELU/tanh guards, or the softmax interval logic
  would fool both instruments. Mitigation: the core is small, hand-written stdlib interval
  math, has been through adversarial review with the precision-floor bugs found and fixed,
  and is corroborated numerically — but torch-free re-derivation is **not** an independent
  mathematical implementation, and "bit-identical" means *determinism plus the soundness of
  that shared core*, not two independent implementations agreeing. We keep the core small
  and invite direct review of it; the strongest upgrade here is an external arithmetic
  audit or a second minimal implementation of the primitives, not more records through the
  same code.

- **Exact-real vs binary32 (the harness is an implementation transcript).** The
  certificate is about the **exact-real function** — the published float32 tensors read as
  exact dyadic rationals, with exact real arithmetic and argmax over exact real logits. The
  pinned float32 harness (φ₂_joint) is an *implementation transcript* — a specific
  hardware/OS/Python/PyTorch/flags/checkpoint/command-line configuration — not a formalised
  binary32 operational semantics. Mitigation: agreement between the harness and the
  certified argmax is reported as measured conformance with its environment pinned and its
  determinism gated (see `docs/claim-freeze.md`, M6) — but it is corroboration, never a
  theorem about binary32 execution.

- **Finite-corpus population (the claim quantifies over `C` only).** The Hoeffding bound
  is over a pinned finite token-id corpus `C` — a fixed list of windows at one context
  length — **not** over TinyStories validation, not over story prompts generally, and not
  over any language-model deployment distribution. `C` is a narrow, first-slice
  construction: a deterministic first-slice of windows, not a representative random sample
  of the validation set. We say so plainly. Mitigation: the population is named as `C`
  everywhere the bound is stated, and the corpus is sha-pinned and committed as token-id
  lists (no tokeniser trust) — but a reader must not read "TinyStories-1M certifiability"
  into a claim that is, precisely, "certifiability over `C`".

- **Corpus representativeness.** Because `C` is a first-slice construction, the observed
  precision profile (required-P p95 ≈ 160 at `P_max = 256`, zero abstentions across
  ctx 8/16/32) could in principle be an artefact of which windows the first slice happens
  to contain. Mitigation: the calibration table is honest about the construction, and a
  randomised-corpus calibration table is on the strengthening agenda to show the profile is
  not an artefact of the first 200 windows — but the headline claim itself makes no
  representativeness assertion beyond `C`.

- **Pre-registration witness = determinism, not pre-commitment.** `prereg.verify()` proves
  the committed `prereg.json` and `sample-index.json` are the deterministic,
  arithmetically-consistent output of the committed seed against the committed corpus — it
  does **not** prove the seed was chosen before the results were seen. A `prereg.json` can
  be regenerated after the fact and still verify `True`. Pre-commitment is therefore
  established **externally**, by the freeze commit (parameters bound, no certificate results
  in that commit) and by publishing `prereg_sha256` as each certificate's `prereg_ref`.
  Honest limit of run 1's evidence: a git commit's own timestamp is **self-attested
  metadata** — it is written by the committer's machine, not by a third party. Run 1 (the
  original headline) was published to GitHub in a **single push after its results already
  existed**, so no third-party timestamp separates its freeze commit from its result commit;
  its pre-commitment rests on three softer supports instead — (1) the **commit-graph
  ordering** (the freeze commit is a parent of the result commit, and contains no result
  files), (2) the declared **nothing-up-my-sleeve seed convention** (seed = the project
  date, `20260703`, not a value that could be shopped across many candidates for a
  flattering draw), and (3) the fact that the **calibration table made the all-success
  outcome expected in advance** (required-P p95 ≈ 160 at `P_max = 256`, zero abstentions),
  so there was no selective-publication pressure to hide a bad run. These are real but they
  are not third-party attestation. **Protocol upgrade (v2, from run R2 onward):** the freeze
  commit is **pushed publicly before the run is started**, so GitHub's server-side receive
  timestamp — a third party's clock, not the committer's — records the pre-commitment ahead
  of any result existing. The replication run R2 (`certificates/prereg/headline-r2/`, seed
  `20260704`, tag `prereg-r2`) is the first run frozen under v2; see `docs/claim-freeze.md`.

- **Lean wrapper status.** Whether the statistical lower-bound theorem is kernel-checked in
  Lean is a timeboxed outcome, not a foregone one. Until the theorem lands and its numeric
  instantiation path is tested, the wrapper is the standard Hoeffding bound with a paper
  proof and a numeric audit, and the public text says exactly that (`docs/claim-freeze.md`,
  M4). Mitigation: "kernel-checked" is never used before the theorem exists; the deferred
  wording is the default. Even when it lands, the kernel checks the **statistics** — the
  per-sample facts remain interval-certified and the sampling/harness provenance remains
  audited; the wrapper is never described as "certified end-to-end".

- **Checkpoint/export two-instrument boundary.** The exact-real object could disagree with
  the artifact the model's authors released if the tensors were read in wrong. The loader
  binds checkpoint bytes to hex weights; the torch-free checker re-derives from those hex
  weights and the committed token-id corpus without Torch. Mitigation: the checkpoint is
  sha256-pinned and the hex export is the audited boundary between the two instruments —
  but the pinned artifact itself is *trusted* to be the published checkpoint, and the
  float→rational export is corroborated, not itself kernel-checked.

Finding an actual instance of any of these is exactly the peer review we are requesting.
Corrections and failed replications are welcome as issues.
