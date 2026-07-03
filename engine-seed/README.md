# lm-certification-foothold

*A preliminary research note: does exact-real interval certification survive real, pretrained LayerNorm transformers — and does the precision it costs scale the way a toy model predicts?*

> **We certify the full-vocabulary (50,257-way) next-token argmax of TinyStories-1M, a real pretrained LayerNorm transformer, by outward-rounded interval arithmetic — 28/28 varied prompts, ~7 s single-thread each. We then use the measured precision-vs-depth law as a genuine out-of-family prediction and certify GPT-2-small's full-vocabulary argmax at the predicted precision, with large headroom to spare. This is a foothold, not a scaling law: two models, short contexts, per-sample point inputs.**

This is the third leg of a small research line on certified guarantees for neural networks:

- [verified-circuits](https://github.com/brian-naughton/verified-circuits) — `Spec == Circuit == Model`, kernel-checked in Lean, for a self-trained toy transformer.
- [certified-grokking](https://github.com/brian-naughton/certified-grokking) — the same discipline applied to a canonical, third-party checkpoint (Nanda et al.'s grokking transformer), certified over its entire finite domain — but explicitly *without* LayerNorm, which the certified-grokking README names as the reason full-domain exact-real certification was feasible there.

LayerNorm was the open problem those two repos left on the table. This note is a first, honest look at what it costs.

## What this is, in plain English

The engines in verified-circuits and certified-grokking prove things about a model's *exact-real* function — treating the published float32 weights as exact rationals and computing with rigorous, outward-rounded interval arithmetic, so every enclosure is a sound bound on the true value, never an approximation that could silently be wrong. Neither of those models has LayerNorm; its variance division is exactly the kind of correlation-losing, "structureless error" operation that naive interval arithmetic is bad at. The question this note asks: does that method survive a real LayerNorm model at all, and if so, at what precision cost?

We extended the certified-grokking interval core with rigorous LayerNorm, tanh/`gelu_new`, and softmax, and pointed it at two real, pretrained, third-party checkpoints — `roneneldan/TinyStories-1M` (a small GPT-Neo model) and `gpt2` (GPT-2-small, 124M parameters) — neither trained nor modified by us. We measured how fast interval widths compound through the layers, found that the wall is real but *linear in precision* for point inputs, and then used that measurement to make and check a genuine prediction about a different, much bigger model.

## Status at a glance

| Claim | Current status |
|---|---|
| TinyStories-1M full-vocabulary argmax certified, 28/28 varied prompts | Certified, independently reproducible |
| Precision-vs-depth law predicts GPT-2-small's certifying precision | Prediction made, then checked — landed inside the predicted band |
| Per-layer amplification is a fixed, model-independent constant | Not established — slope is model-dependent (measured on two models) |
| Naive intervals handle set-valued (perturbation) inputs at this cost | Out of reach — this result is per-sample (point inputs) only |

## The two results

**(a) TinyStories-1M, full vocabulary, 192-bit precision.** For 28 varied prompts (story openers, mid-sentence fragments, degenerate repeats, rare/out-of-distribution token sequences, assorted stress shapes), the next-token argmax is certified over the *entire* 50,257-token vocabulary — the top-1 logit's certified lower bound exceeds every one of the other 50,256 logits' certified upper bounds — in about 7 seconds of single-thread pure-Python integer arithmetic per prompt. 28/28 certified, 0 abstentions. Headroom (bits of slack between the certified gap and the enclosure width) quantiles: minimum 36.1, median 39.1, 90th percentile 40.5. Full prompt list, token ids, and per-prompt numbers are in `evidence/sweep_results.json`.

**(b) GPT-2-small, full vocabulary, out-of-family prediction test.** The TinyStories measurement gives a rough precision-vs-depth law (see "The wall", below). We used it to predict, *before running the experiment*, that GPT-2-small — a different model family, three times the depth, twelve times the width — would need on the order of a few hundred bits to certify. At P = 384 bits, GPT-2-small's full-vocabulary next-token argmax certified in 275.7 seconds of single-thread pure-Python integer arithmetic, with headroom of 87.3 bits.

The honest summary of that result: **"A toy-model precision law got the GPT-2-small order of magnitude right: it predicted a few hundred bits, and GPT-2-small certified over the full vocabulary at P=384 with large headroom."** The implied minimal certifying precision — roughly 384 − 87 ≈ 297 bits — is an *extrapolation implied by the observed headroom*, not a measured minimum; no P≈297 run was performed. Full per-layer widths and the guard-activation audit are in `evidence/widths_gpt2_p384.json`.

## The wall

Within each transformer block, interval widths compound multiplicatively, and the two LayerNorms dominate that compounding — their variance division is where correlation between the interval endpoints is lost, exactly the "structureless error" failure mode naive interval arithmetic is known for. Measured per-layer amplification: roughly 2^17.3 on TinyStories-1M (d_model 64, 8 layers) and roughly 2^23.6 on GPT-2-small (d_model 768, 12 layers) — call it 17–24 bits per layer across the two models measured so far.

Because each certified run is a single *exact point* input (one prompt, not a perturbation ball or a symbolic input class), the starting enclosure width is just the representation floor, 2^-P. That means the wall moves **linearly** in precision: doubling the depth costs roughly a fixed number of extra bits, not a doubling of precision. That's what makes GPT-2-small certifiable at all with the same pure-Python engine — precision buys depth at a predictable, if not yet well-calibrated, exchange rate.

What this is *not*: a calibrated scaling law. The slope is visibly model-dependent (17.3 vs 23.6 bits/layer measured on exactly two models), and the moment an input becomes a *set* — a perturbation ball, a symbolic token class, anything beyond one exact point — the same compounding factor applies to the set's radius instead of the representation floor, and adding precision cannot help; that regime needs correlation-preserving arithmetic (affine forms / zonotopes) at the LayerNorms, not more bits. This is a foothold showing the method reaches real pretrained transformers one sample at a time, not a paper establishing how the constant scales.

## The certified object

As in the sister repos, the certified object is the **exact-real function** obtained by reading the released float32 tensors as exact dyadic rationals and evaluating with real (not floating-point) arithmetic: exact matrix sums, a rigorous `exp` enclosure inside softmax and `gelu_new`/tanh, and argmax taken over exact real logits. The authors' actual binary32 execution (PyTorch's specific reduction order, kernel-dependent softmax, etc.) is a conformance target, not the theorem: we do not formalise binary32 operational semantics. Observed float32-vs-float64 logit drift was 3e-5 (TinyStories) and 2e-4 (GPT-2-small), both many orders of magnitude below the decision gaps involved — conformance evidence, not a proof about float32 execution.

## Honest limits

- **Per-sample, not per-set.** Every certificate here is for one exact point input. Robustness over an input *set* (perturbation balls, token classes) is a different, harder problem that naive intervals cannot solve by adding precision — see "The wall" above.
- **Two models, short contexts.** The law is measured on TinyStories-1M (8 layers, d_model 64) and one GPT-2-small run (12 layers, d_model 768) at sequence lengths of 8–16 tokens. A calibration grid across more models, context lengths, and precisions is the obvious next step before this becomes a planning-grade number.
- **One GPT-2 prompt.** The out-of-family prediction test is a single prompt at a single precision. It is a genuine prediction that landed inside its band, not a distributional measurement.
- **Binary32 is not formalised**, as above — the certificate is about the exact-real function, with float32 conformance checked empirically, not proven.

## Ways this could still be wrong

- **Naive intervals lose correlations.** The whole method's soundness is not in question — outward rounding never lies — but its *tightness* is method-dependent: an affine-arithmetic or zonotope implementation of the same LayerNorms could plausibly certify at much lower precision. The ~17–24 bits/layer number measures this method's cost, not a fundamental limit of the task.
- **Small, non-representative sample.** 28 prompts on one small model, one prompt on one larger model. A heavy tail in headroom, or a model where the slope is far outside 17–24 bits/layer, would not be visible from this data.
- **Engine bugs are a real risk in fixed-point interval code**, and this project found several — see "About this project", below, and the guard-activation audit embedded in every width JSON in `evidence/` (per-layer min/max softmax and GELU/tanh argument ranges, and fast-path hit counts), which lets a reviewer check directly whether a given certified run ever depended on a precision-sensitive fast path.
- **Runtime protocol.** The GPT-2-small full-vocabulary competitor set was chosen after projecting total runtime from the first layer's timing, rather than being fixed in advance; the projection and the actual runtime are both recorded in `evidence/gpt2_run.log` for inspection.

Finding an actual instance of any of these is exactly the kind of review this note is asking for.

## Reproduce

Requires Python 3.11. The certified engine itself (`exact.py`, `ival_ext.py`) is pure standard library; `torch` and `huggingface_hub` are needed only for the float32 reference passes and for fetching weights.

```
python3.11 -m pip install -r requirements.txt
```

**Run the stdlib test suite** (no downloads, ~10 s):

```
python3.11 test_ext.py
```

Prints `ALL OK` if every enclosure check, every multi-precision guard-soundness check (P ∈ {96, 128, 160, 192, 256, 320, 384}), and the exp-tail and pi regressions pass.

**Fetch the model weights** (downloaded to a local, gitignored `data/hf_cache/`; TinyStories-1M is a few MB, GPT-2-small is ~500 MB):

```
python3.11 -c "from huggingface_hub import snapshot_download; \
  snapshot_download(repo_id='roneneldan/TinyStories-1M', cache_dir='data/hf_cache')"
python3.11 -c "from huggingface_hub import snapshot_download; \
  snapshot_download(repo_id='gpt2', cache_dir='data/hf_cache')"
```

(Set the `LMCERT_HF_CACHE` environment variable if you would rather point at an existing Hugging Face cache instead of downloading a second copy.)

**Certify one TinyStories prompt over the full vocabulary** (~5–7 s):

```
python3.11 interval_fwd.py 8 192 -1
```

(`seq_len=8`, `precision=192` bits, `n_logits=-1` meaning the full 50,257-token vocabulary.) Prints the certified logit bounds and `argmax_certified_among_chosen: true`, and writes a width JSON to the working directory — the curated copies used for the numbers in this README are in `evidence/`.

**Run the 28-prompt sweep** (~2–3 min):

```
python3.11 sweep.py
```

**Run the GPT-2-small out-of-family test** (~5 min at P=384):

```
python3.11 gpt2_interval.py 384
```

## What you're trusting

The certified numbers depend on the soundness of the interval core (`exact.py`, `ival_ext.py`) and its LayerNorm/tanh/`gelu_new`/softmax extensions — outward rounding throughout, spot-checked against float references in `test_ext.py` and, for the guard fast paths, against `mpmath` arbitrary-precision truth at seven precisions. You do not have to trust our numbers: `evidence/` holds the raw run logs and width JSONs (including the guard-activation audit) for every certified result quoted above, and every run is reproducible with the commands above from the published model weights alone.

## About this project and review request

This project was executed AI-first: Claude (Anthropic) was used as the researcher-engineer in an agentic implementation loop, and GPT-5.5 (OpenAI Codex) was used for standing adversarial AI review passes — with the project directed, judged, reviewed, and owned by Brian Naughton. Transparently: two rounds of adversarial review on this experiment caught four engine bugs — two precision-dependent soundness bugs in the exp/tanh fast paths (sound only below roughly P=100–140, silently inward above it) found in the first pass, and two further precision-*independent* floors (a hard-coded Taylor-series cutoff in the exp enclosure, and a 61-digit pi constant) found by the confirmation experiments themselves, when a certificate refused to tighten with added precision — before any claim in this note stood. All four are fixed as strict tightenings with multi-precision regression tests in `test_ext.py`.

**Peer review is genuinely requested** — AI review is not a substitute for it. It is especially welcome on the interval arithmetic (particularly the LayerNorm and softmax extensions), the precision-vs-depth measurement and its limits, and the runtime/competitor-set protocol for the GPT-2-small run. Corrections and failed replications are welcome as issues.

I am also looking for AI research and engineering roles: [Brian Naughton on LinkedIn](https://www.linkedin.com/in/bnaughton/).

## Licence

MIT — see `LICENSE`.
