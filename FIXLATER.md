# Fix later

A tracked, public register of known follow-ups that are not blockers for the current
headline claims (see [`README.md`](README.md) and [`docs/PROVENANCE.md`](docs/PROVENANCE.md)),
but are worth doing next. Nothing below is a correctness gap in what is currently claimed —
each is either a strengthening of evidence, a pin that reduces future drift risk, or a
test that would make an existing guarantee easier to trust at a glance.

- **Corpus rebuild-determinism test.** Add an automated test that rebuilds
  `certificates/corpora/wikitext103-test.ids.json` (and the TinyStories equivalent) from
  source and asserts the resulting token-id file is byte-identical to the committed one.
  Currently this is checked manually; see `docs/PROVENANCE.md`'s "Follow-ups tracked"
  note.

- **HF revision pins.** `certinf/corpus.py`'s `AutoTokenizer.from_pretrained(...)` calls
  (GPT-2 tokenizer, WikiText dataset load) do not pin an explicit HF Hub revision/commit
  hash. Pinning would remove any dependency on upstream default-branch drift for corpus
  rebuilds.

- **GPT-2 checker path.** The torch-free checker (`certificates/check.py`) is
  TinyStories-shaped and does not yet re-derive the GPT-2-small confirmation
  certificates (`certificates/gpt2-confirmation/`) from the (gitignored, ~2.1 GB) hex
  weight export. Already tracked in the README roadmap; listed here too so it is visible
  from a single follow-up index.

- **Near-tie synthetic terminal-path test.** Add a synthetic (constructed, not sampled)
  test case that drives `certinf/certify.py`'s near-tie abstention path
  (`abstain_reason="near-tie"`) end-to-end, confirming escalation correctly terminates
  without retrying, rather than relying only on the taxonomy's unit-level checks.

- **Grid summary location.** `certinf/grid.py` writes `<out_dir>/summary.json`, while
  `certinf/lawfit.py`'s default `--summary` argument points at
  `calibration/summary.json`. These agree by convention in the current calibration run,
  but the relationship is implicit; worth either documenting explicitly or making the
  grid runner's output path the single source of truth `lawfit` defaults to.

- **CLI transcript-sha environment pin note.** `certinf/harness.py`'s
  `transcript_sha256` is deliberately path-independent (see its docstring), but the CLI
  entry point does not currently print or pin the exact environment (OS build, Python
  patch version) alongside the transcript hash. Worth a short note or flag so a reader
  reproducing the harness has the pinned environment values in one place rather than
  cross-referencing the run manifest.
