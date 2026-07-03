# DRY RUN — NOT A HEADLINE FREEZE

These artifacts exercise the pre-registration and completeness machinery
end-to-end on a **small subset** of the pinned TinyStories validation corpus.
They are evidence that the freeze → certify → independent-check pipeline works;
they are **not** a headline population claim and must never be cited as one.

- `tinystories-dryrun.ids.json` — 12 real ctx-16 windows (a slice of the pinned
  validation corpus), token-ids only.
- `prereg.json` / `sample-index.json` — a format-2 pre-registration (exact
  rational δ budget, seed 1, n = 12). The drawn index multiset contains
  duplicates (with-replacement design).
- `records.jsonl` — 12 certified per-sample records carrying this freeze's
  `prereg_ref`.

The full transcript (commands + verbatim outputs, including the deliberate
failure paths and the determinism gate) is in `docs/prereg-dryrun.md`.

The headline freeze — a full-size pinned corpus, committed before certification
begins — is a separate, later artifact.
