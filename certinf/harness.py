"""Pinned float32 inference harness + implementation transcript (A1).

"Pinned float32 execution" is an implementation transcript, not "binary32
semantics": hardware, OS, Python version, PyTorch version, deterministic
flags, TF32 off, eval() mode, the checkpoint sha256, and the exact command
line. This is empirical conformance evidence only — never the theorem. The
theorem is the exact-real certificate produced by certinf.certify; this
module's job is to record, reproducibly, how the *deployed* float32 model was
actually run, and to check that running is deterministic in this environment
(A1 CI-D kill: if it is not, the deployment-gap claim φ2_joint downgrades to
an implementation-specific observation, no population claim).

Default deployment harness = the from-scratch float32 reference forward
(certinf.float_fwd.forward / certinf.gpt2_float.forward) — the same manual
implementation whose float64 pass already supplies the certified engine's
diagnostic top1/top2 gap, so the "pinned harness" and "the reference the
certified engine compares against" are the same code path, avoiding an
extra silent implementation to trust. The stock HuggingFace model's top-1 is
NOT computed by this module (out of Task 1.2's scope — no test requires it);
if it is added later it must be recorded ALONGSIDE the from-scratch pass as
a second observation, never substituted for it (see the design brief).
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys

from certinf import corpus as corpus_mod

_DETERMINISM_CONFIGURED = False


def _configure_deterministic_cpu_float32() -> None:
    """Idempotently set the A1 pinned-execution flags: TF32 off, deterministic
    algorithms, CPU float32."""
    global _DETERMINISM_CONFIGURED
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    _DETERMINISM_CONFIGURED = True


def transcript(checkpoint_sha256: str | None = None,
               command_line: str | None = None) -> dict:
    """A1 verbatim implementation-transcript block.

    Keys: platform, os, python, torch_version, deterministic_flags, tf32
    (False), eval_mode (True), checkpoint_sha256, command_line.
    """
    _configure_deterministic_cpu_float32()
    import torch

    return {
        "platform": platform.platform(),
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version,
        "torch_version": torch.__version__,
        "deterministic_flags": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "use_deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()),
        },
        "tf32": False,
        "eval_mode": True,
        "checkpoint_sha256": checkpoint_sha256,
        "command_line": command_line if command_line is not None else " ".join(sys.argv),
    }


def transcript_sha256(checkpoint_sha256: str | None = None,
                      command_line: str | None = None) -> str:
    """sha256 of the canonical-JSON transcript block."""
    t = transcript(checkpoint_sha256=checkpoint_sha256, command_line=command_line)
    canonical = json.dumps(t, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _forward_last_logits(model: str, weights, token_ids: list[int]):
    """Pinned float32 forward, last-position logits, via the from-scratch
    reference implementation (the deployment harness — see module docstring).
    `weights` is either a weights_path (str) or an already-loaded state dict.
    """
    _configure_deterministic_cpu_float32()
    import torch

    sd = weights if isinstance(weights, dict) else torch.load(
        weights, map_location="cpu", weights_only=True)

    if model == "tinystories":
        from certinf.float_fwd import forward
    elif model == "gpt2":
        from certinf.gpt2_float import forward
    else:
        raise ValueError(f"unknown model: {model!r}")

    with torch.no_grad():
        logits = forward(sd, list(token_ids), torch.float32)
    return logits[-1]


def top1(model: str, weights, token_ids: list[int]) -> int:
    """Pinned float32 top-1 argmax over the full vocabulary, last position."""
    return int(_forward_last_logits(model, weights, token_ids).argmax())


def determinism_check(model: str, weights, token_ids: list[int],
                      repeats: int = 5) -> bool:
    """Run the pinned harness `repeats` times; True iff every top-1 matches.

    CPU float32 with no dropout/randomness is expected to be bit-for-bit
    deterministic; a False here is the A1 CI-D kill signal.
    """
    import torch

    sd = weights if isinstance(weights, dict) else torch.load(
        weights, map_location="cpu", weights_only=True)
    tops = [top1(model, sd, token_ids) for _ in range(repeats)]
    return len(set(tops)) == 1


def _sha256_file(path: str) -> str:
    """sha256 of a file's bytes (streamed) — the checkpoint hash for the
    implementation transcript."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_command_line(model: str, weights: str, corpus: str,
                            context_length: int) -> str:
    """Environment-independent transcript command line: basenames only, so the
    transcript sha does not depend on absolute cache/checkout paths. This is
    the exact form recorded in docs/prereg-dryrun.md §4."""
    import os
    return (f"python3.11 -m certinf.harness --model {model} "
            f"--weights {os.path.basename(weights)} "
            f"--corpus {os.path.basename(corpus)} "
            f"--context-length {context_length}")


def run_determinism_gate(model: str, weights_path: str, corpus_path: str,
                         context_length: int, prompt_indices: list[int],
                         reps: int = 5, command_line: str | None = None,
                         out=None) -> tuple[bool, str, dict]:
    """The A1 CI-D determinism gate at frozen conditions.

    Loads the pinned checkpoint once, prints the implementation transcript and
    its sha256, then for each prompt index runs the pinned float32 harness
    `reps` times and reports whether the top-1 token is identical across
    repetitions. Returns `(all_deterministic, transcript_sha256, transcript)`.

    Args:
        model: "tinystories" or "gpt2".
        weights_path: path to the pinned `pytorch_model.bin`.
        corpus_path: committed token-id corpus JSON.
        context_length: window list to draw prompts from.
        prompt_indices: the pinned prompt indices to test.
        reps: repetitions per prompt (the A1 CI-D gate uses 5).
        command_line: transcript command line; defaults to the canonical
            basename form (environment-independent).
        out: file-like to print to (defaults to sys.stdout).

    Returns:
        (all_deterministic, transcript_sha256, transcript_dict).
    """
    import torch

    stream = out if out is not None else sys.stdout
    checkpoint_sha256 = _sha256_file(weights_path)
    if command_line is None:
        command_line = _canonical_command_line(
            model, weights_path, corpus_path, context_length)

    t = transcript(checkpoint_sha256=checkpoint_sha256, command_line=command_line)
    t_sha = hashlib.sha256(
        json.dumps(t, separators=(",", ":"), sort_keys=True).encode()).hexdigest()

    corpus_doc = corpus_mod.load(corpus_path)
    windows = corpus_doc["windows"].get(str(context_length))
    if windows is None:
        raise ValueError(f"corpus {corpus_path!r} has no windows at "
                         f"context_length={context_length}")

    print("### implementation transcript", file=stream)
    print(json.dumps(t, indent=1, sort_keys=True), file=stream)
    print(f"transcript_sha256: {t_sha}", file=stream)
    print("", file=stream)

    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    print(f"=== determinism gate: {reps} repetitions x {len(prompt_indices)} "
          f"distinct prompts (ctx={context_length}) ===", file=stream)
    all_det = True
    for pi in prompt_indices:
        ids = windows[pi]
        tops = [top1(model, sd, ids) for _ in range(reps)]
        det = len(set(tops)) == 1
        all_det = all_det and det
        verdict = "DETERMINISTIC" if det else "NONDETERMINISTIC"
        print(f"  prompt_index={pi:>2}  top1 x{reps} = {tops}  -> {verdict}",
              file=stream)
    print("", file=stream)
    if all_det:
        print("VERDICT: DETERMINISTIC across all prompts (A1 CI-D gate PASSES)",
              file=stream)
    else:
        print("VERDICT: NONDETERMINISTIC — A1 CI-D KILL (phi2_joint demotes to "
              "measured conformance)", file=stream)
    return all_det, t_sha, t


def main(argv: list[str] | None = None) -> int:
    """CLI: run the pinned-harness determinism gate and print per-rep top-1s
    and the transcript sha. Exit 0 iff deterministic across all prompts."""
    import argparse

    p = argparse.ArgumentParser(
        prog="python3.11 -m certinf.harness",
        description="Pinned float32 harness determinism gate (A1 CI-D).")
    p.add_argument("--model", required=True, choices=["tinystories", "gpt2"])
    p.add_argument("--weights", required=True,
                   help="path to the pinned pytorch_model.bin")
    p.add_argument("--corpus", required=True,
                   help="committed token-id corpus JSON")
    p.add_argument("--context-length", type=int, required=True, dest="context_length")
    p.add_argument("--prompt-index", type=int, action="append",
                   dest="prompt_indices", required=True,
                   help="prompt index to test (repeatable)")
    p.add_argument("--reps", type=int, default=5,
                   help="repetitions per prompt (default 5)")
    p.add_argument("--command-line", default=None, dest="command_line",
                   help="override the transcript command line (default: "
                        "canonical basename form)")
    args = p.parse_args(argv)

    all_det, _t_sha, _t = run_determinism_gate(
        model=args.model, weights_path=args.weights, corpus_path=args.corpus,
        context_length=args.context_length, prompt_indices=args.prompt_indices,
        reps=args.reps, command_line=args.command_line)
    return 0 if all_det else 1


if __name__ == "__main__":
    sys.exit(main())
