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
