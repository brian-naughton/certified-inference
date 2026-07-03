import os

from certinf import harness
from certinf.float_fwd import MODEL_BIN, PROMPT_IDS

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TS_VAL_CORPUS = os.path.join(
    _REPO_ROOT, "certificates", "corpora", "tinystories-val.ids.json")


def test_transcript_has_all_a1_keys_and_flags():
    t = harness.transcript()
    for key in ("platform", "os", "python", "torch_version",
               "deterministic_flags", "tf32", "eval_mode",
               "checkpoint_sha256", "command_line"):
        assert key in t, key
    assert t["tf32"] is False
    assert t["eval_mode"] is True


def test_transcript_sha256_stable_within_a_run():
    a = harness.transcript_sha256(checkpoint_sha256="abc123")
    b = harness.transcript_sha256(checkpoint_sha256="abc123")
    assert a == b


def test_determinism_check_true_on_foothold_prompt():
    assert harness.determinism_check("tinystories", MODEL_BIN, PROMPT_IDS,
                                     repeats=5) is True


def test_top1_returns_int():
    tok = harness.top1("tinystories", MODEL_BIN, PROMPT_IDS)
    assert isinstance(tok, int)
    assert 0 <= tok < 50257


def test_cli_determinism_gate_runs_and_reports_deterministic(capsys):
    """The argparse CLI runs the determinism gate over pinned ctx-16 prompts,
    prints the transcript sha and per-rep top-1s, and exits 0 when
    deterministic. Uses the canonical basename command line, so the transcript
    sha is the environment-independent one recorded in docs/prereg-dryrun.md."""
    rc = harness.main([
        "--model", "tinystories",
        "--weights", MODEL_BIN,
        "--corpus", _TS_VAL_CORPUS,
        "--context-length", "16",
        "--prompt-index", "0",
        "--prompt-index", "1",
        "--reps", "3",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "transcript_sha256:" in out
    assert "top1 x3" in out
    assert "A1 CI-D gate PASSES" in out
    # canonical basename command line => the pinned transcript sha
    assert "b4e2fa62eb3a242a4993ede2fde10be2d39b1aa166da06ebeb6d790f085f0358" in out
