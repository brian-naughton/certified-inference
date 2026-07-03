from certinf import harness
from certinf.float_fwd import MODEL_BIN, PROMPT_IDS


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
