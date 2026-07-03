import pytest

from certinf import canary


@pytest.mark.canary
def test_no_precision_floor_tinystories_small_P():
    # cheap seq-4 prompt at P=64 vs 128 — must show widths tracking 2^-P
    ratios = canary.assert_no_precision_floor("tinystories", ids=[7454, 2402, 257, 640], P=64)
    assert all(r["ok"] for r in ratios["per_sublayer"])


@pytest.mark.canary
def test_canary_catches_injected_floor(monkeypatch):
    """Monkeypatch a precision-independent floor into a sublayer width and
    confirm the canary trips (guards the guard)."""
    import certinf.canary as C
    orig = C.width_profile

    def floored(model, ids, P):
        prof = orig(model, ids, P)
        prof["neg_log2"]["L0.ln_1"] = 100.0     # frozen: never improves with P
        return prof
    monkeypatch.setattr(C, "width_profile", floored)
    with pytest.raises(AssertionError, match="L0.ln_1"):
        C.assert_no_precision_floor("tinystories", ids=[7454, 2402, 257, 640], P=64)
