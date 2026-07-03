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


@pytest.mark.canary
def test_no_precision_floor_tinystories_grid_precision():
    """P=128 vs 256 — the actual grid precisions. The historical pi-string
    floor lived at ~2^-201, invisible at the P=64/128 pair above but reachable
    here; a short seq-4 prompt keeps runtime sane."""
    ratios = canary.assert_no_precision_floor(
        "tinystories", ids=[7454, 2402, 257, 640], P=128)
    assert all(r["ok"] for r in ratios["per_sublayer"])


@pytest.mark.canary
def test_canary_trips_on_realistic_finite_floor(monkeypatch):
    """Inject a realistic finite precision-independent floor at ~2^-107 (a
    plausible fixed-precision-constant floor) into a sublayer width and confirm
    the gate trips at the standard margin=8 — stronger than the frozen 100.0
    stub because the magnitude is one a real constant-bug would produce."""
    import certinf.canary as C
    orig = C.width_profile
    FLOOR = 107.0     # -log2(2^-107): width never shrinks past this as P grows

    def floored(model, ids, P):
        prof = orig(model, ids, P)
        prof["neg_log2"]["L0.gelu"] = FLOOR
        return prof

    monkeypatch.setattr(C, "width_profile", floored)
    with pytest.raises(AssertionError, match="L0.gelu"):
        C.assert_no_precision_floor(
            "tinystories", ids=[7454, 2402, 257, 640], P=64, margin=8)


@pytest.mark.canary
def test_canary_run_resets_stale_gelu_constant():
    """I3(chunk2) regression: canary._run must invalidate the precision-keyed
    sqrt(2/pi) GELU cache after set_precision, else the 2P forward reuses the
    P-bit constant (a self-injected floor). Poison the cache at P=32, run at
    P=64, and confirm the run recomputed the constant at the current
    precision."""
    import certinf.ival_ext as E
    from certinf import exact

    def endpoints(iv):
        return (iv.lo_i, iv.hi_i)

    ids = [7454, 2402, 257, 640]
    exact.set_precision(32)
    E._GELU_C = None
    stale = endpoints(E.gelu_const_ival())   # the P=32 constant's endpoints
    E._GELU_C = E.gelu_const_ival()          # poison the cache with it

    canary._run("tinystories", ids, P=64)

    exact.set_precision(64)
    assert E._GELU_C is not None
    # Ival has no __eq__ (identity only), so compare endpoints explicitly.
    assert endpoints(E._GELU_C) != stale                    # not the P=32 value
    assert endpoints(E._GELU_C) == endpoints(E.gelu_const_ival())  # current P
