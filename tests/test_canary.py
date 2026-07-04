import pytest

torch = pytest.importorskip("torch")  # noqa: F841 -- canary._run loads via the torch loader

from certinf import canary  # noqa: E402


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
    """I3(chunk2) regression, adapted for Task 0.2's precision-keyed cache
    (certinf/ival_ext.py): sqrt(2/pi) is now memoised per precision in
    E._GELU_C_CACHE, so a value computed at one precision can never leak into
    a run at another -- there is no single-slot cache left to poison or
    invalidate. The old module-global E._GELU_C is now a dead backward-compat
    shim: call sites (including canary._run) still poke it, but poking it has
    no effect on the actual computation. Confirm both: (a) poking the shim
    does not perturb the real per-precision cache, and (b) canary._run at
    P=64 populates/uses the CURRENT-precision constant, distinct from the
    P=32 one computed moments earlier."""
    import certinf.ival_ext as E
    from certinf import exact

    def endpoints(iv):
        return (iv.lo_i, iv.hi_i)

    ids = [7454, 2402, 257, 640]
    exact.set_precision(32)
    stale = endpoints(E.gelu_const_ival())   # the P=32 constant's endpoints
    E._GELU_C = "poisoned-dead-shim"         # poking the shim must be inert

    canary._run("tinystories", ids, P=64)

    exact.set_precision(64)
    assert 64 in E._GELU_C_CACHE
    # Ival has no __eq__ (identity only), so compare endpoints explicitly.
    assert endpoints(E._GELU_C_CACHE[64]) != stale                    # not the P=32 value
    assert endpoints(E._GELU_C_CACHE[64]) == endpoints(E.gelu_const_ival())  # current P
