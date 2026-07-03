#!/usr/bin/env python3
"""Extensions to the audited interval core (exact.py) for LayerNorm/GELU/softmax.

All operations round outward in the same fixed-point style as exact.py:
endpoints are integers in units of 2**-P, lower endpoints floor, upper
endpoints ceil, so every result is a rigorous enclosure of the exact-real
quantity. SCRATCH research code: rigour lives in the rounding directions.
"""
from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import List, Sequence, Tuple

from certinf import exact
from certinf.exact import Ival, _ceil_div, exp_bounds


def _S() -> int:
    return exact._SCALE


def _P() -> int:
    return exact.PRECISION


# --------------------------------------------------------------------------- #
# basic extensions
# --------------------------------------------------------------------------- #
def square_ival(a: Ival) -> Ival:
    """Enclosure of x**2 (tighter than a*a: clamps at 0 when a spans 0)."""
    P = _P()
    lo, hi = a.lo_i, a.hi_i
    if lo >= 0:
        return Ival((lo * lo) >> P, -((-(hi * hi)) >> P))
    if hi <= 0:
        return Ival((hi * hi) >> P, -((-(lo * lo)) >> P))
    m = max(lo * lo, hi * hi)
    return Ival(0, -((-m) >> P))


def div_ival(a: Ival, b: Ival) -> Ival:
    """a / b for b strictly positive (b.lo_i > 0), outward."""
    assert b.lo_i > 0, "div_ival requires strictly positive denominator"
    S = _S()
    cands_num = (a.lo_i, a.hi_i)
    # a/b in units 2**-P: (a_i * S) / b_i
    lo = min((x * S) // (b.hi_i if x >= 0 else b.lo_i) for x in cands_num)
    hi = max(_ceil_div(x * S, b.lo_i if x >= 0 else b.hi_i)
             for x in cands_num)
    return Ival(lo, hi)


def mean_ival(xs: Sequence[Ival]) -> Ival:
    """Interval mean: exact interval sum, then mul by exact Fraction 1/n."""
    lo = sum(x.lo_i for x in xs)
    hi = sum(x.hi_i for x in xs)
    n = len(xs)
    return Ival(lo // n, _ceil_div(hi, n))


def var_ival(xs: Sequence[Ival], mu: Ival) -> Ival:
    """Biased variance E[(x-mu)^2]; mu is an interval (conservative)."""
    n = len(xs)
    lo = 0
    hi = 0
    for x in xs:
        s = square_ival(x - mu)
        lo += s.lo_i
        hi += s.hi_i
    return Ival(lo // n, _ceil_div(hi, n))


def inv_sqrt_of_ival(a: Ival) -> Ival:
    """Rigorous enclosure of 1/sqrt(y) for y in the interval a, a.lo_i > 0.

    1/sqrt is monotone decreasing, so the lower result endpoint comes from the
    UPPER argument endpoint and vice versa. With y = Y/2**P,
    1/sqrt(y) * 2**P = sqrt(2**(3P) / Y):
      lower: isqrt(floor(2**(3P)/Y_hi))          (<= true)
      upper: isqrt(ceil (2**(3P)/Y_lo)) + 1      (>= true)
    """
    assert a.lo_i > 0, "inv_sqrt_of_ival requires positive interval"
    P = _P()
    big = 1 << (3 * P)
    lo = isqrt(big // a.hi_i)
    hi = isqrt(_ceil_div(big, a.lo_i)) + 1
    return Ival(lo, hi)


def sqrt_frac_ival(q: Fraction) -> Ival:
    """Rigorous enclosure of sqrt(q) for a positive rational q."""
    assert q > 0
    P = _P()
    num, den = q.numerator, q.denominator
    t = num << (2 * P)
    lo = isqrt(t // den)
    hi = isqrt(_ceil_div(t, den)) + 1
    return Ival(lo, hi)


# --------------------------------------------------------------------------- #
# tanh / gelu_new
# --------------------------------------------------------------------------- #
# Rational UPPER bound on ln2.  2977044472 / 2**32 = 0.6931471806019...  which
# is strictly greater than ln2 = 0.6931471805599...  (verified: the difference
# is ~4.2e-11 > 0).  Deriving every precision-aware guard from an *upper* bound
# on ln2 makes the guards conservative (they fire slightly later than the exact
# threshold) at every precision P, so the fast paths can never be inward.
_LN2_UP_NUM = 2977044472
_LN2_UP_DEN = 4294967296          # == 1 << 32


def exp_guard_threshold() -> int:
    """Integer X threshold T (units 2**-P): the exp fast path fires at X <= -T."""
    return _ceil_div((_P() + 2) * _LN2_UP_NUM * _S(), _LN2_UP_DEN)


def tanh_guard_threshold() -> int:
    """Integer X threshold T (units 2**-P): tanh saturates at |X| >= T."""
    return _ceil_div((_P() + 2) * _LN2_UP_NUM * _S(), 2 * _LN2_UP_DEN)


# --- artifact instrumentation: track guard-relevant endpoint ranges -------- #
# So guard activation is auditable from the width JSON (requested during adversarial review). Plain
# module-level accumulators; interval_fwd resets them per layer and snapshots.
_TRACK = {}


def reset_tracking() -> None:
    """Clear the per-region endpoint/guard-hit accumulators."""
    _TRACK.clear()
    _TRACK.update(softmax_shift_min=None, softmax_shift_max=None,
                  gelu_inner_min=None, gelu_inner_max=None,
                  exp_guard_hits=0, tanh_guard_hits=0)


def get_tracking() -> dict:
    """Snapshot of the accumulators (endpoints in units 2**-P, hits as counts)."""
    return dict(_TRACK)


def _track_minmax(key_min: str, key_max: str, val: int) -> None:
    lo = _TRACK.get(key_min)
    hi = _TRACK.get(key_max)
    if lo is None or val < lo:
        _TRACK[key_min] = val
    if hi is None or val > hi:
        _TRACK[key_max] = val


reset_tracking()


def exp_bounds_safe(X: int) -> Tuple[int, int]:
    """exp_bounds with a precision-aware, still-rigorous path for very negative
    args.

    The interval upper endpoint hi_i = 1 represents 2**-P (one ulp), so
    returning (0, 1) is a valid outward enclosure of exp(x) iff
    exp(x) <= 2**-P, i.e. iff  x <= -P*ln2.  We use a stronger guard with two
    bits of margin,  x <= -(P+2)*ln2,  which gives exp(x) <= 2**-(P+2) < 2**-P.

    Derivation of the integer threshold (X = x*2**P, S = 2**P):
      want   x <= -(P+2)*ln2                          (real inequality)
      i.e.   X <= -(P+2)*ln2*S.
    Let  T = ceil( (P+2) * LN2_UP_NUM * S / LN2_UP_DEN ).  Because
    LN2_UP := LN2_UP_NUM/LN2_UP_DEN > ln2,
      T >= (P+2)*LN2_UP*S > (P+2)*ln2*S,
    so the tested condition  X <= -T  provably implies  X < -(P+2)*ln2*S,
    hence x < -(P+2)*ln2 and exp(x) < 2**-(P+2) < 2**-P.  Conservative at every
    precision.  (The old fixed cutoff X <= -70*S was sound only at P<=~96 and
    was INWARD for P>~101; e.g. exp(-70)~=4e-31 >> 2**-192.)
    """
    T = exp_guard_threshold()
    if X <= -T:
        _TRACK["exp_guard_hits"] = _TRACK.get("exp_guard_hits", 0) + 1
        return (0, 1)
    return exp_bounds(X)


def _tanh_endpoint(X: int, round_up: bool) -> int:
    """Bound tanh(X/2**P) = 1 - 2/(exp(2x)+1) in units 2**-P.

    Monotone increasing in exp(2x). round_up=False: use exp LOWER bound and
    ceil the subtracted term (result <= true). round_up=True: exp UPPER bound,
    floor the subtracted term (result >= true).
    """
    S = _S()
    # Precision-aware saturation guard.  The shortcut endpoints S-1 (= 1-2**-P)
    # and S (= 1) form a valid outward pair only when 2/(exp(2x)+1) <= 2**-P
    # (so that tanh(x) >= 1-2**-P).  That holds once  x >= (P+2)*ln2/2, because
    # then exp(2x) >= 2**(P+2), hence 2/(exp(2x)+1) < 2/exp(2x) <= 2**-(P+1) <
    # 2**-P.  In integer units the exact threshold is X >= (P+2)*ln2*S/2; with
    # the rational UPPER bound ln2 < LN2_UP,
    #   T = ceil( (P+2)*LN2_UP_NUM*S / (2*LN2_UP_DEN) ) >= (P+2)*LN2_UP*S/2
    #        > (P+2)*ln2*S/2,
    # so  X >= T  provably implies  x > (P+2)*ln2/2  and the shortcut is sound.
    # Symmetric for the negative tail.  (The old fixed cutoff X >= 48*S was
    # sound only at P<=~137 and was INWARD above that, e.g. tanh(48)=1-~4e-42
    # while 1-2**-192 is closer to 1.)  Below the threshold we fall through to
    # the rigorous exp path, whose loop stays bounded (|2x| < (P+2)*ln2).
    T = tanh_guard_threshold()
    if X >= T:
        _TRACK["tanh_guard_hits"] = _TRACK.get("tanh_guard_hits", 0) + 1
        return S - 1 if not round_up else S
    if X <= -T:
        _TRACK["tanh_guard_hits"] = _TRACK.get("tanh_guard_hits", 0) + 1
        return -S if not round_up else -S + 1
    e_lo, e_hi = exp_bounds(2 * X)
    if round_up:
        return S - ((2 * S * S) // (e_hi + S))
    return S - _ceil_div(2 * S * S, e_lo + S)


def tanh_ival(a: Ival) -> Ival:
    """tanh over an interval (monotone increasing), outward."""
    return Ival(_tanh_endpoint(a.lo_i, round_up=False),
                _tanh_endpoint(a.hi_i, round_up=True))


_GELU_A = Fraction(0.044715)  # exact dyadic of the double used by gelu_new


def _atan_inv_bounds(n: int, tol: Fraction) -> Tuple[Fraction, Fraction]:
    """Rigorous rational bounds on arctan(1/n), gap <= tol (n >= 2).

    arctan(1/n) = sum_{k>=0} (-1)^k / ((2k+1) n^(2k+1)) is alternating with
    strictly decreasing terms, so consecutive partial sums bracket the limit:
    after an even-k (positive) term the partial sum is an UPPER bound, after
    an odd-k term a LOWER bound. Exact Fraction arithmetic throughout.
    """
    s = Fraction(0)
    lo, hi = None, None
    k = 0
    while True:
        t = Fraction(1, (2 * k + 1) * n ** (2 * k + 1))
        if k % 2 == 0:
            s += t
            hi = s
        else:
            s -= t
            lo = s
        if lo is not None and hi - lo <= tol:
            return lo, hi
        k += 1


def pi_bounds() -> Tuple[Fraction, Fraction]:
    """Rigorous rational enclosure of pi, width ~2**-(2P+16), via Machin:
    pi = 16*arctan(1/5) - 4*arctan(1/239) (exact identity), with the two
    arctan enclosures combined outward.

    PRECISION-AWARE (2026-07-02 fix): the old hard-coded 61-digit pi string
    gave sqrt(2/pi) an enclosure width ~4e-61 — a precision-INDEPENDENT floor
    that was invisible below P~200 but capped every GELU at ~200 effective
    bits at P=384 (observed: GPT-2 L0 gelu widths 4e-62 instead of ~1e-110).
    """
    tol = Fraction(1, 1 << (2 * _P() + 16))
    a5_lo, a5_hi = _atan_inv_bounds(5, tol / 64)
    a239_lo, a239_hi = _atan_inv_bounds(239, tol / 16)
    return 16 * a5_lo - 4 * a239_hi, 16 * a5_hi - 4 * a239_lo


def gelu_const_ival() -> Ival:
    """Enclosure of sqrt(2/pi): 2/pi in [2/pi_hi, 2/pi_lo], sqrt outward."""
    pi_lo, pi_hi = pi_bounds()
    lo = sqrt_frac_ival(2 / pi_hi).lo_i
    hi = sqrt_frac_ival(2 / pi_lo).hi_i
    return Ival(lo, hi)


_GELU_C: Ival | None = None
_HALF = Fraction(1, 2)


def gelu_new_ival(x: Ival) -> Ival:
    """gelu_new(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))."""
    global _GELU_C
    if _GELU_C is None:
        _GELU_C = gelu_const_ival()
    x3 = square_ival(x) * x
    inner = (x + x3.mul_scalar(_GELU_A)) * _GELU_C
    # record the tanh-input endpoint range so tanh-saturation guard activation
    # is auditable from artifacts (requested during adversarial review)
    _track_minmax("gelu_inner_min", "gelu_inner_max", inner.lo_i)
    _track_minmax("gelu_inner_min", "gelu_inner_max", inner.hi_i)
    t = tanh_ival(inner)
    one = Ival(_S(), _S())
    return (x * (one + t)).mul_scalar(_HALF)


# --------------------------------------------------------------------------- #
# layernorm / softmax
# --------------------------------------------------------------------------- #
def layernorm_ival(xs: Sequence[Ival], gamma: Sequence[Fraction],
                   beta: Sequence[Fraction], eps: Fraction) -> List[Ival]:
    """(x - mu) * invsqrt(var + eps) * gamma + beta, all outward."""
    mu = mean_ival(xs)
    var = var_ival(xs, mu)
    inv = inv_sqrt_of_ival(var + Ival.point(eps))
    out = []
    for x, g, b in zip(xs, gamma, beta):
        out.append(((x - mu) * inv).mul_scalar(g) + Ival.point(b))
    return out


def softmax_ival(scores: Sequence[Ival]) -> List[Ival]:
    """Softmax over unmasked scores with an exact max-shift (width-free).

    Shifting every score by the same integer constant is exact for softmax and
    keeps exp arguments <= 0-ish, so fixed-point exp stays well-conditioned.
    """
    m = max(s.hi_i for s in scores)
    # record the shifted softmax-argument range so exp-underflow guard
    # activation is auditable from artifacts (requested during adversarial review)
    for s in scores:
        _track_minmax("softmax_shift_min", "softmax_shift_max", s.lo_i - m)
        _track_minmax("softmax_shift_min", "softmax_shift_max", s.hi_i - m)
    es = [Ival(exp_bounds_safe(s.lo_i - m)[0],
               exp_bounds_safe(s.hi_i - m)[1]) for s in scores]
    denom = Ival(sum(e.lo_i for e in es), sum(e.hi_i for e in es))
    S = _S()
    unit = Ival(0, S)
    out = []
    for e in es:
        if denom.lo_i > 0:
            iv = div_ival(e, denom)
            # softmax outputs provably lie in [0,1]: intersecting is rigorous
            iv = Ival(max(iv.lo_i, 0), min(iv.hi_i, S))
        else:
            # widths so large the denominator's lower bound collapsed to 0;
            # [0,1] is still a rigorous enclosure — degrade, don't crash
            iv = unit
        out.append(iv)
    return out
