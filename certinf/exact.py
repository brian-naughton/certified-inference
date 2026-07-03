"""Rigorous interval arithmetic core for the certified-grokking project.

The certified object is the Nanda et al. grokking checkpoint (a TinyTransformer
trained on modular addition, mod 113). This module provides the trusted
interval-arithmetic primitives — exact-rational interval endpoints, outward
rounding, and a rigorous `exp` enclosure — that later tasks use to prove (not
just observe) properties of the model's forward pass. The only transcendental
function on the decision path is `exp`, inside the attention softmax. Unlike
the sibling `vcirc` project (where `d_head` is a perfect square and the
attention scale `1/sqrt(d_head)` is therefore an exact rational), here
`d_head = 32` is not a perfect square: `1/sqrt(32)` is irrational, so this
module additionally provides `inv_sqrt_ival` — a rigorous rational enclosure
of `1/sqrt(n)` for positive integers `n` — to bound the attention scale.

Representation. Every interval endpoint is an integer in units of ``2**-P``
(``P = PRECISION`` bits): the value ``v`` is stored as the integer
``round(v * 2**P)``. All operations round **outward** (lower endpoints toward
-inf, upper endpoints toward +inf) using exact integer floor/ceil division, so
the result is always a rigorous enclosure of the true real value — just with a
bounded denominator (``2**P``) that keeps numerators from ballooning through
the layers. Increasing ``P`` only tightens the width.

This is a *verified enclosure* of exact-real quantities; outputs are rigorous
rational bounds. Pure standard library — integers and `fractions.Fraction`
only (no torch, no numpy).
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import List

PRECISION: int = 96          # bits after the binary point (dyadic precision)
_SCALE: int = 1 << PRECISION


def set_precision(p: int) -> None:
    """Set the dyadic precision (bits) used for outward rounding."""
    global PRECISION, _SCALE
    PRECISION = p
    _SCALE = 1 << p


def _ceil_div(a: int, b: int) -> int:
    """Ceil(a / b) for integer a and positive integer b."""
    return -((-a) // b)


class Ival:
    """A rigorous interval whose endpoints are integers in units of 2**-P.

    ``lo_i / 2**P <= true value <= hi_i / 2**P``.
    """

    __slots__ = ("lo_i", "hi_i")

    def __init__(self, lo_i: int, hi_i: int):
        self.lo_i = lo_i
        self.hi_i = hi_i

    @classmethod
    def point(cls, v: Fraction) -> "Ival":
        """Tightest interval around an exact rational (a weight/embedding)."""
        num, den = v.numerator, v.denominator
        lo = (num * _SCALE) // den
        hi = _ceil_div(num * _SCALE, den)
        return cls(lo, hi)

    def __add__(self, o: "Ival") -> "Ival":
        return Ival(self.lo_i + o.lo_i, self.hi_i + o.hi_i)

    def __sub__(self, o: "Ival") -> "Ival":
        return Ival(self.lo_i - o.hi_i, self.hi_i - o.lo_i)

    def __neg__(self) -> "Ival":
        return Ival(-self.hi_i, -self.lo_i)

    def mul_scalar(self, w: Fraction) -> "Ival":
        """Multiply by an exact rational scalar (a weight), rounding outward."""
        wn, wd = w.numerator, w.denominator     # wd > 0
        if wn >= 0:
            return Ival((self.lo_i * wn) // wd, _ceil_div(self.hi_i * wn, wd))
        return Ival((self.hi_i * wn) // wd, _ceil_div(self.lo_i * wn, wd))

    def __mul__(self, o: "Ival") -> "Ival":
        a, b, c, d = self.lo_i, self.hi_i, o.lo_i, o.hi_i
        p1, p2, p3, p4 = a * c, a * d, b * c, b * d   # units 2**-2P
        lo = min(p1, p2, p3, p4) >> PRECISION          # floor / 2**P
        hi = -((-max(p1, p2, p3, p4)) >> PRECISION)    # ceil  / 2**P
        return Ival(lo, hi)

    def relu(self) -> "Ival":
        return Ival(max(0, self.lo_i), max(0, self.hi_i))

    @property
    def lo(self) -> Fraction:
        return Fraction(self.lo_i, _SCALE)

    @property
    def hi(self) -> Fraction:
        return Fraction(self.hi_i, _SCALE)

    def width(self) -> Fraction:
        return Fraction(self.hi_i - self.lo_i, _SCALE)

    def max_bits(self) -> int:
        return max(self.lo_i.bit_length(), self.hi_i.bit_length())


# --------------------------------------------------------------------------- #
# rigorous exp enclosure (fixed-point), for an argument given in units of 2**-P
# --------------------------------------------------------------------------- #
def _exp_lower_nonneg(X: int) -> int:
    """Lower bound on exp(x), x = X/2**P >= 0; every term rounded DOWN.

    Term count is PRECISION-AWARE (2026-07-02 fix): the old fixed cap
    N = floor(x)+60 left a truncation tail x^(N+1)/(N+1)! that is independent
    of P, giving the enclosure a precision-independent relative-width floor
    (~6e-33 at |x|=11.6 — measured identical at P=128/192/256). Sound but not
    P-parametric; at P >= ~128 it dominated the ulp error and became the width
    bottleneck. With N = floor(x)+60+P the tail sits below one ulp for every
    argument the guards admit (the loop self-truncates far earlier for small
    x, so typical cost grows only mildly). Adding terms only TIGHTENS the
    lower bound (all terms are >= 0 and floored), so soundness is unchanged.
    """
    N = (X >> PRECISION) + 60 + PRECISION
    term = _SCALE
    S = _SCALE
    for k in range(1, N + 1):
        term = (term * X) // (k * _SCALE)     # x^k/k!, floored
        if term == 0:
            break                             # omitted terms are >= 0
        S += term
    return S


def _exp_upper_nonneg(X: int) -> int:
    """Upper bound on exp(x), x = X/2**P >= 0; terms rounded UP + tail bound.

    PRECISION-AWARE term count + rigorous early tail cut (2026-07-02 fix; see
    _exp_lower_nonneg for the failure mode of the old fixed N = floor(x)+60).

    Early cut soundness: t_k (the running ceiled term, integer ulps) satisfies
    tau_k <= t_k by induction (ceil >= exact at every step), where tau_k =
    x^k/k! in ulps. Once t_k <= 1 and r := x/(k+1) <= 1/2, the true remaining
    tail is sum_{j>=1} tau_k * r^j <= tau_k * r/(1-r) <= tau_k <= 1 ulp, so
    adding one ulp and stopping is a valid upper bound. (t_k == 0 forces
    tau_k = 0 and hence a zero tail: x^k/k! is nonincreasing once k > x, and
    tau_k = 0 exactly only when x = 0.) Without the cut the ceiled term never
    reaches 0 (ceil of a positive quantity is >= 1), so the old loop always
    ran all N iterations and accumulated ~N spurious ulps.
    """
    N = (X >> PRECISION) + 60 + PRECISION
    term = _SCALE
    S = _SCALE
    for k in range(1, N + 1):
        term = _ceil_div(term * X, k * _SCALE)
        S += term
        if term == 0:
            return S                          # exact-zero tail (x == 0)
        if term <= 1 and 2 * X <= (k + 1) * _SCALE:
            return S + 1                      # remaining tail <= 1 ulp
    next_term = _ceil_div(term * X, (N + 1) * _SCALE)
    ratio_up = _ceil_div(X, N + 2)            # x/(N+2) in units 2**-P, ceil
    denom = _SCALE - ratio_up                 # (1 - ratio) in units 2**-P
    assert denom > 0, "exp tail ratio >= 1; increase term count"
    tail = _ceil_div(next_term * _SCALE, denom)
    return S + tail


def exp_bounds(X: int):
    """Rigorous [lo_i, hi_i] (units 2**-P) enclosing exp(X/2**P) for any int X."""
    if X >= 0:
        return _exp_lower_nonneg(X), _exp_upper_nonneg(X)
    lo_p = _exp_lower_nonneg(-X)              # 0 < lo_p <= exp(-x)*2**P <= hi_p
    hi_p = _exp_upper_nonneg(-X)
    return (_SCALE * _SCALE) // hi_p, _ceil_div(_SCALE * _SCALE, lo_p)   # 1/exp(-x)


def exp_ival(a: Ival) -> Ival:
    """Enclose exp over an interval (exp is monotone increasing)."""
    lo, _ = exp_bounds(a.lo_i)
    _, hi = exp_bounds(a.hi_i)
    return Ival(lo, hi)


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def _F(v) -> Fraction:
    """Exact Fraction from a float (dyadic), a [num, den] pair, or a Fraction."""
    if isinstance(v, Fraction):
        return v
    if isinstance(v, (list, tuple)):
        return Fraction(int(v[0]), int(v[1]))
    return Fraction(v)  # float -> exact dyadic rational


def _mat(rows) -> List[List[Fraction]]:
    return [[_F(x) for x in row] for row in rows]


def _vec(xs) -> List[Fraction]:
    return [_F(x) for x in xs]


def _hex_to_float(x):
    if isinstance(x, str):
        return float.fromhex(x)
    return [_hex_to_float(e) for e in x]


def inv_sqrt_ival(n: int) -> Ival:
    """Rigorous enclosure of 1/sqrt(n) for a positive integer n.

    sqrt(n) is enclosed by r/2**P <= sqrt(n) <= (r+1)/2**P where
    r = isqrt(n * 4**P); the reciprocal endpoints are then rounded outward.
    Needed because the attention scale 1/sqrt(d_head)=1/sqrt(32) is
    irrational (vcirc's perfect-square assertion does not hold here).
    """
    assert n > 0
    r = isqrt(n * _SCALE * _SCALE)      # floor(sqrt(n) * 2**P)
    # 1/sqrt(n): lower endpoint uses the UPPER sqrt bound and floors;
    # upper endpoint uses the LOWER sqrt bound and ceils.
    lo = (_SCALE * _SCALE) // (r + 1)
    hi = _ceil_div(_SCALE * _SCALE, r)
    return Ival(lo, hi)
