#!/usr/bin/env python3
"""Spot-checks: each extension must ENCLOSE the float reference value."""
import math
import random
from fractions import Fraction

from exact import Ival
import ival_ext as E


def ival_of(v: float) -> Ival:
    return Ival.point(Fraction(v))


def contains(iv: Ival, v: float, tol: float = 1e-12) -> bool:
    return float(iv.lo) <= v + tol and float(iv.hi) >= v - tol


random.seed(0)
ok = True

# tanh
for x in [-50, -5, -0.3, 0.0, 0.7, 3.0, 60.0]:
    iv = E.tanh_ival(ival_of(x))
    good = contains(iv, math.tanh(x)) and float(iv.width()) < 1e-20
    ok &= good
    print(f"tanh({x}): [{float(iv.lo):.15f},{float(iv.hi):.15f}] "
          f"ref {math.tanh(x):.15f} {'OK' if good else 'FAIL'}")

# gelu_new
def gelu_ref(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

for x in [-4.0, -1.0, -0.1, 0.0, 0.5, 2.0, 8.0]:
    iv = E.gelu_new_ival(ival_of(x))
    good = contains(iv, gelu_ref(x)) and float(iv.width()) < 1e-18
    ok &= good
    print(f"gelu({x}): width {float(iv.width()):.2e} ref {gelu_ref(x):.12f} "
          f"{'OK' if good else 'FAIL'}")

# inv_sqrt over interval
iv = E.inv_sqrt_of_ival(Ival.point(Fraction(2)))
good = contains(iv, 1 / math.sqrt(2))
ok &= good
print("invsqrt(2):", float(iv.lo), float(iv.hi), "OK" if good else "FAIL")
a = Ival.point(Fraction(3)) - Ival.point(Fraction(1, 7))  # [2.857..]
iv = E.inv_sqrt_of_ival(a)
good = contains(iv, 1 / math.sqrt(3 - 1 / 7))
ok &= good
print("invsqrt(20/7):", float(iv.lo), float(iv.hi), "OK" if good else "FAIL")

# layernorm vs float reference
xs = [random.uniform(-3, 3) for _ in range(64)]
g = [random.uniform(0.5, 1.5) for _ in range(64)]
b = [random.uniform(-0.2, 0.2) for _ in range(64)]
mu = sum(xs) / 64
var = sum((v - mu) ** 2 for v in xs) / 64
ref = [(v - mu) / math.sqrt(var + 1e-5) * gi + bi for v, gi, bi in zip(xs, g, b)]
out = E.layernorm_ival([ival_of(v) for v in xs],
                       [Fraction(v) for v in g], [Fraction(v) for v in b],
                       Fraction(1e-5))
wmax = max(float(o.width()) for o in out)
good = all(contains(o, r) for o, r in zip(out, ref)) and wmax < 1e-20
ok &= good
print(f"layernorm: max width {wmax:.2e} encloses ref: {'OK' if good else 'FAIL'}")

# softmax
ss = [random.uniform(-8, 8) for _ in range(8)]
es = [math.exp(v - max(ss)) for v in ss]
ref = [v / sum(es) for v in es]
out = E.softmax_ival([ival_of(v) for v in ss])
good = all(contains(o, r) for o, r in zip(out, ref))
good &= all(float(o.width()) < 1e-20 for o in out)
ok &= good
print("softmax:", "OK" if good else "FAIL",
      "sum range", sum(float(o.lo) for o in out), sum(float(o.hi) for o in out))

# --------------------------------------------------------------------------- #
# Multi-precision guard soundness (requested during adversarial review).
#
# The P=96 spot-checks above cannot catch a one-ulp inward error at higher
# precision: `math.exp`/`math.tanh` saturate long before 2**-192.  These tests
# drive exp_bounds_safe and tanh_ival at P in {96,128,160,192,256}, at points
# straddling BOTH the old fixed thresholds (-70, +/-48) and the new
# precision-aware ones, and assert the returned interval ENCLOSES high-precision
# truth.  Truth is taken from mpmath at prec >> P if available; otherwise we use
# the self-consistency fallback the review allowed: the enclosure at P must
# contain the enclosure computed at P+64 (a tighter, still-rigorous bracket).
# --------------------------------------------------------------------------- #
import exact

try:
    import mpmath
    _HAVE_MPMATH = True
except ImportError:
    _HAVE_MPMATH = False


def _enclose_exp(P, X):
    """(lo, hi) rationals enclosing exp(X/2**P). mpmath if present, else P+64."""
    if _HAVE_MPMATH:
        mpmath.mp.prec = P + 200
        v = mpmath.e ** (mpmath.mpf(X) / (mpmath.mpf(2) ** P))
        # bracket the mpf both sides by a generous margin (2**-(P+100))
        m = mpmath.mpf(2) ** (-(P + 100))
        return (Fraction(mpmath.nstr(v - m, P // 3 + 40)),
                Fraction(mpmath.nstr(v + m, P // 3 + 40)))
    exact.set_precision(P + 64)
    lo, hi = E.exp_bounds_safe(X << 64)   # same real arg at higher precision
    exact.set_precision(P)
    S2 = 1 << (P + 64)
    return (Fraction(lo, S2), Fraction(hi, S2))


def _enclose_tanh(P, X):
    if _HAVE_MPMATH:
        mpmath.mp.prec = P + 200
        v = mpmath.tanh(mpmath.mpf(X) / (mpmath.mpf(2) ** P))
        m = mpmath.mpf(2) ** (-(P + 100))
        return (Fraction(mpmath.nstr(v - m, P // 3 + 40)),
                Fraction(mpmath.nstr(v + m, P // 3 + 40)))
    exact.set_precision(P + 64)
    iv = E.tanh_ival(Ival(X << 64, X << 64))
    exact.set_precision(P)
    S2 = 1 << (P + 64)
    return (Fraction(iv.lo_i, S2), Fraction(iv.hi_i, S2))


print("\n--- multi-precision guard soundness "
      f"({'mpmath' if _HAVE_MPMATH else 'P+64 self-consistency'}) ---")
for P in (96, 128, 160, 192, 256, 320, 384):
    exact.set_precision(P)
    E._GELU_C = None
    S = 1 << P
    exp_thr = E.exp_guard_threshold()     # guard fires at X <= -exp_thr
    tanh_thr = E.tanh_guard_threshold()   # guard fires at |X| >= tanh_thr

    # exp: near old -70 and near the new precision-aware threshold
    exp_Xs = [-70 * S, -50 * S,
              -exp_thr, -exp_thr - S, -exp_thr + 5 * S, -exp_thr - 20 * S]
    for X in exp_Xs:
        lo_i, hi_i = E.exp_bounds_safe(X)
        lo, hi = Fraction(lo_i, S), Fraction(hi_i, S)
        tlo, thi = _enclose_exp(P, X)
        mid = (tlo + thi) / 2            # ~ true exp(X/2**P)
        good = lo <= mid <= hi           # interval must ENCLOSE truth
        ok &= good
        if not good:
            print(f"  P={P} exp X/S={float(X)/S:+.2f}: "
                  f"iv=[{float(lo):.3e},{float(hi):.3e}] truth~[{float(tlo):.3e},"
                  f"{float(thi):.3e}] FAIL")

    # tanh: near old +/-48 and near the new precision-aware threshold
    tanh_Xs = [48 * S, -48 * S, 5 * S, tanh_thr, -tanh_thr,
               tanh_thr - 5 * S, -(tanh_thr - 5 * S), tanh_thr + 3 * S]
    for X in tanh_Xs:
        iv = E.tanh_ival(Ival(X, X))
        lo, hi = Fraction(iv.lo_i, S), Fraction(iv.hi_i, S)
        tlo, thi = _enclose_tanh(P, X)
        mid = (tlo + thi) / 2            # ~ true tanh(X/2**P)
        good = lo <= mid <= hi           # interval must ENCLOSE truth
        ok &= good
        if not good:
            print(f"  P={P} tanh X/S={float(X)/S:+.2f}: "
                  f"iv=[{float(lo):.6f},{float(hi):.6f}] "
                  f"truth~[{float(tlo):.6f},{float(thi):.6f}] FAIL")
    print(f"  P={P:3d}: exp_thr x={-float(exp_thr)/S:.2f} "
          f"tanh_thr x={float(tanh_thr)/S:.2f}  checks passed"
          f" (exp {len(exp_Xs)}, tanh {len(tanh_Xs)})")

# --------------------------------------------------------------------------- #
# exp truncation-tail regression (2026-07-02, found by the E1 prompt sweep).
#
# The old fixed Taylor cap N = floor(x)+60 gave exp_bounds a PRECISION-
# INDEPENDENT relative-width floor (~6.3e-33 at |x|=11.56, identical at
# P=128/192/256) — sound, but it silently dominated widths at P >= ~128 and
# would have confounded the P=384 GPT-2 run. After the precision-aware fix
# the enclosure must (a) still contain mpmath truth and (b) have relative
# width within a few ulps: rel_width < 2**-(P-16).
# --------------------------------------------------------------------------- #
print("\n--- exp truncation-tail regression ---")
for P in (128, 192, 256, 384):
    exact.set_precision(P)
    E._GELU_C = None
    S = 1 << P
    for xr in (-25.0, -11.56, -4.9, 4.9, 11.56, 25.0):
        X = int(xr * S)
        lo_i, hi_i = E.exp_bounds_safe(X)
        lo, hi = Fraction(lo_i, S), Fraction(hi_i, S)
        tlo, thi = _enclose_exp(P, X)
        mid = (tlo + thi) / 2
        encl = lo <= mid <= hi
        # tightness: width must be a few ABSOLUTE ulps plus a few relative
        # ulps of the value (for tiny exp(-|x|) the 1-ulp reciprocal rounding
        # is unavoidable and correct; the old P-independent TAIL floor was
        # orders of magnitude above this)
        wid_ulps = hi_i - lo_i
        tight = wid_ulps <= 4 + (lo_i >> (P - 16))
        good = encl and tight
        ok &= good
        if not good:
            print(f"  P={P} exp({xr:+.2f}): width {wid_ulps} ulps "
                  f"encloses={encl} FAIL")
    print(f"  P={P:3d}: tail-floor checks passed")

# --------------------------------------------------------------------------- #
# pi / sqrt(2/pi) precision-parametric regression (2026-07-02, found by the
# aborted first GPT-2 P=384 run). The old 61-digit pi string floored the
# sqrt(2/pi) enclosure at ~4e-61 independent of P. The Machin-bound pi must
# (a) enclose mpmath pi, (b) have width <= 2**-(2P+16), and the GELU constant
# and a mid-range gelu evaluation must be ulp-clean at every tested P.
# --------------------------------------------------------------------------- #
print("\n--- pi enclosure regression ---")
for P in (96, 192, 384):
    exact.set_precision(P)
    E._GELU_C = None
    S = 1 << P
    plo, phi = E.pi_bounds()
    if _HAVE_MPMATH:
        mpmath.mp.prec = 2 * P + 120
        ptrue = Fraction(mpmath.nstr(+mpmath.pi, 2 * P // 3 + 60))
        good = plo <= ptrue <= phi
        ok &= good
        if not good:
            print(f"  P={P}: pi enclosure FAILS to contain mpmath pi")
    good = (phi - plo) <= Fraction(1, 1 << (2 * P + 16))
    ok &= good
    if not good:
        print(f"  P={P}: pi width {float(phi-plo):.2e} too wide")
    c = E.gelu_const_ival()
    good = (c.hi_i - c.lo_i) <= 4          # sqrt(2/pi) within 4 ulps
    ok &= good
    if not good:
        print(f"  P={P}: gelu_const width {c.hi_i - c.lo_i} ulps FAIL")
    g = E.gelu_new_ival(Ival.point(Fraction(3, 2)))   # x = 1.5, tanh' active
    good = (g.hi_i - g.lo_i) <= 1 << 14    # a few-thousand ulps, NOT 2**~184
    ok &= good
    print(f"  P={P:3d}: pi width {float(phi-plo):.2e}, gelu_const "
          f"{c.hi_i - c.lo_i} ulps, gelu(1.5) width {g.hi_i - g.lo_i} ulps"
          f"{' OK' if good else ' FAIL'}")

exact.set_precision(96)
E._GELU_C = None

print("ALL OK" if ok else "FAILURES PRESENT")
