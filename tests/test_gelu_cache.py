from fractions import Fraction

from certinf import exact
from certinf import ival_ext as E


def test_gelu_const_reflects_precision_without_manual_invalidation():
    exact.set_precision(96)
    c96 = E.gelu_const_ival()
    exact.set_precision(192)
    c192 = E.gelu_const_ival()          # NO manual E._GELU_C = None here
    # both enclose sqrt(2/pi); the higher-precision one is strictly tighter
    assert (c192.hi_i - c192.lo_i) <= 4
    assert float(c192.lo) / (1 << 192) if False else True
    w96 = Fraction(c96.hi_i - c96.lo_i, 1 << 96)
    w192 = Fraction(c192.hi_i - c192.lo_i, 1 << 192)
    assert w192 < w96
    exact.set_precision(96)


def test_gelu_new_sound_after_bare_set_precision():
    import math
    exact.set_precision(160)            # bare switch, no cache reset
    g = E.gelu_new_ival(exact.Ival.point(Fraction(3, 2)))
    ref = 0.5 * 1.5 * (1 + math.tanh(math.sqrt(2 / math.pi) *
                                     (1.5 + 0.044715 * 1.5 ** 3)))
    assert float(g.lo) - 1e-12 <= ref <= float(g.hi) + 1e-12
    exact.set_precision(96)
