"""Hoeffding population lower-bound reporter (certinf/bound_report.py) and its
independent cross-check (tools/bound_crosscheck.py) — Task 2.5.
"""
import math
import os
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction

import pytest

from certinf.bound_report import hoeffding_lower_bound

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BOUND_REPORT_MODULE = "certinf.bound_report"
_CROSSCHECK = os.path.join(_REPO, "tools", "bound_crosscheck.py")


# --------------------------------------------------------------------------- #
# reporter maths
# --------------------------------------------------------------------------- #
def test_headline_case_reproduces_claim_freeze_0_9570():
    """n=1000, k=1000, delta=1/40 must reproduce docs/claim-freeze.md's
    RESULTS figure (0.9570) exactly."""
    report = hoeffding_lower_bound(1000, 1000, "1/40")
    assert report.lower_bound_display == "0.9570"


def test_fallback_n_case_reproduces_claim_freeze_0_9504():
    """n=750, k=750, delta=1/40 must reproduce the fallback headline
    (docs/claim-freeze.md: "at least 95.04%")."""
    report = hoeffding_lower_bound(750, 750, "1/40")
    assert report.lower_bound_display == "0.9504"


def test_lower_bound_is_a_true_lower_bound_vs_high_precision_float():
    """The displayed bound never overstates a high-precision independent
    float computation of the same formula."""
    n, k, delta = 1000, 950, Fraction(1, 40)
    report = hoeffding_lower_bound(n, k, delta)
    eps_float = math.sqrt(math.log(1.0 / float(delta)) / (2.0 * n))
    true_value = (k / n) - eps_float
    assert Decimal(report.lower_bound_display) <= Decimal(repr(true_value))


def test_delta_accepts_fraction_int_string_and_pair():
    base = hoeffding_lower_bound(100, 90, Fraction(1, 20))
    assert hoeffding_lower_bound(100, 90, "1/20").lower_bound_display == \
        base.lower_bound_display
    assert hoeffding_lower_bound(100, 90, "0.05").lower_bound_display == \
        base.lower_bound_display
    assert hoeffding_lower_bound(100, 90, [1, 20]).lower_bound_display == \
        base.lower_bound_display


def test_delta_rejects_bare_float():
    with pytest.raises(ValueError):
        hoeffding_lower_bound(100, 90, 0.05)


# --------------------------------------------------------------------------- #
# clipping
# --------------------------------------------------------------------------- #
def test_clips_to_zero_when_epsilon_exceeds_k_over_n():
    """A tiny k with a wide-epsilon delta must clip to 0.0000, never go
    negative."""
    report = hoeffding_lower_bound(10, 0, "0.5")
    assert report.lower_bound_display == "0.0000"
    assert Decimal(report.lower_bound_display) >= 0


def test_never_exceeds_one():
    """k == n still leaves epsilon > 0, so the raw value is always < 1, but
    the clip is exercised directly here as a boundary guarantee."""
    report = hoeffding_lower_bound(5, 5, "0.5")
    assert Decimal(report.lower_bound_display) <= 1


# --------------------------------------------------------------------------- #
# tiny / huge n, delta edges
# --------------------------------------------------------------------------- #
def test_tiny_n_one():
    report = hoeffding_lower_bound(1, 1, "1/40")
    assert 0 <= Decimal(report.lower_bound_display) <= 1


def test_huge_n_approaches_k_over_n():
    n = 10_000_000
    report = hoeffding_lower_bound(n, n, "1/40")
    # epsilon shrinks as n grows, so the bound should be extremely close to 1
    assert Decimal(report.lower_bound_display) > Decimal("0.999")


def test_delta_close_to_one_gives_a_looser_higher_bound():
    """A larger delta (weaker confidence requirement, e.g. only 1% confidence
    at delta=0.99) admits a smaller epsilon and hence a HIGHER (less
    conservative) displayed lower bound than a small, strict delta."""
    report_strict = hoeffding_lower_bound(1000, 1000, "1/40")   # 97.5% confidence
    report_loose = hoeffding_lower_bound(1000, 1000, "0.99")    # 1% confidence
    assert Decimal(report_loose.lower_bound_display) > Decimal(report_strict.lower_bound_display)


@pytest.mark.parametrize("bad_delta", ["0", "1", "-0.1", "1.5"])
def test_delta_out_of_range_raises(bad_delta):
    with pytest.raises(ValueError):
        hoeffding_lower_bound(1000, 1000, bad_delta)


def test_k_greater_than_n_raises():
    with pytest.raises(ValueError):
        hoeffding_lower_bound(10, 11, "1/40")


def test_k_negative_raises():
    with pytest.raises(ValueError):
        hoeffding_lower_bound(10, -1, "1/40")


def test_n_nonpositive_raises():
    with pytest.raises(ValueError):
        hoeffding_lower_bound(0, 0, "1/40")


# --------------------------------------------------------------------------- #
# line() / property labelling
# --------------------------------------------------------------------------- #
def test_line_contains_population_claim_prefix_and_property_label():
    report = hoeffding_lower_bound(1000, 1000, "1/40")
    line = report.line(property_name="phi1")
    assert line.startswith("population claim: p >= 0.9570")
    assert "(phi1)" in line
    assert "1-1/40" in line


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_reproduces_headline_bound():
    p = subprocess.run(
        [sys.executable, "-m", _BOUND_REPORT_MODULE,
         "--n", "1000", "--k", "1000", "--delta", "1/40"],
        capture_output=True, text=True, cwd=_REPO,
    )
    assert p.returncode == 0, p.stderr
    assert "population claim: p >= 0.9570" in p.stdout


# --------------------------------------------------------------------------- #
# independent cross-check (tools/bound_crosscheck.py) runs in the suite
# --------------------------------------------------------------------------- #
def test_crosscheck_script_agrees_with_reporter():
    p = subprocess.run([sys.executable, _CROSSCHECK], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "OK: 4/4 cases agree" in p.stdout
