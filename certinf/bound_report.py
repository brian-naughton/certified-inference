"""Hoeffding population lower-bound reporter (Task 2.5; Codex SHOULD #5,
the pre-freeze external evaluation (internal)).

Given exact integers ``n``, ``k`` and an exact failure probability ``delta``,
computes and renders the one-sided Hoeffding lower bound

    L = k/n - sqrt(ln(1/delta) / (2n))

clipped to ``[0, 1]``. This is the number a reader quotes, so it must never
*overstate* the certified population rate:

  - ``epsilon = sqrt(ln(1/delta) / (2n))`` is computed at high internal
    precision (stdlib :mod:`decimal`, ln + sqrt) and rounded UP,
  - the resulting ``L = k/n - epsilon`` is rounded DOWN at the display
    precision,

so any displayed lower bound understates rather than overstates the true
value — the same safe-direction convention as
``certinf.prereg._hoeffding_epsilon_display`` (reused here directly rather
than re-derived, so the two conventions cannot silently drift apart).

This module's own arithmetic is the ``decimal`` route. `tools/bound_crosscheck.py`
re-derives the same bound via a genuinely different arithmetic route (stdlib
`math`, IEEE-754 double precision) and asserts agreement — see that module's
docstring for why the two are independent and for the one documented,
deliberate discrepancy (the n=500 reference figure).

CLI:

    python3.11 -m certinf.bound_report --n 1000 --k 1000 --delta 1/40
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal, localcontext
from fractions import Fraction

from certinf.prereg import _hoeffding_epsilon_display, _to_fraction

# Internal precision for epsilon before the final display rounding. Far above
# any plausible display_dp, so the display rounding is the only place a
# value is thrown away.
_INTERNAL_DP = 30

# Default display precision for the LOWER BOUND itself (4 dp = 2dp-percent,
# matching docs/claim-freeze.md's "95.70%" / "95.04%" convention).
_DEFAULT_DISPLAY_DP = 4


@dataclass(frozen=True)
class BoundReport:
    """A Hoeffding population lower-bound report.

    Attributes:
        n: Sample size (exact).
        k: Successes out of ``n`` (exact).
        delta: Failure probability (exact ``Fraction``).
        epsilon_display: ``epsilon(n, delta)``, rounded UP, at
            ``_INTERNAL_DP`` places (advisory; the authoritative record is
            ``(n, k, delta)``).
        lower_bound_raw: The unrounded (but still epsilon-rounded-up, hence
            still a valid lower bound) ``k/n - epsilon``, clipped to
            ``[0, 1]``, at ``_INTERNAL_DP`` places of precision — provided
            for independent cross-checking, never for display.
        lower_bound_display: ``lower_bound_raw`` rounded DOWN at
            ``display_dp`` places — the number to quote.
        display_dp: Decimal places in ``lower_bound_display``.
    """

    n: int
    k: int
    delta: Fraction
    epsilon_display: str
    lower_bound_raw: Decimal
    lower_bound_display: str
    display_dp: int

    def line(self, *, property_name: str | None = None) -> str:
        """The one-line population-claim statement quoted downstream (e.g. by
        ``certificates/check.py``'s ``--prereg`` output)."""
        conf = f"1-{self.delta.numerator}/{self.delta.denominator}"
        suffix = f" ({property_name})" if property_name else ""
        return (f"population claim: p >= {self.lower_bound_display} "
                f"at confidence >= {conf}{suffix}")


def _validate_inputs(n: int, k: int, delta: Fraction) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError(f"n must be a positive int: {n!r}")
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError(f"k must be a non-negative int: {k!r}")
    if k > n:
        raise ValueError(f"k must not exceed n: k={k!r}, n={n!r}")
    if not (0 < delta < 1):
        raise ValueError(f"delta must lie in the open interval (0, 1): {delta!r}")


def hoeffding_lower_bound(n: int, k: int, delta, *,
                          display_dp: int = _DEFAULT_DISPLAY_DP) -> BoundReport:
    """Compute the one-sided Hoeffding population lower bound.

    Args:
        n: Sample size (exact positive int).
        k: Successes out of ``n`` (exact int, ``0 <= k <= n``).
        delta: Failure probability — an exact ``Fraction``, an int, a
            decimal/ratio string (``"0.05"``, ``"1/40"``), or a
            ``[num, den]`` pair. Never a bare ``float`` (see
            ``certinf.prereg._to_fraction``).
        display_dp: Decimal places in the returned ``lower_bound_display``.

    Returns:
        The :class:`BoundReport`.

    Raises:
        ValueError: If ``n``, ``k``, or ``delta`` is out of range, or
            ``delta`` cannot be parsed as an exact rational.
    """
    delta_frac = delta if isinstance(delta, Fraction) else _to_fraction(delta)
    _validate_inputs(n, k, delta_frac)

    eps_display = _hoeffding_epsilon_display(n, delta_frac, dp=_INTERNAL_DP)
    with localcontext() as ctx:
        ctx.prec = _INTERNAL_DP + 40
        k_over_n = Decimal(k) / Decimal(n)
        eps_dec = Decimal(eps_display)
        raw = k_over_n - eps_dec
        clipped = max(Decimal(0), min(Decimal(1), raw))
        quantum = Decimal(1).scaleb(-display_dp)
        display = clipped.quantize(quantum, rounding=ROUND_FLOOR)

    return BoundReport(
        n=n, k=k, delta=delta_frac,
        epsilon_display=eps_display,
        lower_bound_raw=clipped,
        lower_bound_display=str(display),
        display_dp=display_dp,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Hoeffding population lower-bound reporter: "
                    "L = k/n - sqrt(ln(1/delta)/(2n)), clipped to [0,1].")
    ap.add_argument("--n", type=int, required=True, help="sample size")
    ap.add_argument("--k", type=int, required=True, help="successes out of n")
    ap.add_argument("--delta", type=str, required=True,
                    help="failure probability: exact decimal or ratio string, "
                         "e.g. '0.05' or '1/40' (never a bare float)")
    ap.add_argument("--dp", type=int, default=_DEFAULT_DISPLAY_DP,
                    help=f"display decimal places (default {_DEFAULT_DISPLAY_DP})")
    ap.add_argument("--property", type=str, default=None,
                    help="optional property label for the printed line "
                         "(e.g. phi1, phi2_joint)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    report = hoeffding_lower_bound(args.n, args.k, args.delta, display_dp=args.dp)
    print(f"n = {report.n}")
    print(f"k = {report.k}")
    print(f"delta = {report.delta.numerator}/{report.delta.denominator} (exact)")
    print(f"epsilon(n, delta) = {report.epsilon_display} "
          f"(rounded UP, {_INTERNAL_DP} dp internal precision)")
    print(report.line(property_name=args.property))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
