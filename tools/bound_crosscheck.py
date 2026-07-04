#!/usr/bin/env python3
"""Independent cross-check for `certinf/bound_report.py` (Task 2.5; Codex
SHOULD #5, `docs/reviews/2026-07-03-codex-whole-corpus-evaluation.md`: "Cross-
check against a small independent script").

`certinf.bound_report` computes the Hoeffding lower bound

    L = k/n - sqrt(ln(1/delta) / (2n)),  clipped to [0, 1]

via the stdlib `decimal` module (60-significant-digit `ln`/`sqrt`, directed
rounding). This script re-derives the SAME bound via a DIFFERENT arithmetic
route — stdlib `math` (IEEE-754 double precision `math.log` / `math.sqrt`) —
and asserts the two agree, on a table that includes the frozen headline case
and the `docs/claim-freeze.md` n=750/500 reference cases.

It deliberately does not import `certinf.prereg` or reuse any of
`bound_report`'s helpers for the arithmetic itself; the only thing shared
with `certinf.bound_report` is the value under test (imported explicitly, and
only to obtain the value being cross-checked, never its internals).

One documented discrepancy: `docs/claim-freeze.md`'s "n decision rule" table
lists the n=500 case as "reference only... approx 0.9393". That figure uses
ordinary round-to-nearest. `bound_report.hoeffding_lower_bound` deliberately
rounds the final lower bound DOWN (never overstate a published rate), and
0.93926... sits just below the 0.9393 rounding boundary, so its safe-direction
display is 0.9392, one part in the last displayed digit below the freeze
doc's pre-run approximation. That is expected and is not an arithmetic
disagreement — this script checks numeric agreement on the raw (unrounded)
value, and only checks the *displayed* string where both rounding
conventions agree (n=1000, n=750).

Run standalone: `python3.11 tools/bound_crosscheck.py`
"""
from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from certinf.bound_report import hoeffding_lower_bound  # noqa: E402  (value under test only)

# Numeric agreement tolerance between the two arithmetic routes' RAW (unrounded)
# lower-bound value. IEEE-754 doubles carry ~15-17 significant decimal digits,
# so 1e-9 is generous headroom while still catching a wrong formula, a sign
# error, a missing factor of 2, or a swapped k/n.
_RAW_TOLERANCE = 1e-9

# (n, k, delta, expected DISPLAY string -- None where the doc's pre-run
# reference figure uses a different (round-to-nearest) rounding convention
# than bound_report's deliberate round-down; see module docstring).
CASES = [
    (1000, 1000, "1/40", "0.9570"),   # headline (docs/claim-freeze.md RESULTS)
    (750, 750, "1/40", "0.9504"),     # fallback n (docs/claim-freeze.md)
    (500, 500, "1/40", None),         # reference only; doc's approx is 0.9393
                                       # (round-to-nearest); round-down is 0.9392
    (20, 20, "1/20", "0.7263"),       # small-n sanity case (Phase 1 pilot scale)
]


def independent_lower_bound(n: int, k: int, delta) -> float:
    """Hoeffding lower bound via stdlib `math` double precision -- a
    different arithmetic route than `bound_report`'s `decimal` computation.
    Clipped to [0, 1]. This is a cross-check value only, never a published
    claim (no directed rounding, no safe-direction guarantee)."""
    delta_f = float(Fraction(delta)) if not isinstance(delta, float) else delta
    if not (0.0 < delta_f < 1.0):
        raise ValueError(f"delta must lie in (0, 1): {delta_f!r}")
    eps = math.sqrt(math.log(1.0 / delta_f) / (2.0 * n))
    return max(0.0, min(1.0, (k / n) - eps))


def check_case(n: int, k: int, delta: str, expected_display: str | None) -> str:
    """Run one (n, k, delta) case; return a one-line PASS summary or raise
    AssertionError describing the disagreement."""
    report = hoeffding_lower_bound(n, k, delta)
    independent = independent_lower_bound(n, k, delta)
    raw = float(report.lower_bound_raw)

    diff = abs(independent - raw)
    assert diff < _RAW_TOLERANCE, (
        f"n={n} k={k} delta={delta}: bound_report raw={raw!r} vs "
        f"independent (math, double precision)={independent!r}, "
        f"diff={diff!r} >= tolerance {_RAW_TOLERANCE!r}")

    if expected_display is not None:
        assert report.lower_bound_display == expected_display, (
            f"n={n} k={k} delta={delta}: bound_report display "
            f"{report.lower_bound_display!r} != expected {expected_display!r}")

    return (f"PASS  n={n:>5} k={k:>5} delta={delta:>6}  "
            f"bound_report={report.lower_bound_display}  "
            f"independent(math)={independent:.6f}  diff={diff:.2e}")


def main() -> int:
    lines = [check_case(n, k, delta, expected) for n, k, delta, expected in CASES]
    for line in lines:
        print(line)
    print(f"OK: {len(lines)}/{len(lines)} cases agree "
          f"(raw tolerance {_RAW_TOLERANCE:.0e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
