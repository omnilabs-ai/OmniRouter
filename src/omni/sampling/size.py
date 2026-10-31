"""Sample size from a confidence-interval target (ADR 0007).

The primary quality metric of a recommendation is a paired win rate, a proportion. The
worst-case (p = 0.5) normal-approximation sample size for a 95% CI of half-width ``h`` is

    n = ceil((z / (2 h))**2),  z = 1.96

so h = 0.05 needs 385 requests, h = 0.10 needs 97, h = 0.03 needs 1068. It assumes
independent requests and ignores stratification gains and the finite population, so it is a
conservative target. The result is clamped to [floor, cap] and then to what is available.
"""

from __future__ import annotations

import math

Z_95 = 1.96


def target_size(
    available: int, ci_half_width: float = 0.05, floor: int = 50, cap: int = 1000
) -> int:
    if not 0 < ci_half_width < 0.5:
        raise ValueError("ci_half_width must be in (0, 0.5)")
    if floor < 1 or cap < floor:
        raise ValueError("need 1 <= floor <= cap")
    n = math.ceil((Z_95 / (2 * ci_half_width)) ** 2)
    n = min(max(n, floor), cap)
    return min(n, max(available, 0))
