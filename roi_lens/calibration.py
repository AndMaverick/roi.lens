"""Calibration: are your 90 percent intervals actually right 90 percent of the time?

Estimators are routinely overconfident: intervals meant to catch the truth nine times in ten catch it five or six. Feed this module the
intervals you gave on past questions, with the answers that turned out true, and it reports your hit rate with an interval around it, and
the factor that would have made your intervals right."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .stats import quantile_nearest, wilson


@dataclass(frozen=True)
class Answer:
    low: float
    high: float
    truth: float

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"an interval needs low < high, got {self.low} and {self.high}")

    @property
    def hit(self) -> bool:
        return self.low <= self.truth <= self.high

    @property
    def needed_scale(self) -> float:
        """How many times wider (about the centre) the interval had to be to contain the truth."""
        centre, half = (self.low + self.high) / 2, (self.high - self.low) / 2
        return abs(self.truth - centre) / half


@dataclass(frozen=True)
class Calibration:
    n: int
    hits: int
    hit_rate: float
    ci_low: float
    ci_high: float
    verdict: str
    scale_factor: float

    def widen(self, low: float, high: float) -> tuple[float, float]:
        """Apply the correction to a new 90 percent interval."""
        return scale_interval(low, high, self.scale_factor)


def scale_interval(low: float, high: float, factor: float) -> tuple[float, float]:
    if factor <= 0:
        raise ValueError("factor must be positive")
    centre, half = (low + high) / 2, (high - low) / 2
    return centre - half * factor, centre + half * factor


def calibrate(answers: Sequence[Answer], target: float = 0.90, confidence: float = 0.95) -> Calibration:
    """Hit rate against ``target``. The verdict is ``overconfident`` or ``underconfident`` only when the Wilson interval around the hit
    rate excludes the target; otherwise ``consistent with target`` (which with few answers is a weak statement, so read the interval)."""
    if not answers:
        raise ValueError("need at least one answer")
    n = len(answers)
    hits = sum(a.hit for a in answers)
    lo, hi = wilson(hits, n, confidence)
    verdict = "overconfident" if hi < target else "underconfident" if lo > target else "consistent with target"
    # the ceil(target * n)-th smallest needed scale, so that scaling by it contains at least ``target`` of the answers (a tiny margin
    # keeps the boundary answer inside despite floating-point rounding)
    scale = quantile_nearest(sorted(a.needed_scale for a in answers), target) * (1 + 1e-12)
    return Calibration(n, hits, hits / n, lo, hi, verdict, scale)


def required_answers(target: float = 0.90, margin: float = 0.10, confidence: float = 0.95) -> int:
    """Smallest number of answered questions for which a hit rate sitting exactly at ``target`` has a Wilson interval narrower than
    +/- ``margin``. It tells you how many past estimates you need before the verdict means much."""
    n = 1
    while True:
        lo, hi = wilson(round(target * n), n, confidence)
        if (hi - lo) / 2 <= margin and not math.isnan(lo):
            return n
        n += 1
