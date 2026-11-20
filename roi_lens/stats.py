"""Statistics with honest uncertainty: every estimate that matters comes with an interval you can check."""
from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Quantile of an already-sorted sample by linear interpolation between order statistics (the usual 'type 7' definition)."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("no values")
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, n - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def quantile_nearest(sorted_values: Sequence[float], q: float) -> float:
    """The ceil(q n)-th order statistic. Unlike interpolation it stays defined when some values are infinite (a payback that never comes)."""
    n = len(sorted_values)
    return sorted_values[min(n - 1, max(0, math.ceil(q * n) - 1))]


def mean_se(values: Sequence[float]) -> tuple[float, float]:
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var / n)


def wilson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion. Stays inside [0, 1] and behaves at 0 and n, where the textbook formula breaks."""
    if n <= 0:
        raise ValueError("n must be positive")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return lo, hi


def _binom_cdf_table(n: int, p: float) -> list[float]:
    """P(X <= k) for k = 0..n, X ~ Binomial(n, p), accumulated from log probabilities so large n does not underflow."""
    if p <= 0:
        return [1.0] * (n + 1)
    if p >= 1:
        return [0.0] * n + [1.0]
    lp, lq = math.log(p), math.log1p(-p)
    out, total = [], 0.0
    for k in range(n + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * lp + (n - k) * lq)
        out.append(min(1.0, total))
    return out


def quantile_ci(sorted_values: Sequence[float], q: float, confidence: float = 0.95) -> tuple[float, float]:
    """Distribution-free confidence interval for the q-quantile from order statistics. It assumes nothing about the shape of the data
    (only independence), and its coverage is at least ``confidence`` by construction."""
    n = len(sorted_values)
    alpha = 1 - confidence
    cdf = _binom_cdf_table(n, q)
    # X(j) <= true quantile iff at least j sample points lie at or below it; choose the widest safe pair of ranks
    lo_rank = 1
    for j in range(1, n + 1):
        if cdf[j - 1] <= alpha / 2:
            lo_rank = j
        else:
            break
    hi_rank = n
    for k in range(1, n + 1):
        if cdf[k - 1] >= 1 - alpha / 2:
            hi_rank = k
            break
    return sorted_values[lo_rank - 1], sorted_values[hi_rank - 1]


def cvar(values: Sequence[float], alpha: float = 0.05) -> float:
    """Expected shortfall: the mean of the worst ``alpha`` share of outcomes. It answers 'when it goes badly, how badly?'"""
    ordered = sorted(values)
    k = max(1, math.ceil(alpha * len(ordered)))
    return sum(ordered[:k]) / k


def summarize(values: Sequence[float], confidence: float = 0.95) -> dict[str, float]:
    ordered = sorted(values)
    n = len(ordered)
    m, se = mean_se(ordered)
    sd = se * math.sqrt(n) if n > 1 else 0.0
    positive = sum(1 for v in ordered if v > 0)
    plo, phi = wilson(positive, n, confidence)
    p50lo, p50hi = quantile_ci(ordered, 0.5, confidence)
    return {
        "n": float(n), "mean": m, "mean_se": se, "sd": sd, "min": ordered[0], "max": ordered[-1],
        "p05": quantile(ordered, 0.05), "p10": quantile(ordered, 0.10), "p25": quantile(ordered, 0.25), "p50": quantile(ordered, 0.5),
        "p75": quantile(ordered, 0.75), "p90": quantile(ordered, 0.90), "p95": quantile(ordered, 0.95),
        "p50_ci_low": p50lo, "p50_ci_high": p50hi,
        "prob_positive": positive / n, "prob_positive_ci_low": plo, "prob_positive_ci_high": phi,
        "cvar05": cvar(ordered, 0.05),
    }
