"""What drives the answer, and what is it worth to find out more."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .engine import Run, evaluate
from .model import Model
from .sampling import spearman


@dataclass(frozen=True)
class Driver:
    name: str
    rank_correlation: float


def rank_drivers(run: Run, metric: str = "npv") -> list[Driver]:
    """Spearman rank correlation of each input (and each risk event, as a 0/1 flag) with the outcome, strongest first. It measures which
    uncertainties move the result over the whole simulated range, whatever the shape of the relationship, as long as it is monotone."""
    y = run.metric(metric)
    keep = [i for i, v in enumerate(y) if not math.isnan(v) and not math.isinf(v)]
    yy = [y[i] for i in keep]
    out = [Driver(n, spearman([xs[i] for i in keep], yy)) for n, xs in run.values.items()]
    out += [Driver(n, spearman([1.0 if fs[i] else 0.0 for i in keep], yy)) for n, fs in run.fired.items()]
    return sorted(out, key=lambda d: (-abs(d.rank_correlation), d.name))


def _medians(model: Model) -> dict[str, float]:
    return {n: d.ppf(0.5) for n, d in model.inputs.items()}


@dataclass(frozen=True)
class Swing:
    name: str
    low_input: float
    high_input: float
    low_result: float
    high_result: float

    @property
    def swing(self) -> float:
        return abs(self.high_result - self.low_result)


def tornado(model: Model, metric: str = "npv", lo: float = 0.10, hi: float = 0.90) -> list[Swing]:
    """Move one input at a time from its ``lo`` to its ``hi`` quantile with every other input at its median, and record the result.
    Deterministic. Risk events are held off. Largest swing first."""
    if not 0 < lo < hi < 1:
        raise ValueError("need 0 < lo < hi < 1")
    base = _medians(model)
    out = []
    for name, dist in model.inputs.items():
        a, b = dist.ppf(lo), dist.ppf(hi)
        ra = getattr(evaluate(model, {**base, name: a}), metric)
        rb = getattr(evaluate(model, {**base, name: b}), metric)
        out.append(Swing(name, a, b, ra, rb))
    return sorted(out, key=lambda s: (-s.swing, s.name))


def evpi(npv: Sequence[float]) -> tuple[float, float]:
    """Expected value of perfect information for a go/no-go decision: E[max(NPV, 0)] minus max(E[NPV], 0). It is the most that any study,
    pilot or consultant could be worth, because it assumes the uncertainty is resolved completely before you decide. Returns the estimate
    and a standard error."""
    n = len(npv)
    mean = sum(npv) / n
    floor = max(mean, 0.0)
    gains = [max(x, 0.0) for x in npv]
    m = sum(gains) / n
    se = math.sqrt(sum((g - m) ** 2 for g in gains) / (n - 1) / n) if n > 1 else 0.0
    return m - floor, se


def evppi(x: Sequence[float], npv: Sequence[float], bins: int = 10) -> float:
    """Expected value of partial perfect information about one input: if you learned that input exactly before deciding, how much better is
    the decision? Estimated by cutting the trials into ``bins`` equal-count groups by that input, taking the mean NPV in each (the best
    estimate of E[NPV | input]), and valuing a go/no-go choice per group. It is biased slightly upward with few trials per bin."""
    n = len(x)
    if n != len(npv):
        raise ValueError("x and npv must be the same length")
    bins = max(1, min(bins, n))
    order = sorted(range(n), key=lambda i: (x[i], i))
    total = 0.0
    for b in range(bins):
        idx = order[b * n // bins:(b + 1) * n // bins]
        if idx:
            total += max(sum(npv[i] for i in idx) / len(idx), 0.0) * len(idx)
    return total / n - max(sum(npv) / n, 0.0)


def value_of_information(run: Run, bins: int = 10) -> list[tuple[str, float]]:
    npv = run.metric("npv")
    out = [(n, evppi(xs, npv, bins)) for n, xs in run.values.items()]
    return sorted(out, key=lambda t: (-t[1], t[0]))


@dataclass(frozen=True)
class BreakEven:
    name: str
    value: float | None
    probability_worse: float | None
    direction: str
    note: str


def breakeven(model: Model, name: str, metric: str = "npv", target: float = 0.0) -> BreakEven:
    """The value of one input at which the outcome crosses ``target`` (zero NPV by default), other inputs at their medians, and the
    probability that the input turns out on the losing side of it."""
    if name not in model.inputs:
        raise ValueError(f"unknown input {name!r}")
    dist = model.inputs[name]
    base = _medians(model)

    def f(x: float) -> float:
        return float(getattr(evaluate(model, {**base, name: x}), metric)) - target

    lo, hi = dist.ppf(0.0005), dist.ppf(0.9995)
    span = (hi - lo) or abs(hi) or 1.0
    lo, hi = lo - span, hi + span
    flo, fhi = f(lo), f(hi)
    if flo == 0 or fhi == 0:
        root = lo if flo == 0 else hi
    elif (flo > 0) == (fhi > 0):
        return BreakEven(name, None, None, "none", "the outcome does not cross the target for any value of this input")
    else:
        a, b, fa = lo, hi, flo
        for _ in range(200):
            mid = (a + b) / 2
            fm = f(mid)
            if fm == 0 or (b - a) < 1e-12 * max(1.0, abs(mid)):
                a = b = mid
                break
            if (fm > 0) == (fa > 0):
                a, fa = mid, fm
            else:
                b = mid
        root = (a + b) / 2
    rising = fhi > flo
    p_below = dist.cdf(root)
    worse = p_below if rising else 1 - p_below
    return BreakEven(name, root, worse, "higher is better" if rising else "lower is better",
                     f"{metric} reaches {target:g} at {name} = {root:.6g}")


def pilot_value(run: Run, name: str, cost: float, bins: int = 10) -> float:
    """What a pilot that would pin down one input exactly is worth after its own cost. Negative means do not bother."""
    if name not in run.values:
        raise ValueError(f"unknown input {name!r}")
    return evppi(run.values[name], run.metric("npv"), bins) - cost
