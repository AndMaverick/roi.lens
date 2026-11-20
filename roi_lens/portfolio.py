"""Choosing among several candidate projects under one budget."""
from __future__ import annotations

from array import array
from collections.abc import Sequence
from dataclasses import dataclass

from .engine import simulate
from .model import Model
from .stats import quantile, wilson

MAX_CASES = 20


@dataclass(frozen=True)
class Case:
    model: Model
    commit: float

    @property
    def name(self) -> str:
        return self.model.name


@dataclass(frozen=True)
class Option:
    names: tuple[str, ...]
    commit: float
    mean_npv: float
    p10: float
    p50: float
    p90: float
    prob_positive: float
    prob_positive_ci: tuple[float, float]


@dataclass(frozen=True)
class Plan:
    budget: float
    objective: str
    best: Option
    alternatives: tuple[Option, ...]
    case_means: dict[str, float]
    feasible_subsets: int


def derive_seed(seed: int, index: int) -> int:
    return (seed * 1_000_003 + index * 7_919) % (2 ** 31 - 1)


def _option(names: tuple[str, ...], commit: float, sums: Sequence[float]) -> Option:
    ordered = sorted(sums)
    pos = sum(1 for x in ordered if x > 0)
    return Option(names, commit, sum(ordered) / len(ordered), quantile(ordered, 0.1), quantile(ordered, 0.5), quantile(ordered, 0.9),
                  pos / len(ordered), wilson(pos, len(ordered)))


def plan(cases: Sequence[Case], budget: float, *, trials: int = 3000, seed: int = 7, method: str = "lhs", objective: str = "mean",
         top: int = 5, must_include: Sequence[str] = ()) -> Plan:
    """Pick the set of projects that fits ``budget`` and does best on ``objective``.

    Every feasible subset is scored by its summed mean NPV exactly (this is exhaustive, so the best subset on ``mean`` is the true best,
    not a heuristic). For ``p10`` (a risk-averse objective) and ``prob_positive``, the 64 best subsets by mean are re-ranked on the
    simulated portfolio, which is a heuristic that can in principle miss a subset with a worse mean but a much safer spread. Cases are
    simulated independently of each other; correlation between projects is not modelled."""
    if objective not in ("mean", "p10", "prob_positive"):
        raise ValueError("objective must be mean, p10 or prob_positive")
    if not cases:
        raise ValueError("need at least one case")
    if len(cases) > MAX_CASES:
        raise ValueError(f"at most {MAX_CASES} cases (every subset is scored)")
    ordered_cases = sorted(cases, key=lambda c: c.name)
    names = [c.name for c in ordered_cases]
    if len(set(names)) != len(names):
        raise ValueError("case names must be unique")
    unknown = sorted(set(must_include) - set(names))
    if unknown:
        raise ValueError(f"must_include names unknown case {unknown[0]!r}")
    if any(c.commit < 0 for c in ordered_cases):
        raise ValueError("commit must not be negative")
    runs = [simulate(c.model, trials, derive_seed(seed, i), method).metric("npv") for i, c in enumerate(ordered_cases)]
    means = [sum(r) / len(r) for r in runs]
    n = len(ordered_cases)
    must = 0
    for i, nm in enumerate(names):
        if nm in must_include:
            must |= 1 << i
    sums_mean = array("d", [0.0]) * (1 << n)
    sums_commit = array("d", [0.0]) * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & -mask
        i = low.bit_length() - 1
        rest = mask ^ low
        sums_mean[mask] = sums_mean[rest] + means[i]
        sums_commit[mask] = sums_commit[rest] + ordered_cases[i].commit
    feasible = [m for m in range(1 << n) if sums_commit[m] <= budget + 1e-9 and (m & must) == must]
    if not feasible:
        raise ValueError("no combination fits the budget with the required cases")
    feasible.sort(key=lambda m: (-sums_mean[m], sums_commit[m], m))

    def sims(mask: int) -> list[float]:
        idx = [i for i in range(n) if mask >> i & 1]
        return [sum(runs[i][t] for i in idx) for t in range(trials)]

    def option(mask: int) -> Option:
        return _option(tuple(nm for i, nm in enumerate(names) if mask >> i & 1), sums_commit[mask], sims(mask) if mask else [0.0] * trials)

    pool = feasible[: max(top, 64)] if objective != "mean" else feasible[:top]
    options = [option(m) for m in pool]
    if objective == "p10":
        options.sort(key=lambda o: (-o.p10, o.commit, o.names))
    elif objective == "prob_positive":
        options.sort(key=lambda o: (-o.prob_positive, -o.mean_npv, o.names))
    return Plan(budget, objective, options[0], tuple(options[:top]), dict(zip(names, means, strict=True)), len(feasible))
