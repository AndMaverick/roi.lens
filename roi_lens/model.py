"""The decision model: uncertain inputs, formulas for monthly benefits and costs, and the timing and risk structure around them."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .distributions import Distribution
from .expr import Expr

EVENT_KINDS = ("delay", "benefit_factor", "extra_cost", "cancel")


@dataclass(frozen=True)
class Event:
    """A discrete risk that happens with some probability.

    ``delay``: go-live slips by ``amount`` months (rounded up). ``benefit_factor``: benefits are multiplied by ``amount`` (0.7 means 30
    percent less). ``extra_cost``: a one-off cost of ``amount`` in ``month``. ``cancel``: the programme is shut down after ``month``;
    spend already made stays spent and nothing further is earned or paid."""

    name: str
    probability: float
    kind: str
    amount: Distribution | None = None
    month: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"event {self.name!r}: kind must be one of {', '.join(EVENT_KINDS)}")
        if not 0 <= self.probability <= 1:
            raise ValueError(f"event {self.name!r}: probability must be between 0 and 1")
        if self.kind != "cancel" and self.amount is None:
            raise ValueError(f"event {self.name!r}: {self.kind} needs an amount")
        if self.month is not None and self.month < 1:
            raise ValueError(f"event {self.name!r}: month must be at least 1")


@dataclass(frozen=True)
class Model:
    name: str
    inputs: dict[str, Distribution]
    benefits: dict[str, Expr]
    run_costs: dict[str, Expr] = field(default_factory=dict)
    one_time_costs: dict[str, Expr] = field(default_factory=dict)
    horizon_months: int = 36
    build_months: int = 0
    ramp_months: int = 0
    ramp: str = "linear"
    benefit_decay_per_year: float = 0.0
    discount_rate: float = 0.10
    events: tuple[Event, ...] = ()
    correlations: tuple[tuple[str, str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.horizon_months < 1:
            raise ValueError("horizon_months must be at least 1")
        if self.build_months < 0 or self.ramp_months < 0:
            raise ValueError("build_months and ramp_months must not be negative")
        if self.build_months >= self.horizon_months:
            raise ValueError("build_months must be shorter than the horizon, or there is no time to earn anything")
        if self.ramp not in ("linear", "smooth"):
            raise ValueError("ramp must be linear or smooth")
        if not 0 <= self.benefit_decay_per_year < 1:
            raise ValueError("benefit_decay_per_year must be in [0, 1)")
        if self.discount_rate <= -1:
            raise ValueError("discount_rate must be above -100 percent")
        if not self.benefits:
            raise ValueError("a model needs at least one benefit")
        names = set(self.inputs)
        for group, items in (("benefit", self.benefits), ("run cost", self.run_costs), ("one-time cost", self.one_time_costs)):
            for label, expr in items.items():
                missing = sorted(expr.names - names)
                if missing:
                    raise ValueError(f"{group} {label!r} uses unknown input {missing[0]!r}")
        for a, b, _ in self.correlations:
            if a not in names or b not in names:
                raise ValueError(f"correlation names unknown input {a if a not in names else b!r}")
        seen = [e.name for e in self.events]
        if len(seen) != len(set(seen)):
            raise ValueError("event names must be unique")
        clash = sorted(set(seen) & names)
        if clash:
            raise ValueError(f"{clash[0]!r} is used as both an input and an event")

    @property
    def input_names(self) -> list[str]:
        """Sorted, so results never depend on the order keys were written in a file."""
        return sorted(self.inputs)

    @property
    def monthly_discount(self) -> float:
        return float((1 + self.discount_rate) ** (1 / 12) - 1)


def ramp_factor(k: int, ramp_months: int, kind: str) -> float:
    """Share of full benefit in live month ``k`` (1 is the first live month)."""
    if k <= 0:
        return 0.0
    if ramp_months <= 0:
        return 1.0
    x = min(1.0, k / ramp_months)
    return x if kind == "linear" else x * x * (3 - 2 * x)


def irr(cashflows: list[float]) -> float:
    """Annualised internal rate of return of monthly flows (month 0 first), or nan when there is no sign change. Found by bisection on
    the monthly rate; with several sign changes there can be more than one root and this returns the one the bracket converges to."""
    if not (any(c < 0 for c in cashflows) and any(c > 0 for c in cashflows)):
        return math.nan

    def npv(rate: float) -> float:
        return sum(c / (1 + rate) ** t for t, c in enumerate(cashflows))

    lo, hi = -0.99, 10.0
    flo, fhi = npv(lo), npv(hi)
    if flo * fhi > 0:
        return math.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = npv(mid)
        if fm == 0 or (hi - lo) < 1e-14:
            break
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    monthly = (lo + hi) / 2
    return (1 + monthly) ** 12 - 1
