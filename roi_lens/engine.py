"""The simulation engine: draws the inputs and risk events, builds a monthly cash flow for each trial, and measures it."""
from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field

from .model import Model, irr, ramp_factor
from .sampling import correlate, correlation_matrix, uniforms


@dataclass(frozen=True)
class Flow:
    """Monthly benefits and costs, index 0 to horizon (index 0 is money spent before the first month)."""

    benefit: list[float]
    cost: list[float]

    @property
    def net(self) -> list[float]:
        return [b - c for b, c in zip(self.benefit, self.cost, strict=True)]


@dataclass(frozen=True)
class Outcome:
    npv: float
    net: float
    roi: float
    payback: float
    pv_benefit: float
    pv_cost: float
    cashflow: list[float]

    @property
    def irr(self) -> float:
        return irr(self.cashflow)


def build_flow(model: Model, v: Mapping[str, float], *, delay: int = 0, benefit_factor: float = 1.0,
               extras: tuple[tuple[int, float], ...] = (), cancel_after: int | None = None) -> Flow:
    """Monthly flow for one set of input values and one set of fired events."""
    h = model.horizon_months
    benefit = [0.0] * (h + 1)
    cost = [0.0] * (h + 1)
    monthly_benefit = sum(e(v) for e in model.benefits.values())
    monthly_run = sum(e(v) for e in model.run_costs.values())
    one_time = sum(e(v) for e in model.one_time_costs.values())
    if model.build_months == 0:
        cost[0] += one_time
    else:
        for t in range(1, model.build_months + 1):
            cost[t] += one_time / model.build_months
    live_from = model.build_months + delay + 1
    decay = 1 - model.benefit_decay_per_year
    for t in range(live_from, h + 1):
        k = t - live_from + 1
        benefit[t] = monthly_benefit * ramp_factor(k, model.ramp_months, model.ramp) * decay ** ((k - 1) / 12) * benefit_factor
        cost[t] += monthly_run
    for month, amount in extras:
        cost[min(month, h)] += amount
    if cancel_after is not None:
        for t in range(cancel_after + 1, h + 1):
            benefit[t] = 0.0
            cost[t] = 0.0
    return Flow(benefit, cost)


def measure(model: Model, flow: Flow) -> Outcome:
    m = model.monthly_discount
    disc = [(1 + m) ** -t for t in range(model.horizon_months + 1)]
    pv_b = sum(b * d for b, d in zip(flow.benefit, disc, strict=True))
    pv_c = sum(c * d for c, d in zip(flow.cost, disc, strict=True))
    net_flow = flow.net
    cum, running = [], 0.0
    for x in net_flow:
        running += x
        cum.append(running)
    trough = min(range(len(cum)), key=lambda i: (cum[i], i))
    payback = 0.0 if cum[trough] >= 0 else next((float(t) for t in range(trough, len(cum)) if cum[t] >= 0), math.inf)
    npv = pv_b - pv_c
    return Outcome(npv=npv, net=sum(net_flow), roi=(npv / pv_c) if pv_c > 0 else math.nan, payback=payback,
                   pv_benefit=pv_b, pv_cost=pv_c, cashflow=net_flow)


def evaluate(model: Model, values: Mapping[str, float], fired: Mapping[str, float] | None = None) -> Outcome:
    """Deterministic outcome for given input values. ``fired`` maps event names to the amount each takes when it happens (omit for none)."""
    missing = sorted(set(model.inputs) - set(values))
    if missing:
        raise ValueError(f"missing value for input {missing[0]!r}")
    delay, factor, extras, cancel = _apply_events(model, fired or {})
    return measure(model, build_flow(model, values, delay=delay, benefit_factor=factor, extras=extras, cancel_after=cancel))


def _apply_events(model: Model, fired: Mapping[str, float]) -> tuple[int, float, tuple[tuple[int, float], ...], int | None]:
    delay, factor, cancel = 0, 1.0, None
    extras: list[tuple[int, float]] = []
    known = {e.name for e in model.events}
    unknown = sorted(set(fired) - known)
    if unknown:
        raise ValueError(f"unknown event {unknown[0]!r}")
    for e in model.events:
        if e.name not in fired:
            continue
        amount = fired[e.name]
        if e.kind == "delay":
            delay += max(0, math.ceil(amount))
        elif e.kind == "benefit_factor":
            factor *= amount
        elif e.kind == "extra_cost":
            extras.append((e.month or 1, amount))
        else:
            when = e.month or max(1, math.ceil(model.horizon_months / 2))
            cancel = when if cancel is None else min(cancel, when)
    return delay, factor, tuple(extras), cancel


@dataclass
class Run:
    model: Model
    trials: int
    seed: int
    method: str
    values: dict[str, list[float]]
    fired: dict[str, list[bool]]
    outcomes: list[Outcome]
    _cache: dict[str, list[float]] = field(default_factory=dict, repr=False)

    def metric(self, name: str) -> list[float]:
        """Per-trial values of ``npv``, ``net``, ``roi``, ``payback``, ``irr``, ``pv_benefit`` or ``pv_cost``."""
        if name not in self._cache:
            if name not in ("npv", "net", "roi", "payback", "irr", "pv_benefit", "pv_cost"):
                raise ValueError(f"unknown metric {name!r}")
            self._cache[name] = [getattr(o, name) for o in self.outcomes]
        return self._cache[name]


def simulate(model: Model, trials: int = 5000, seed: int = 7, method: str = "lhs") -> Run:
    """Monte Carlo over the model. Same model, trials, seed and method always give the same run, whatever order the inputs were written in."""
    if trials < 2:
        raise ValueError("trials must be at least 2")
    rng = random.Random(seed)
    names = model.input_names
    events = list(model.events)
    d = len(names)
    cols = uniforms(trials, d, rng, method)
    if model.correlations and d:
        index = {n: i for i, n in enumerate(names)}
        pairs = [(index[a], index[b], r) for a, b, r in model.correlations]
        cols = correlate(cols, correlation_matrix(d, pairs), rng)
    ev_fire = uniforms(trials, len(events), rng, "random")
    ev_amount = uniforms(trials, len(events), rng, method)
    values = {n: [model.inputs[n].ppf(u) for u in cols[i]] for i, n in enumerate(names)}
    fired: dict[str, list[bool]] = {e.name: [] for e in events}
    outcomes: list[Outcome] = []
    for i in range(trials):
        v = {n: values[n][i] for n in names}
        happened: dict[str, float] = {}
        for j, e in enumerate(events):
            hit = ev_fire[j][i] < e.probability
            fired[e.name].append(hit)
            if hit:
                happened[e.name] = e.amount.ppf(ev_amount[j][i]) if e.amount is not None else 0.0
        outcomes.append(evaluate(model, v, happened))
    return Run(model, trials, seed, method, values, fired, outcomes)
