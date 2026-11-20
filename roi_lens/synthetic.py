"""Seeded generators of random expressions and random models, used to test the library against independent oracles."""
from __future__ import annotations

import random

from .distributions import Distribution, Lognormal, Normal, Pert, Triangular, Uniform
from .expr import compile_expr
from .model import Event, Model


def random_expression(rng: random.Random, names: list[str], depth: int = 3) -> str:
    """A random formula over ``names`` that is valid both here and in plain Python, so Python's own ``eval`` can act as the oracle."""
    if depth == 0 or rng.random() < 0.25:
        if names and rng.random() < 0.6:
            return rng.choice(names)
        return repr(round(rng.uniform(0.5, 9.5), 2))
    kind = rng.random()
    if kind < 0.55:
        return f"({random_expression(rng, names, depth - 1)} {rng.choice('+-*/')} {random_expression(rng, names, depth - 1)})"
    if kind < 0.65:
        return f"(-{random_expression(rng, names, depth - 1)})"
    if kind < 0.8:
        return f"{rng.choice(['min', 'max'])}({random_expression(rng, names, depth - 1)}, {random_expression(rng, names, depth - 1)})"
    if kind < 0.88:
        return f"abs({random_expression(rng, names, depth - 1)})"
    a, b = random_expression(rng, names, depth - 1), random_expression(rng, names, depth - 1)
    return f"({a} if {random_expression(rng, names, depth - 1)} > {rng.choice(['1', '2', '5'])} else {b})"


def random_distribution(rng: random.Random) -> Distribution:
    kind = rng.choice(["uniform", "triangular", "pert", "normal", "lognormal", "truncated"])
    lo = rng.uniform(1, 50)
    span = rng.uniform(1, 40)
    if kind == "uniform":
        return Uniform(lo, lo + span)
    if kind == "triangular":
        return Triangular(lo, lo + rng.random() * span, lo + span)
    if kind == "pert":
        return Pert(lo, lo + rng.random() * span, lo + span, rng.choice([2.0, 4.0, 6.0]))
    if kind == "normal":
        return Normal(lo + span / 2, span / 4)
    if kind == "lognormal":
        return Lognormal(rng.uniform(0, 3), rng.uniform(0.1, 0.6))
    return Normal(lo + span / 2, span / 3, lower=lo, upper=lo + span)


def random_model(rng: random.Random, name: str = "random") -> Model:
    """A random but valid model: a few inputs, one benefit, some costs, and sometimes timing and risk structure."""
    names = [f"x{i}" for i in range(rng.randint(2, 5))]
    inputs = {n: random_distribution(rng) for n in names}
    benefit = f"{rng.choice(names)} * {rng.choice(names)} * {rng.choice([5, 10, 20])}"
    run = f"{rng.choice(names)} * {rng.choice([2, 5, 10])}" if rng.random() < 0.8 else None
    one = f"{rng.choice(names)} * {rng.choice([50, 100, 400])}"
    events = []
    if rng.random() < 0.7:
        events.append(Event("slip", rng.uniform(0.05, 0.5), "delay", Uniform(1, 4)))
    if rng.random() < 0.5:
        events.append(Event("incident", rng.uniform(0.05, 0.4), "benefit_factor", Uniform(0.4, 0.9)))
    if rng.random() < 0.4:
        events.append(Event("surprise", rng.uniform(0.05, 0.4), "extra_cost", Uniform(500, 3000), rng.randint(1, 10)))
    if rng.random() < 0.3:
        events.append(Event("axe", rng.uniform(0.02, 0.2), "cancel", None, rng.randint(2, 12)))
    horizon = rng.randint(12, 48)
    return Model(
        name=name, inputs=inputs, benefits={"b": compile_expr(benefit)}, run_costs={"r": compile_expr(run)} if run else {},
        one_time_costs={"o": compile_expr(one)}, horizon_months=horizon, build_months=rng.randint(0, min(6, horizon - 1)),
        ramp_months=rng.randint(0, 8), ramp=rng.choice(["linear", "smooth"]), benefit_decay_per_year=rng.choice([0.0, 0.05, 0.15]),
        discount_rate=rng.choice([0.0, 0.08, 0.12]), events=tuple(events))
