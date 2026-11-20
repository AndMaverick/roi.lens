"""Scenario files: strict JSON in, a validated Model and decision gates out. Unknown keys are errors, so a typo cannot silently change an answer."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .distributions import from_dict
from .expr import ExprError, compile_expr
from .model import Event, Model

GATE_KEYS = {"min_expected_npv", "min_prob_positive", "min_p10_npv", "max_payback_p90_months", "min_cvar05_npv"}
TOP_KEYS = {"name", "description", "horizon_months", "build_months", "ramp_months", "ramp", "benefit_decay_per_year", "discount_rate",
            "inputs", "benefits", "run_costs", "one_time_costs", "events", "correlations", "gates"}


@dataclass(frozen=True)
class Scenario:
    model: Model
    gates: dict[str, float]
    raw: dict[str, Any]

    def fingerprint(self, trials: int, seed: int, method: str, version: str) -> str:
        """SHA-256 of the scenario, the run settings and the library version. Same fingerprint, same numbers, on any machine."""
        canon = json.dumps({"scenario": self.raw, "trials": trials, "seed": seed, "method": method, "version": version},
                           sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canon.encode("ascii")).hexdigest()


def _need(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ValueError(f"{where}: missing required key {key!r}")
    return d[key]


def _number(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where}: expected a number, got {v!r}")
    return float(v)


def _int(v: Any, where: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{where}: expected a whole number, got {v!r}")
    return v


def _exprs(d: Any, where: str) -> dict[str, Any]:
    if not isinstance(d, dict):
        raise ValueError(f"{where}: expected an object of name to formula")
    out = {}
    for k, v in d.items():
        try:
            out[k] = compile_expr(v) if isinstance(v, str) else compile_expr(repr(_number(v, f"{where}.{k}")))
        except ExprError as exc:
            raise ValueError(f"{where}.{k}: {exc}") from exc
    return out


def parse(raw: Any) -> Scenario:
    if not isinstance(raw, dict):
        raise ValueError("a scenario must be a JSON object")
    extra = sorted(set(raw) - TOP_KEYS)
    if extra:
        raise ValueError(f"unknown key {extra[0]!r} (allowed: {', '.join(sorted(TOP_KEYS))})")
    inputs_raw = _need(raw, "inputs", "scenario")
    if not isinstance(inputs_raw, dict) or not inputs_raw:
        raise ValueError("inputs: expected a non-empty object")
    inputs = {k: from_dict(v, f"inputs.{k}") for k, v in inputs_raw.items()}
    events = []
    for i, e in enumerate(raw.get("events", [])):
        where = f"events[{i}]"
        if not isinstance(e, dict):
            raise ValueError(f"{where}: expected an object")
        bad = sorted(set(e) - {"name", "probability", "kind", "amount", "month"})
        if bad:
            raise ValueError(f"{where}: unknown key {bad[0]!r}")
        amount = from_dict(e["amount"], f"{where}.amount") if "amount" in e else None
        events.append(Event(str(_need(e, "name", where)), _number(_need(e, "probability", where), f"{where}.probability"),
                            str(_need(e, "kind", where)), amount, _int(e["month"], f"{where}.month") if "month" in e else None))
    corr = []
    for i, c in enumerate(raw.get("correlations", [])):
        where = f"correlations[{i}]"
        if not isinstance(c, dict) or set(c) != {"a", "b", "rho"}:
            raise ValueError(f"{where}: expected exactly the keys a, b, rho")
        corr.append((str(c["a"]), str(c["b"]), _number(c["rho"], f"{where}.rho")))
    gates_raw = raw.get("gates", {})
    if not isinstance(gates_raw, dict):
        raise ValueError("gates: expected an object")
    bad = sorted(set(gates_raw) - GATE_KEYS)
    if bad:
        raise ValueError(f"gates: unknown gate {bad[0]!r} (allowed: {', '.join(sorted(GATE_KEYS))})")
    model = Model(
        name=str(_need(raw, "name", "scenario")), inputs=inputs, benefits=_exprs(_need(raw, "benefits", "scenario"), "benefits"),
        run_costs=_exprs(raw.get("run_costs", {}), "run_costs"), one_time_costs=_exprs(raw.get("one_time_costs", {}), "one_time_costs"),
        horizon_months=_int(raw.get("horizon_months", 36), "horizon_months"), build_months=_int(raw.get("build_months", 0), "build_months"),
        ramp_months=_int(raw.get("ramp_months", 0), "ramp_months"), ramp=str(raw.get("ramp", "linear")),
        benefit_decay_per_year=_number(raw.get("benefit_decay_per_year", 0.0), "benefit_decay_per_year"),
        discount_rate=_number(raw.get("discount_rate", 0.10), "discount_rate"), events=tuple(events), correlations=tuple(corr))
    return Scenario(model, {k: _number(v, f"gates.{k}") for k, v in gates_raw.items()}, json.loads(json.dumps(raw)))


def load(path: str | Path) -> Scenario:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc
    try:
        return parse(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
