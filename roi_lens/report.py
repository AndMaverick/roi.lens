"""Run a scenario end to end and render the result as text, markdown or JSON."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ._version import __version__
from .analysis import breakeven, evpi, rank_drivers, tornado, value_of_information
from .engine import evaluate, simulate
from .scenario import Scenario
from .stats import mean_se, quantile, quantile_nearest, summarize

IRR_SAMPLE = 500


@dataclass(frozen=True)
class Gate:
    name: str
    threshold: float
    actual: float
    passed: bool


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if not (math.isnan(v) or math.isinf(v))]


def analyze(scenario: Scenario, trials: int = 5000, seed: int = 7, method: str = "lhs") -> dict[str, Any]:
    """The full analysis as plain data (JSON-ready). Deterministic for a given scenario, trials, seed and method."""
    model = scenario.model
    run = simulate(model, trials, seed, method)
    npv = run.metric("npv")
    s = summarize(npv)
    base = evaluate(model, {n: d.ppf(0.5) for n, d in model.inputs.items()})
    payback = sorted(run.metric("payback"))
    paid = sum(1 for p in payback if math.isfinite(p))
    rois = _finite(run.metric("roi"))
    irrs = _finite([o.irr for o in run.outcomes[:IRR_SAMPLE]])
    ev, ev_se = evpi(npv)
    voi = value_of_information(run)
    swings = tornado(model)
    p90_payback = quantile_nearest(payback, 0.90)
    actuals = {"min_expected_npv": s["mean"], "min_prob_positive": s["prob_positive"], "min_p10_npv": s["p10"],
               "max_payback_p90_months": p90_payback, "min_cvar05_npv": s["cvar05"]}
    gates = []
    for key, threshold in sorted(scenario.gates.items()):
        actual = actuals[key]
        ok = actual <= threshold if key.startswith("max_") else actual >= threshold
        gates.append(Gate(key, threshold, actual, ok))
    return {
        "name": model.name, "version": __version__, "trials": trials, "seed": seed, "method": method,
        "fingerprint": scenario.fingerprint(trials, seed, method, __version__),
        "base_case": {"npv": base.npv, "net": base.net, "payback_months": base.payback, "note": "every input at its median, no risk events"},
        "npv": s,
        "payback": {"prob_within_horizon": paid / len(payback), "p50_months": quantile_nearest(payback, 0.50), "p90_months": p90_payback,
                    "horizon_months": model.horizon_months},
        "roi": {"mean": mean_se(rois)[0] if rois else None, "p10": quantile(sorted(rois), 0.1) if rois else None,
                "p50": quantile(sorted(rois), 0.5) if rois else None, "p90": quantile(sorted(rois), 0.9) if rois else None},
        "irr_annual": {"p50": quantile(sorted(irrs), 0.5) if irrs else None, "defined_in": len(irrs), "sampled": min(IRR_SAMPLE, trials)},
        "events": {name: sum(flags) / len(flags) for name, flags in run.fired.items()},
        "drivers": [{"name": d.name, "rank_correlation": d.rank_correlation} for d in rank_drivers(run)[:8]],
        "tornado": [{"name": w.name, "low_input": w.low_input, "high_input": w.high_input, "npv_at_p10": w.low_result,
                     "npv_at_p90": w.high_result, "swing": w.swing} for w in swings[:8]],
        "value_of_information": {"evpi": ev, "evpi_se": ev_se, "evppi": [{"name": n, "value": v} for n, v in voi[:5]]},
        "breakeven": [{"input": b.name, "value": b.value, "probability_worse": b.probability_worse, "direction": b.direction}
                      for b in (breakeven(model, w.name) for w in swings[:3])],
        "gates": [{"name": g.name, "threshold": g.threshold, "actual": g.actual, "passed": g.passed} for g in gates],
        "passed": all(g.passed for g in gates),
    }


def _m(x: float | None) -> str:
    if x is None:
        return "n/a"
    if math.isinf(x):
        return "never"
    return f"{x:,.0f}"


def _months(x: float) -> str:
    return "not within the horizon" if math.isinf(x) else f"{x:.0f} months"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _gate_value(name: str) -> Any:
    if name == "min_prob_positive":
        return _pct
    if name == "max_payback_p90_months":
        return _months
    return _m


def render_text(a: dict[str, Any]) -> str:
    n, p, v = a["npv"], a["payback"], a["value_of_information"]
    lines = [f"{a['name']}  (roi.lens {a['version']}, {a['trials']} trials, seed {a['seed']}, {a['method']})", "",
             f"Expected NPV        {_m(n['mean'])}  (standard error {_m(n['mean_se'])})",
             f"Base case NPV       {_m(a['base_case']['npv'])}  ({a['base_case']['note']})",
             f"P(NPV > 0)          {_pct(n['prob_positive'])}  (95% interval {_pct(n['prob_positive_ci_low'])} to {_pct(n['prob_positive_ci_high'])})",
             f"NPV p10 / p50 / p90 {_m(n['p10'])} / {_m(n['p50'])} / {_m(n['p90'])}",
             f"Median NPV, 95% CI  {_m(n['p50_ci_low'])} to {_m(n['p50_ci_high'])}",
             f"Bad-case average    {_m(n['cvar05'])}  (mean of the worst 5% of outcomes)",
             f"Payback             {_pct(p['prob_within_horizon'])} pay back within {p['horizon_months']} months; "
             f"median {_months(p['p50_months'])}, p90 {_months(p['p90_months'])}"]
    if a["roi"]["p50"] is not None:
        lines.append(f"ROI p10 / p50 / p90 {a['roi']['p10']:.2f} / {a['roi']['p50']:.2f} / {a['roi']['p90']:.2f}")
    if a["irr_annual"]["p50"] is not None:
        lines.append(f"Median IRR          {100 * a['irr_annual']['p50']:.0f}% a year (from {a['irr_annual']['defined_in']} of "
                     f"{a['irr_annual']['sampled']} sampled trials)")
    if a["events"]:
        lines += ["", "Risk events (share of trials in which each happened): " + ", ".join(f"{k} {_pct(x)}" for k, x in a["events"].items())]
    lines += ["", "What moves the answer (rank correlation with NPV)"]
    lines += [f"  {d['name']:<28}{d['rank_correlation']:+.2f}" for d in a["drivers"][:5]]
    lines += ["", "Tornado (NPV with this input at its p10 and p90, all others at the median)"]
    lines += [f"  {w['name']:<28}{_m(w['npv_at_p10']):>12} to {_m(w['npv_at_p90']):>12}" for w in a["tornado"][:5]]
    lines += ["", f"Value of information: perfect information is worth at most {_m(v['evpi'])} (+/- {_m(v['evpi_se'])}).",
              "  Learning one input exactly is worth: " + ", ".join(f"{e['name']} {_m(e['value'])}" for e in v["evppi"][:3])]
    lines += ["", "Break-even (NPV = 0, others at median)"]
    for b in a["breakeven"]:
        lines.append(f"  {b['input']:<28}" + ("none in range" if b["value"] is None
                                              else f"{b['value']:.4g}  (chance the input is on the losing side: {_pct(b['probability_worse'])})"))
    if a["gates"]:
        lines += ["", "Decision gates"]
        for g in a["gates"]:
            fmt = _gate_value(g["name"])
            lines.append(f"  {'PASS' if g['passed'] else 'FAIL'}  {g['name']}: {fmt(g['actual'])} against {fmt(g['threshold'])}")
    lines += ["", f"Fingerprint {a['fingerprint']}", "Same fingerprint on any machine gives the same numbers."]
    return "\n".join(lines)


def render_markdown(a: dict[str, Any]) -> str:
    n, p, v = a["npv"], a["payback"], a["value_of_information"]
    out = [f"# {a['name']}", "", f"*roi.lens {a['version']}, {a['trials']} trials, seed {a['seed']}, {a['method']} sampling*", "",
           "| Measure | Value |", "|---|---|",
           f"| Expected NPV | {_m(n['mean'])} (standard error {_m(n['mean_se'])}) |",
           f"| Probability NPV is positive | {_pct(n['prob_positive'])} ({_pct(n['prob_positive_ci_low'])} to {_pct(n['prob_positive_ci_high'])}) |",
           f"| NPV p10 / p50 / p90 | {_m(n['p10'])} / {_m(n['p50'])} / {_m(n['p90'])} |",
           f"| Average of the worst 5% | {_m(n['cvar05'])} |",
           f"| Pays back within {p['horizon_months']} months | {_pct(p['prob_within_horizon'])} (median {_months(p['p50_months'])}, p90 {_months(p['p90_months'])}) |",
           f"| Value of perfect information | {_m(v['evpi'])} |", "", "## What moves the answer", "", "| Input | Rank correlation with NPV |", "|---|---|"]
    out += [f"| {d['name']} | {d['rank_correlation']:+.2f} |" for d in a["drivers"][:6]]
    if a["gates"]:
        out += ["", "## Decision gates", "", "| Gate | Result | Threshold | Status |", "|---|---|---|---|"]
        out += [f"| {g['name']} | {_gate_value(g['name'])(g['actual'])} | {_gate_value(g['name'])(g['threshold'])} | "
                f"{'pass' if g['passed'] else 'FAIL'} |" for g in a["gates"]]
    out += ["", f"Fingerprint `{a['fingerprint']}`"]
    return "\n".join(out)
