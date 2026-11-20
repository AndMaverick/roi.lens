"""Command line: ``python -m roi_lens analyze scenario.json``. Exit 0 when every gate passes, 1 when a gate fails, 2 on a bad input."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ._version import __version__
from .analysis import breakeven, evppi, tornado
from .calibration import Answer, calibrate
from .engine import simulate
from .portfolio import Case, plan
from .report import analyze, render_markdown, render_text
from .scenario import load, parse


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--trials", type=int, default=5000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--method", choices=("lhs", "random"), default="lhs")


def _build() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="roi-lens", description="Decision-grade ROI for AI projects.")
    p.add_argument("--version", action="version", version=f"roi.lens {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="full analysis of a scenario, with decision gates")
    a.add_argument("scenario")
    a.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    _common(a)
    s = sub.add_parser("sensitivity", help="tornado, value of information and break-evens")
    s.add_argument("scenario")
    s.add_argument("--pilot-cost", type=float, default=None, help="cost of a pilot that would pin one input down exactly")
    s.add_argument("--format", choices=("text", "json"), default="text")
    _common(s)
    f = sub.add_parser("portfolio", help="choose among projects under a budget")
    f.add_argument("portfolio")
    f.add_argument("--format", choices=("text", "json"), default="text")
    _common(f)
    c = sub.add_parser("calibrate", help="check whether your 90 percent intervals are right 90 percent of the time")
    c.add_argument("answers")
    c.add_argument("--format", choices=("text", "json"), default="text")
    fp = sub.add_parser("fingerprint", help="print the reproducibility fingerprint of a scenario and run settings")
    fp.add_argument("scenario")
    _common(fp)
    return p


def _emit(data: Any, fmt: str, text: str) -> None:
    print(json.dumps(data, indent=2, sort_keys=True) if fmt == "json" else text)


def _sensitivity(args: argparse.Namespace) -> int:
    sc = load(args.scenario)
    run = simulate(sc.model, args.trials, args.seed, args.method)
    npv = run.metric("npv")
    swings = tornado(sc.model)
    rows: list[dict[str, Any]] = []
    for w in swings:
        value = evppi(run.values[w.name], npv)
        rows.append({"input": w.name, "npv_at_p10": w.low_result, "npv_at_p90": w.high_result, "swing": w.swing, "evppi": value,
                     "pilot_net_value": None if args.pilot_cost is None else value - args.pilot_cost,
                     "breakeven": breakeven(sc.model, w.name).value})
    lines = [f"{'input':<28}{'NPV at p10':>14}{'NPV at p90':>14}{'EVPPI':>12}{'break-even':>14}"]
    for r in rows:
        be = "none" if r["breakeven"] is None else f"{r['breakeven']:.4g}"
        lines.append(f"{r['input']:<28}{r['npv_at_p10']:>14,.0f}{r['npv_at_p90']:>14,.0f}{r['evppi']:>12,.0f}{be:>14}")
        net = r["pilot_net_value"]
        if net is not None:
            verdict = "worth it" if net > 0 else "not worth it"
            lines.append(f"{'':<28}pilot on this input at {args.pilot_cost:,.0f}: {net:,.0f} net ({verdict})")
    _emit({"scenario": sc.model.name, "rows": rows}, args.format, "\n".join(lines))
    return 0


def _portfolio(args: argparse.Namespace) -> int:
    path = Path(args.portfolio)
    raw = json.loads(path.read_text(encoding="utf-8"))
    extra = sorted(set(raw) - {"budget", "objective", "cases", "must_include"})
    if extra:
        raise ValueError(f"unknown key {extra[0]!r}")
    cases = []
    for i, c in enumerate(raw["cases"]):
        bad = sorted(set(c) - {"scenario", "commit"})
        if bad:
            raise ValueError(f"cases[{i}]: unknown key {bad[0]!r}")
        sc = parse(c["scenario"]) if isinstance(c["scenario"], dict) else load(path.parent / c["scenario"])
        cases.append(Case(sc.model, float(c["commit"])))
    result = plan(cases, float(raw["budget"]), trials=args.trials, seed=args.seed, method=args.method,
                  objective=raw.get("objective", "mean"), must_include=raw.get("must_include", []))
    lines = [f"Budget {result.budget:,.0f}, objective {result.objective}, {result.feasible_subsets} feasible combinations",
             "", "Best: " + (", ".join(result.best.names) or "(none)"),
             f"  commits {result.best.commit:,.0f}; expected NPV {result.best.mean_npv:,.0f}; "
             f"p10 {result.best.p10:,.0f}, p50 {result.best.p50:,.0f}, p90 {result.best.p90:,.0f}; P(NPV>0) {100 * result.best.prob_positive:.1f}%",
             "", "Per project expected NPV: " + ", ".join(f"{k} {v:,.0f}" for k, v in sorted(result.case_means.items())), "", "Alternatives"]
    lines += [f"  {', '.join(o.names) or '(none)'}: commits {o.commit:,.0f}, expected NPV {o.mean_npv:,.0f}, p10 {o.p10:,.0f}"
              for o in result.alternatives[1:]]
    data = {"budget": result.budget, "objective": result.objective, "feasible_subsets": result.feasible_subsets, "case_means": result.case_means,
            "best": result.best.__dict__, "alternatives": [o.__dict__ for o in result.alternatives]}
    _emit(data, args.format, "\n".join(lines))
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    raw = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    rows = raw["answers"] if isinstance(raw, dict) else raw
    result = calibrate([Answer(float(r["low"]), float(r["high"]), float(r["truth"])) for r in rows])
    text = (f"{result.hits} of {result.n} intervals contained the truth: {100 * result.hit_rate:.1f}% against a target of 90%\n"
            f"95% interval for your true hit rate: {100 * result.ci_low:.1f}% to {100 * result.ci_high:.1f}%\n"
            f"Verdict: {result.verdict}\n"
            f"Scale your 90% interval widths by {result.scale_factor:.2f} to be right 90% of the time")
    _emit(result.__dict__, args.format, text)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build().parse_args(argv)
    try:
        if args.command == "analyze":
            result = analyze(load(args.scenario), args.trials, args.seed, args.method)
            _emit(result, args.format, render_markdown(result) if args.format == "markdown" else render_text(result))
            return 0 if result["passed"] else 1
        if args.command == "sensitivity":
            return _sensitivity(args)
        if args.command == "portfolio":
            return _portfolio(args)
        if args.command == "calibrate":
            return _calibrate(args)
        print(load(args.scenario).fingerprint(args.trials, args.seed, args.method, __version__))
        return 0
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
