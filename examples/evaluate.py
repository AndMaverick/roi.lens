"""Reproduces every verification figure quoted in the README. Run: python examples/evaluate.py

Each line compares the library with an answer computed a different way (a closed form, a second implementation, or Python itself), so a
mismatch means a bug, not noise. Everything is seeded, so the numbers are the same on every machine."""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roi_lens import (  # noqa: E402
    Answer,
    Model,
    analyze,
    calibrate,
    compile_expr,
    evaluate,
    evpi,
    evppi,
    load,
    scale_interval,
    simulate,
)
from roi_lens.distributions import Normal, Uniform, betainc, betainc_inv  # noqa: E402
from roi_lens.stats import quantile_ci, wilson  # noqa: E402
from roi_lens.synthetic import random_expression, random_model  # noqa: E402

N = NormalDist()
ROOT = Path(__file__).resolve().parent


def positive_part(m: float, s: float) -> float:
    z = m / s
    return m * N.cdf(z) + s * N.pdf(z)


def line(label: str, detail: str) -> None:
    print(f"{label:<46}{detail}")


def main() -> int:
    t0 = time.time()
    rng = random.Random(1)

    worst = 0.0
    for _ in range(20000):
        a, b, p = 1 + rng.random() * 8, 1 + rng.random() * 8, rng.random()
        worst = max(worst, abs(betainc(a, b, betainc_inv(a, b, p)) - p))
    line("PERT quantile solver, 20000 random shapes", f"largest CDF residual {worst:.1e}")

    bad = 0
    for _ in range(2000):
        a, b, x = rng.randint(1, 15), rng.randint(1, 15), rng.random()
        n = a + b - 1
        exact = sum(math.comb(n, j) * x ** j * (1 - x) ** (n - j) for j in range(a, n + 1))
        bad += abs(betainc(a, b, x) - exact) > 1e-9
    line("incomplete beta vs exact binomial sum", f"{2000 - bad}/2000 agree to 1e-9")

    names, good, total = ["a", "b", "c"], 0, 0
    for _ in range(5000):
        src = random_expression(rng, names)
        env = {n: round(rng.uniform(-6, 9), 2) for n in names}
        try:
            want = float(eval(src, {"__builtins__": {}}, {"min": min, "max": max, "abs": abs, **env}))  # noqa: S307 (oracle)
        except ZeroDivisionError:
            continue
        total += 1
        good += abs(compile_expr(src)(env) - want) <= 1e-9 * max(1.0, abs(want))
    line("formula evaluator vs Python eval", f"{good}/{total} random formulas identical")

    a_s = [rng.gauss(-1, 2) for _ in range(300000)]
    b_s = [rng.gauss(0.5, 3) for _ in range(300000)]
    y = [p + q for p, q in zip(a_s, b_s, strict=True)]
    ev, se = evpi(y)
    line("EVPI, sum of two normals", f"{ev:.4f} against closed form {positive_part(-0.5, math.sqrt(13)):.4f} (std. error {se:.4f})")
    line("EVPPI of the first input", f"{evppi(a_s, y, 40):.4f} against closed form {positive_part(-0.5, 2):.4f}")
    line("EVPPI of the second input", f"{evppi(b_s, y, 40):.4f} against closed form {positive_part(-0.5, 3):.4f}")

    m = Model("n", {"a": Normal(5, 2), "b": Normal(-1, 3)}, {"x": compile_expr("a + b")}, {}, {}, horizon_months=1, discount_rate=0.0)
    npv = sorted(simulate(m, 40000, 3).metric("npv"))
    errs = [abs(npv[int(q * 40000)] - (4 + math.sqrt(13) * N.inv_cdf(q))) for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
    line("simulated quantiles of a normal sum", f"largest error {max(errs):.3f} on a spread of {math.sqrt(13):.2f}")

    def spread(method: str) -> float:
        mu = Model("p", {"a": Uniform(0, 10), "b": Uniform(0, 10)}, {"x": compile_expr("a + b")}, {}, {}, horizon_months=1, discount_rate=0.0)
        means = [sum(simulate(mu, 100, s, method).metric("npv")) / 100 for s in range(200)]
        c = sum(means) / len(means)
        return sum((v - c) ** 2 for v in means) / len(means)
    line("Latin hypercube vs simple random (mean)", f"variance of the estimate {spread('random') / spread('lhs'):.0f}x lower")

    cover, reps = 0, 2000
    for _ in range(reps):
        s = sorted(rng.random() for _ in range(101))
        lo, hi = quantile_ci(s, 0.5)
        cover += lo <= 0.5 <= hi
    line("median confidence interval coverage", f"{100 * cover / reps:.1f}% (nominal 95%)")
    cover = 0
    for _ in range(reps):
        k = sum(rng.random() < 0.3 for _ in range(80))
        lo, hi = wilson(k, 80)
        cover += lo <= 0.3 <= hi
    line("Wilson interval coverage", f"{100 * cover / reps:.1f}% (nominal 95%)")

    worst = 0.0
    for i in range(1500):
        mdl = random_model(rng, f"m{i}")
        v = {n: d.ppf(rng.random()) for n, d in mdl.inputs.items()}
        mm = mdl.monthly_discount
        benefit = sum(e(v) for e in mdl.benefits.values())
        run = sum(e(v) for e in mdl.run_costs.values())
        one = sum(e(v) for e in mdl.one_time_costs.values())
        total_pv = 0.0
        for t in range(mdl.horizon_months + 1):
            flow = -one if (t == 0 and mdl.build_months == 0) else 0.0
            if 1 <= t <= mdl.build_months:
                flow -= one / mdl.build_months
            k = t - mdl.build_months
            if k >= 1:
                r = 1.0 if mdl.ramp_months == 0 else min(1.0, k / mdl.ramp_months)
                if mdl.ramp == "smooth":
                    r = r * r * (3 - 2 * r)
                flow += benefit * r * (1 - mdl.benefit_decay_per_year) ** ((k - 1) / 12) - run
            total_pv += flow / (1 + mm) ** t
        worst = max(worst, abs(evaluate(mdl, v).npv - total_pv) / max(1.0, abs(total_pv)))
    line("cash-flow engine vs month-by-month oracle", f"1500 random models, largest relative gap {worst:.1e}")

    rng2, ok = random.Random(9), 0
    for _ in range(300):
        answers = [Answer(0, 10, rng2.gauss(5, rng2.uniform(2, 8))) for _ in range(rng2.randint(10, 80))]
        c = calibrate(answers)
        fixed = [Answer(*scale_interval(a.low, a.high, c.scale_factor), a.truth) for a in answers]
        ok += sum(a.hit for a in fixed) / len(fixed) >= 0.9 - 1e-9
    line("calibration fix reaches 90% in-sample", f"{ok}/300 trials")

    sc = load(ROOT / "support_copilot.json")
    t1 = time.time()
    r = analyze(sc, 5000, 7)
    line("full analysis, 9 inputs, 4 risks, 5000 trials", f"{time.time() - t1:.2f}s, P(NPV > 0) {100 * r['npv']['prob_positive']:.1f}%")
    again = analyze(sc, 5000, 7)
    line("repeat run is identical", "yes" if again == r else "NO")
    print(f"\nTotal {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
