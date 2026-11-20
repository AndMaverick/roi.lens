import math
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path
from statistics import NormalDist

from roi_lens.distributions import Fixed, Normal, Uniform
from roi_lens.engine import build_flow, evaluate, measure, simulate
from roi_lens.expr import compile_expr
from roi_lens.model import Event, Model, irr, ramp_factor
from roi_lens.stats import wilson
from roi_lens.synthetic import random_model

ROOT = Path(__file__).resolve().parent.parent


def simple(**kw):
    base = {"name": "m", "inputs": {"v": Fixed(100.0)}, "benefits": {"b": compile_expr("v")}, "run_costs": {"r": compile_expr("10")},
            "one_time_costs": {"o": compile_expr("300")}, "horizon_months": 12, "discount_rate": 0.0}
    base.update(kw)
    return Model(**base)


def oracle_npv(model, v, delay=0, factor=1.0, extras=(), cancel=None):
    """A second, independent calculation of NPV written from the definition, month by month."""
    rate = (1 + model.discount_rate) ** (1 / 12) - 1
    benefit = sum(e(v) for e in model.benefits.values())
    run = sum(e(v) for e in model.run_costs.values())
    one = sum(e(v) for e in model.one_time_costs.values())
    total = 0.0
    for t in range(0, model.horizon_months + 1):
        if cancel is not None and t > cancel:
            continue
        flow = 0.0
        if t == 0 and model.build_months == 0:
            flow -= one
        if 1 <= t <= model.build_months:
            flow -= one / model.build_months
        k = t - (model.build_months + delay)
        if k >= 1:
            ramp = 1.0 if model.ramp_months == 0 else min(1.0, k / model.ramp_months)
            if model.ramp == "smooth":
                ramp = ramp * ramp * (3 - 2 * ramp)
            flow += benefit * ramp * (1 - model.benefit_decay_per_year) ** ((k - 1) / 12) * factor - run
        for month, amount in extras:
            if t == min(month, model.horizon_months):
                flow -= amount
        total += flow / (1 + rate) ** t
    return total


class FlowTests(unittest.TestCase):
    def test_hand_computed_flat_case(self):
        o = evaluate(simple(), {"v": 100.0})
        # 12 months of (100 - 10) less 300 up front
        self.assertAlmostEqual(o.npv, 12 * 90 - 300)
        self.assertEqual(o.net, 12 * 90 - 300)
        self.assertAlmostEqual(o.roi, (12 * 100 - 12 * 10 - 300) / (12 * 10 + 300))
        self.assertEqual(o.payback, 4.0)          # cumulative: -300 + 90 t >= 0 at t = 4 (3.33 rounds up)

    def test_build_phase_spreads_cost_and_delays_benefit(self):
        o = evaluate(simple(build_months=3), {"v": 100.0})
        self.assertEqual(o.cashflow[:5], [0.0, -100.0, -100.0, -100.0, 90.0])
        self.assertAlmostEqual(o.npv, 9 * 90 - 300)

    def test_never_pays_back(self):
        o = evaluate(simple(), {"v": 5.0})
        self.assertEqual(o.payback, math.inf)

    def test_no_upfront_loss_means_payback_zero(self):
        o = evaluate(simple(one_time_costs={}), {"v": 100.0})
        self.assertEqual(o.payback, 0.0)

    def test_ramp_factors(self):
        self.assertEqual([ramp_factor(k, 4, "linear") for k in (0, 1, 2, 4, 9)], [0.0, 0.25, 0.5, 1.0, 1.0])
        self.assertAlmostEqual(ramp_factor(2, 4, "smooth"), 0.5)
        self.assertAlmostEqual(ramp_factor(1, 4, "smooth"), 0.25 ** 2 * (3 - 0.5))
        self.assertEqual(ramp_factor(1, 0, "linear"), 1.0)

    def test_decay_halves_over_the_right_span(self):
        m = simple(one_time_costs={}, run_costs={}, benefit_decay_per_year=0.5, horizon_months=13)
        f = build_flow(m, {"v": 100.0})
        self.assertAlmostEqual(f.benefit[1], 100.0)
        self.assertAlmostEqual(f.benefit[13], 50.0)

    def test_discounting_matches_annual_rate(self):
        m = simple(run_costs={}, one_time_costs={}, discount_rate=0.12, horizon_months=12)
        o = evaluate(m, {"v": 100.0})
        rate = 1.12 ** (1 / 12) - 1
        self.assertAlmostEqual(o.npv, sum(100 / (1 + rate) ** t for t in range(1, 13)))

    def test_events_change_the_flow_as_documented(self):
        m = simple(events=(Event("d", 0.5, "delay", Fixed(2)), Event("f", 0.5, "benefit_factor", Fixed(0.5)),
                           Event("x", 0.5, "extra_cost", Fixed(70), 3), Event("c", 0.5, "cancel", None, 8)))
        none = evaluate(m, {"v": 100.0})
        self.assertAlmostEqual(none.npv, 12 * 90 - 300)
        self.assertAlmostEqual(evaluate(m, {"v": 100.0}, {"d": 1.2}).npv, 10 * 90 - 300)        # 1.2 months rounds up to 2
        self.assertAlmostEqual(evaluate(m, {"v": 100.0}, {"f": 0.5}).npv, 12 * 40 - 300)
        self.assertAlmostEqual(evaluate(m, {"v": 100.0}, {"x": 70}).npv, 12 * 90 - 300 - 70)
        self.assertAlmostEqual(evaluate(m, {"v": 100.0}, {"c": 0}).npv, 8 * 90 - 300)          # nothing after month 8

    def test_bad_evaluate_arguments(self):
        with self.assertRaises(ValueError):
            evaluate(simple(), {})
        with self.assertRaises(ValueError):
            evaluate(simple(), {"v": 1.0}, {"nope": 1.0})

    def test_model_validation(self):
        with self.assertRaises(ValueError):
            simple(benefits={"b": compile_expr("zzz")})
        with self.assertRaises(ValueError):
            simple(build_months=12)
        with self.assertRaises(ValueError):
            simple(ramp="cubic")
        with self.assertRaises(ValueError):
            simple(benefit_decay_per_year=1.0)
        with self.assertRaises(ValueError):
            simple(benefits={})
        with self.assertRaises(ValueError):
            simple(correlations=(("v", "ghost", 0.2),))
        with self.assertRaises(ValueError):
            simple(events=(Event("e", 0.1, "cancel"), Event("e", 0.2, "cancel")))
        with self.assertRaises(ValueError):
            Event("e", 1.5, "cancel")
        with self.assertRaises(ValueError):
            Event("e", 0.5, "delay")

    def test_matches_independent_oracle_on_random_models(self):
        rng = random.Random(21)
        for i in range(400):
            m = random_model(rng, f"m{i}")
            v = {n: d.ppf(rng.random()) for n, d in m.inputs.items()}
            fired = {}
            delay, factor, extras, cancel = 0, 1.0, [], None
            for e in m.events:
                if rng.random() < 0.5:
                    amt = e.amount.ppf(rng.random()) if e.amount else 0.0
                    fired[e.name] = amt
                    if e.kind == "delay":
                        delay += max(0, math.ceil(amt))
                    elif e.kind == "benefit_factor":
                        factor *= amt
                    elif e.kind == "extra_cost":
                        extras.append((e.month or 1, amt))
                    else:
                        when = e.month or max(1, math.ceil(m.horizon_months / 2))
                        cancel = when if cancel is None else min(cancel, when)
            got = evaluate(m, v, fired).npv
            want = oracle_npv(m, v, delay, factor, tuple(extras), cancel)
            self.assertAlmostEqual(got, want, delta=1e-7 * max(1.0, abs(want)), msg=f"model {i}")


class IrrTests(unittest.TestCase):
    def test_annuity_closed_form(self):
        # pay 1000 now, receive 100 a month for 12 months: the monthly rate r solves 1000 = 100 * (1 - (1 + r) ** -12) / r
        flows = [-1000.0] + [100.0] * 12
        monthly = (1 + irr(flows)) ** (1 / 12) - 1
        self.assertAlmostEqual(1000 - 100 * (1 - (1 + monthly) ** -12) / monthly, 0.0, places=8)

    def test_npv_is_zero_at_irr_for_random_flows(self):
        rng = random.Random(8)
        done = 0
        for _ in range(300):
            flows = [-rng.uniform(100, 1000)] + [rng.uniform(20, 200) for _ in range(rng.randint(3, 30))]
            r = irr(flows)
            if math.isnan(r):
                continue
            monthly = (1 + r) ** (1 / 12) - 1
            self.assertAlmostEqual(sum(c / (1 + monthly) ** t for t, c in enumerate(flows)), 0.0, delta=1e-6)
            done += 1
        self.assertGreater(done, 250)

    def test_undefined_without_sign_change(self):
        self.assertTrue(math.isnan(irr([1.0, 2.0, 3.0])))
        self.assertTrue(math.isnan(irr([-1.0, -2.0])))


class SimulationTests(unittest.TestCase):
    def test_same_arguments_same_run(self):
        m = random_model(random.Random(31))
        a, b = simulate(m, 400, 5), simulate(m, 400, 5)
        self.assertEqual(a.metric("npv"), b.metric("npv"))
        self.assertNotEqual(a.metric("npv"), simulate(m, 400, 6).metric("npv"))

    def test_result_does_not_depend_on_key_order(self):
        m = random_model(random.Random(32))
        shuffled = Model(**{**m.__dict__, "inputs": dict(reversed(list(m.inputs.items())))})
        self.assertEqual(simulate(m, 300, 9).metric("npv"), simulate(shuffled, 300, 9).metric("npv"))

    def test_identical_across_hash_seeds(self):
        code = ("import json,sys;sys.path.insert(0,sys.argv[1]);import roi_lens as r;"
                "s=r.load(sys.argv[1]+'/examples/support_copilot.json');a=r.analyze(s,600,3);print(a['npv']['mean'],a['fingerprint'])")
        outs = set()
        for seed in ("0", "1", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            outs.add(subprocess.run([sys.executable, "-c", code, str(ROOT)], env=env, capture_output=True, text=True, check=True).stdout)
        self.assertEqual(len(outs), 1)

    def test_independent_product_mean(self):
        # E[a * b] = E[a] E[b] for independent inputs; benefit counted over 1 month, no costs
        a, b = Uniform(2, 6), Normal(10, 2)
        m = Model("p", {"a": a, "b": b}, {"x": compile_expr("a * b")}, {}, {}, horizon_months=1, discount_rate=0.0)
        run = simulate(m, 20000, 1, "random")
        npv = run.metric("npv")
        mean = sum(npv) / len(npv)
        se = math.sqrt(sum((x - mean) ** 2 for x in npv) / len(npv) / len(npv))
        self.assertLess(abs(mean - a.mean() * b.mean()), 4 * se)

    def test_lhs_estimates_the_mean_more_tightly_than_simple_random(self):
        m = Model("p", {"a": Uniform(0, 10), "b": Uniform(0, 10)}, {"x": compile_expr("a + b")}, {}, {}, horizon_months=1, discount_rate=0.0)

        def spread(method):
            means = [sum(simulate(m, 100, s, method).metric("npv")) / 100 for s in range(120)]
            c = sum(means) / len(means)
            return sum((x - c) ** 2 for x in means) / len(means)
        self.assertLess(spread("lhs"), spread("random") / 5)

    def test_event_frequencies_match_probabilities(self):
        m = simple(events=(Event("a", 0.2, "extra_cost", Fixed(1)), Event("b", 0.7, "extra_cost", Fixed(1))))
        run = simulate(m, 5000, 4)
        for name, p in (("a", 0.2), ("b", 0.7)):
            k = sum(run.fired[name])
            lo, hi = wilson(k, 5000, 0.999)
            self.assertTrue(lo <= p <= hi, msg=name)

    def test_correlated_inputs_move_together(self):
        m = Model("c", {"a": Uniform(0, 1), "b": Uniform(0, 1)}, {"x": compile_expr("a + b")}, {}, {}, horizon_months=1,
                  discount_rate=0.0, correlations=(("a", "b", 0.8),))
        from roi_lens.sampling import spearman
        run = simulate(m, 4000, 2)
        self.assertAlmostEqual(spearman(run.values["a"], run.values["b"]), 0.8, delta=0.03)

    def test_value_at_risk_of_sum_of_normals(self):
        # NPV = a + b over one month with no costs: a sum of independent normals, so quantiles are known exactly
        m = Model("n", {"a": Normal(5, 2), "b": Normal(-1, 3)}, {"x": compile_expr("a + b")}, {}, {}, horizon_months=1, discount_rate=0.0)
        npv = sorted(simulate(m, 20000, 3).metric("npv"))
        sd = math.sqrt(13)
        for q in (0.1, 0.5, 0.9):
            want = 4 + sd * NormalDist().inv_cdf(q)
            self.assertAlmostEqual(npv[int(q * 20000)], want, delta=0.06)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            simulate(simple(), 1)
        with self.assertRaises(ValueError):
            simulate(simple(), 10).metric("bogus")

    def test_measure_is_consistent_with_build_flow(self):
        m = simple(build_months=2, ramp_months=3)
        o = measure(m, build_flow(m, {"v": 50.0}))
        self.assertAlmostEqual(o.npv, o.pv_benefit - o.pv_cost)


if __name__ == "__main__":
    unittest.main()
