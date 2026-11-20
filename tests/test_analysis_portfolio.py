import math
import random
import unittest
from statistics import NormalDist

from roi_lens.analysis import breakeven, evpi, evppi, pilot_value, rank_drivers, tornado, value_of_information
from roi_lens.calibration import Answer, calibrate, required_answers, scale_interval
from roi_lens.distributions import Fixed, Normal, Uniform
from roi_lens.engine import evaluate, simulate
from roi_lens.expr import compile_expr
from roi_lens.model import Model
from roi_lens.portfolio import Case, derive_seed, plan
from roi_lens.synthetic import random_model

N = NormalDist()


def expected_positive_part(m, s):
    """E[max(X, 0)] for X ~ Normal(m, s): m Phi(m / s) + s phi(m / s)."""
    z = m / s
    return m * N.cdf(z) + s * N.pdf(z)


def linear_model(name="lin", a=None, b=None, horizon=1):
    a, b = a or Normal(-1, 2), b or Normal(0.5, 3)
    return Model(name, {"a": a, "b": b}, {"x": compile_expr("a + b")}, {}, {}, horizon_months=horizon, discount_rate=0.0)


class ValueOfInformationTests(unittest.TestCase):
    def setUp(self):
        rng = random.Random(3)
        self.n = 200000
        self.a = [rng.gauss(-1, 2) for _ in range(self.n)]
        self.b = [rng.gauss(0.5, 3) for _ in range(self.n)]
        self.y = [p + q for p, q in zip(self.a, self.b, strict=True)]

    def test_evpi_matches_closed_form(self):
        # NPV ~ Normal(-0.5, sqrt(13)); EVPI = E[max(NPV, 0)] - max(E[NPV], 0) = E[max(NPV, 0)] since the mean is negative
        value, se = evpi(self.y)
        self.assertAlmostEqual(value, expected_positive_part(-0.5, math.sqrt(13)), delta=4 * se)

    def test_evppi_matches_closed_form_for_each_input(self):
        # E[NPV | a] = a - 0.5 + 0.5 ... = a + 0.5 (the mean of b), so EVPPI(a) = E[max(a + 0.5, 0)] - max(-0.5, 0)
        self.assertAlmostEqual(evppi(self.a, self.y, 40), expected_positive_part(-0.5, 2), delta=0.02)
        self.assertAlmostEqual(evppi(self.b, self.y, 40), expected_positive_part(-0.5, 3), delta=0.02)

    def test_partial_never_exceeds_perfect(self):
        self.assertLessEqual(evppi(self.a, self.y, 20), evpi(self.y)[0] + 1e-6)
        self.assertGreaterEqual(evppi(self.a, self.y, 20), -1e-9)

    def test_uninformative_input_is_worth_about_nothing(self):
        rng = random.Random(9)
        noise = [rng.random() for _ in range(self.n)]
        self.assertLess(evppi(noise, self.y, 10), 0.02)

    def test_always_go_decision_has_no_information_value(self):
        self.assertEqual(evpi([5.0, 6.0, 7.0])[0], 0.0)
        self.assertEqual(evppi([1.0, 2.0, 3.0], [5.0, 6.0, 7.0], 3), 0.0)

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            evppi([1.0], [1.0, 2.0])

    def test_pilot_value_subtracts_cost(self):
        m = linear_model()
        run = simulate(m, 4000, 1)
        full = pilot_value(run, "a", 0.0)
        self.assertAlmostEqual(pilot_value(run, "a", 0.25), full - 0.25)
        with self.assertRaises(ValueError):
            pilot_value(run, "ghost", 1.0)

    def test_value_of_information_ranks_the_wider_input_first(self):
        run = simulate(linear_model(), 6000, 1)
        self.assertEqual(value_of_information(run)[0][0], "b")


class DriverTests(unittest.TestCase):
    def test_strongest_driver_first_with_the_right_sign(self):
        m = Model("d", {"big": Uniform(0, 10), "small": Uniform(0, 1), "cost": Uniform(0, 5)},
                  {"x": compile_expr("big * 100 + small")}, {"r": compile_expr("cost * 3")}, {}, horizon_months=1, discount_rate=0.0)
        d = rank_drivers(simulate(m, 3000, 2))
        self.assertEqual(d[0].name, "big")
        self.assertGreater(d[0].rank_correlation, 0.9)
        self.assertLess(next(x for x in d if x.name == "cost").rank_correlation, 0)

    def test_perfect_monotone_relation(self):
        m = Model("d", {"x": Uniform(1, 2)}, {"b": compile_expr("x ** 3")}, {}, {}, horizon_months=1, discount_rate=0.0)
        self.assertAlmostEqual(rank_drivers(simulate(m, 500, 1))[0].rank_correlation, 1.0, places=9)


class TornadoBreakevenTests(unittest.TestCase):
    def test_tornado_values_are_exact_evaluations(self):
        rng = random.Random(14)
        for i in range(40):
            m = random_model(rng, f"t{i}")
            base = {n: d.ppf(0.5) for n, d in m.inputs.items()}
            for s in tornado(m):
                self.assertAlmostEqual(s.low_result, evaluate(m, {**base, s.name: m.inputs[s.name].ppf(0.1)}).npv)
                self.assertAlmostEqual(s.high_result, evaluate(m, {**base, s.name: m.inputs[s.name].ppf(0.9)}).npv)
            swings = [s.swing for s in tornado(m)]
            self.assertEqual(swings, sorted(swings, reverse=True))

    def test_breakeven_lands_on_zero(self):
        rng = random.Random(15)
        found = 0
        for i in range(60):
            m = random_model(rng, f"b{i}")
            base = {n: d.ppf(0.5) for n, d in m.inputs.items()}
            for name in m.inputs:
                be = breakeven(m, name)
                if be.value is not None:
                    self.assertAlmostEqual(evaluate(m, {**base, name: be.value}).npv, 0.0, delta=1e-4 * max(1.0, abs(be.value)))
                    self.assertTrue(0.0 <= be.probability_worse <= 1.0)
                    found += 1
        self.assertGreater(found, 20)

    def test_breakeven_closed_form(self):
        # NPV = 12 months * (v - 10) - 300 = 0  =>  v = 35; v ~ Uniform(20, 60) so P(v < 35) = 0.375
        m = Model("c", {"v": Uniform(20, 60)}, {"b": compile_expr("v")}, {"r": compile_expr("10")}, {"o": compile_expr("300")},
                  horizon_months=12, discount_rate=0.0)
        be = breakeven(m, "v")
        self.assertAlmostEqual(be.value, 35.0, places=6)
        self.assertAlmostEqual(be.probability_worse, 0.375, places=6)
        self.assertEqual(be.direction, "higher is better")

    def test_breakeven_none_when_never_crossing(self):
        m = Model("c", {"v": Uniform(20, 60), "w": Fixed(1000)}, {"b": compile_expr("w + v * 0")}, {}, {}, horizon_months=3, discount_rate=0.0)
        self.assertIsNone(breakeven(m, "v").value)

    def test_unknown_input(self):
        with self.assertRaises(ValueError):
            breakeven(linear_model(), "ghost")
        with self.assertRaises(ValueError):
            tornado(linear_model(), lo=0.9, hi=0.1)


class CalibrationTests(unittest.TestCase):
    def test_hit_rate_and_verdicts(self):
        rng = random.Random(1)
        over = [Answer(0, 10, rng.gauss(5, 4)) for _ in range(300)]             # true sd 4 against a +/-5 interval: too narrow
        c = calibrate(over)
        self.assertEqual(c.verdict, "overconfident")
        self.assertLess(c.hit_rate, 0.85)
        ok = [Answer(-6.58, 6.58, rng.gauss(0, 4)) for _ in range(300)]          # +/- 1.645 sd: right
        self.assertEqual(calibrate(ok).verdict, "consistent with target")
        wide = [Answer(-20, 20, rng.gauss(0, 4)) for _ in range(300)]
        self.assertEqual(calibrate(wide).verdict, "underconfident")

    def test_scale_factor_fixes_the_in_sample_hit_rate(self):
        rng = random.Random(2)
        for _ in range(30):
            answers = [Answer(0, 10, rng.gauss(5, rng.uniform(2, 8))) for _ in range(rng.randint(10, 80))]
            c = calibrate(answers)
            fixed = [Answer(*scale_interval(a.low, a.high, c.scale_factor), a.truth) for a in answers]
            self.assertGreaterEqual(sum(a.hit for a in fixed) / len(fixed), 0.9 - 1e-9)

    def test_scale_interval(self):
        self.assertEqual(scale_interval(0, 10, 2), (-5, 15))
        with self.assertRaises(ValueError):
            scale_interval(0, 10, 0)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            Answer(5, 5, 5)
        with self.assertRaises(ValueError):
            calibrate([])

    def test_required_answers_is_a_sensible_size(self):
        n = required_answers(0.9, 0.10)
        self.assertTrue(20 <= n <= 60)
        self.assertLess(required_answers(0.9, 0.15), n)


class PortfolioTests(unittest.TestCase):
    def cases(self, k, rng, integer=True):
        out = []
        for i in range(k):
            gain = rng.randint(1, 9)
            m = Model(f"p{i}", {"v": Uniform(gain * 10 - 5, gain * 10 + 5)}, {"b": compile_expr("v")}, {}, {"o": compile_expr(str(rng.randint(20, 60)))},
                      horizon_months=12, discount_rate=0.0)
            out.append(Case(m, float(rng.randint(2, 9) * 10)))
        return out

    def knapsack(self, means, commits, budget):
        best = [0.0] * (int(budget) + 1)
        for mean, cost in zip(means, commits, strict=True):
            c = int(cost)
            for b in range(int(budget), c - 1, -1):
                best[b] = max(best[b], best[b - c] + mean)
        return best[int(budget)]

    def test_matches_dynamic_programming_oracle(self):
        rng = random.Random(5)
        for _ in range(25):
            cases = self.cases(rng.randint(3, 9), rng)
            budget = float(rng.randint(5, 25) * 10)
            result = plan(cases, budget, trials=200, seed=3)
            names = sorted(c.name for c in cases)
            means = [result.case_means[n] for n in names]
            commits = [next(c.commit for c in cases if c.name == n) for n in names]
            self.assertAlmostEqual(result.best.mean_npv, self.knapsack(means, commits, budget), places=6)
            self.assertLessEqual(result.best.commit, budget)

    def test_budget_always_respected_and_alternatives_sorted(self):
        rng = random.Random(6)
        cases = self.cases(7, rng)
        result = plan(cases, 150.0, trials=200)
        means = [o.mean_npv for o in result.alternatives]
        self.assertEqual(means, sorted(means, reverse=True))
        self.assertTrue(all(o.commit <= 150.0 for o in result.alternatives))

    def test_input_order_does_not_matter(self):
        rng = random.Random(7)
        cases = self.cases(6, rng)
        a = plan(cases, 120.0, trials=300, seed=2)
        b = plan(list(reversed(cases)), 120.0, trials=300, seed=2)
        self.assertEqual(a.best, b.best)

    def test_must_include_and_infeasible(self):
        rng = random.Random(8)
        cases = self.cases(5, rng)
        result = plan(cases, 1000.0, trials=100, must_include=["p0"])
        self.assertIn("p0", result.best.names)
        self.assertEqual(plan(cases, 1.0, trials=100).best.names, ())      # nothing fits, so the best plan is to do nothing
        with self.assertRaises(ValueError):
            plan(cases, 1.0, trials=100, must_include=["p0"])
        with self.assertRaises(ValueError):
            plan(cases, 1000.0, trials=100, must_include=["ghost"])

    def test_risk_averse_objective_prefers_the_safer_project(self):
        safe = Model("safe", {"v": Uniform(95, 105)}, {"b": compile_expr("v")}, {}, {"o": compile_expr("600")}, horizon_months=12, discount_rate=0.0)
        risky = Model("risky", {"v": Uniform(0, 240)}, {"b": compile_expr("v")}, {}, {"o": compile_expr("600")}, horizon_months=12, discount_rate=0.0)
        cases = [Case(safe, 100.0), Case(risky, 100.0)]
        self.assertEqual(plan(cases, 100.0, trials=2000, objective="mean").best.names, ("risky",))
        self.assertEqual(plan(cases, 100.0, trials=2000, objective="p10").best.names, ("safe",))

    def test_validation(self):
        rng = random.Random(9)
        cases = self.cases(2, rng)
        with self.assertRaises(ValueError):
            plan(cases, 100.0, objective="vibes")
        with self.assertRaises(ValueError):
            plan([], 100.0)
        with self.assertRaises(ValueError):
            plan([cases[0], cases[0]], 100.0)
        with self.assertRaises(ValueError):
            plan(self.cases(21, rng), 100.0)
        self.assertNotEqual(derive_seed(1, 0), derive_seed(1, 1))


if __name__ == "__main__":
    unittest.main()
