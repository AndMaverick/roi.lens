import math
import random
import unittest

from roi_lens.distributions import (
    Empirical,
    Fixed,
    Lognormal,
    Normal,
    Pert,
    Triangular,
    Uniform,
    betainc,
    betainc_inv,
    from_ci90,
    from_dict,
)
from roi_lens.synthetic import random_distribution


def midpoint(f, n=4000):
    return sum(f((i + 0.5) / n) for i in range(n)) / n


class BetaTests(unittest.TestCase):
    def test_known_value(self):
        # Beta(3, 3): I_x = 10x^3 - 15x^4 + 6x^5, so at 0.25 it is exactly 0.103515625
        self.assertAlmostEqual(betainc(3, 3, 0.25), 0.103515625, places=12)

    def test_integer_shapes_match_binomial_sum(self):
        # for integer a, b: I_x(a, b) = P(Binomial(a + b - 1, x) >= a)
        rng = random.Random(1)
        for _ in range(300):
            a, b, x = rng.randint(1, 12), rng.randint(1, 12), rng.random()
            n = a + b - 1
            exact = sum(math.comb(n, j) * x ** j * (1 - x) ** (n - j) for j in range(a, n + 1))
            self.assertAlmostEqual(betainc(a, b, x), exact, places=10)

    def test_inverse_round_trip(self):
        rng = random.Random(2)
        for _ in range(500):
            a, b, p = 1 + rng.random() * 8, 1 + rng.random() * 8, rng.random()
            self.assertAlmostEqual(betainc(a, b, betainc_inv(a, b, p)), p, places=10)

    def test_symmetry(self):
        rng = random.Random(3)
        for _ in range(200):
            a, b, x = 1 + rng.random() * 6, 1 + rng.random() * 6, rng.random()
            self.assertAlmostEqual(betainc(a, b, x), 1 - betainc(b, a, 1 - x), places=10)


class DistributionTests(unittest.TestCase):
    def test_round_trip_and_mean_and_variance_for_random_distributions(self):
        rng = random.Random(4)
        for _ in range(60):
            d = random_distribution(rng)
            for p in (0.01, 0.2, 0.5, 0.8, 0.99):
                self.assertAlmostEqual(d.cdf(d.ppf(p)), p, places=7, msg=repr(d))
            m = midpoint(d.ppf)
            v = midpoint(lambda p, d=d, m=m: (d.ppf(p) - m) ** 2)
            self.assertAlmostEqual(m, d.mean(), delta=0.006 * max(1.0, abs(d.mean())), msg=repr(d))
            self.assertAlmostEqual(v, d.var(), delta=0.04 * d.var() + 1e-9, msg=repr(d))

    def test_ppf_is_monotone(self):
        rng = random.Random(5)
        for _ in range(60):
            d = random_distribution(rng)
            xs = [d.ppf(i / 200) for i in range(1, 200)]
            self.assertEqual(xs, sorted(xs))

    def test_triangular_closed_forms(self):
        t = Triangular(1, 2, 6)
        self.assertAlmostEqual(t.mean(), 3.0)
        self.assertAlmostEqual(t.cdf(2), 0.2)           # (m - a) / (c - a)
        self.assertAlmostEqual(t.var(), (1 + 4 + 36 - 2 - 6 - 12) / 18)

    def test_pert_mean_formula(self):
        p = Pert(2, 5, 14)
        self.assertAlmostEqual(p.mean(), (2 + 4 * 5 + 14) / 6)

    def test_truncated_normal_half_normal(self):
        h = Normal(0, 1, lower=0)
        self.assertAlmostEqual(h.mean(), math.sqrt(2 / math.pi), places=12)
        self.assertAlmostEqual(h.var(), 1 - 2 / math.pi, places=12)
        self.assertGreaterEqual(min(h.ppf(i / 1000) for i in range(1, 1000)), 0.0)

    def test_ci90_puts_five_percent_in_each_tail(self):
        for kind, lo, hi in (("normal", -3.0, 11.0), ("lognormal", 2.0, 40.0)):
            d = from_ci90(lo, hi, kind=kind)
            self.assertAlmostEqual(d.cdf(lo), 0.05, places=9)
            self.assertAlmostEqual(d.cdf(hi), 0.95, places=9)

    def test_empirical(self):
        e = Empirical([4, 1, 3, 2])
        self.assertEqual(e.mean(), 2.5)
        self.assertEqual(e.cdf(2), 0.5)
        self.assertEqual(e.ppf(0.0), 1)
        self.assertEqual(e.ppf(0.99), 4)

    def test_fixed(self):
        f = Fixed(7)
        self.assertEqual((f.ppf(0.3), f.mean(), f.var(), f.cdf(6.9), f.cdf(7)), (7, 7, 0, 0.0, 1.0))

    def test_bad_ranges_rejected(self):
        for build in (lambda: Uniform(2, 1), lambda: Triangular(1, 5, 3), lambda: Pert(3, 2, 1), lambda: Normal(0, -1),
                      lambda: Lognormal(0, -1), lambda: Empirical([]), lambda: from_ci90(5, 5), lambda: Normal(0, 1, lower=9)):
            with self.assertRaises(ValueError):
                build()


class FromDictTests(unittest.TestCase):
    def test_number_is_fixed(self):
        self.assertEqual(from_dict(5), Fixed(5.0))

    def test_each_kind(self):
        specs = [{"dist": "uniform", "low": 1, "high": 2}, {"dist": "triangular", "low": 1, "mode": 2, "high": 3},
                 {"dist": "pert", "low": 1, "mode": 2, "high": 3}, {"dist": "normal", "mu": 0, "sigma": 1},
                 {"dist": "lognormal", "mu": 0, "sigma": 1}, {"dist": "ci90", "low": 1, "high": 3}, {"dist": "fixed", "value": 3},
                 {"dist": "empirical", "values": [1, 2]}]
        for s in specs:
            self.assertTrue(hasattr(from_dict(s), "ppf"))

    def test_strict(self):
        for bad in ({"dist": "uniform", "low": 1, "high": 2, "colour": 1}, {"dist": "weibull"}, {"low": 1}, True, "5", [1],
                    {"dist": "uniform", "low": 1}):
            with self.assertRaises(ValueError):
                from_dict(bad)


if __name__ == "__main__":
    unittest.main()
