import math
import random
import unittest

from roi_lens.sampling import (
    cholesky,
    correlate,
    correlation_matrix,
    pearson_for_spearman,
    ranks,
    spearman,
    spearman_for_pearson,
    uniforms,
)
from roi_lens.stats import cvar, mean_se, quantile, quantile_ci, quantile_nearest, summarize, wilson


class SamplingTests(unittest.TestCase):
    def test_lhs_puts_exactly_one_point_in_every_stratum(self):
        n = 500
        for col in uniforms(n, 4, random.Random(1), "lhs"):
            self.assertEqual(sorted(int(u * n) for u in col), list(range(n)))

    def test_random_method_is_in_open_unit_interval(self):
        for col in uniforms(1000, 2, random.Random(2), "random"):
            self.assertTrue(all(0 < u < 1 for u in col))

    def test_lhs_beats_simple_random_on_the_mean(self):
        # the sample mean of an LHS column has far lower variance than a simple random one
        def spread(method):
            means = [sum(uniforms(200, 1, random.Random(s), method)[0]) / 200 for s in range(300)]
            m = sum(means) / len(means)
            return sum((x - m) ** 2 for x in means) / len(means)
        self.assertLess(spread("lhs"), spread("random") / 20)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            uniforms(0, 1, random.Random(1))
        with self.assertRaises(ValueError):
            uniforms(5, 1, random.Random(1), "sobol")

    def test_conversion_between_pearson_and_spearman_is_inverse(self):
        for r in (-0.9, -0.5, 0.0, 0.3, 0.8):
            self.assertAlmostEqual(spearman_for_pearson(pearson_for_spearman(r)), r, places=12)
        self.assertAlmostEqual(pearson_for_spearman(0.5), 2 * math.sin(math.pi / 12))

    def test_ranks_with_ties(self):
        self.assertEqual(ranks([10, 20, 20, 30]), [1.0, 2.5, 2.5, 4.0])

    def test_spearman_known_values(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertEqual(spearman([1, 1, 1], [1, 2, 3]), 0.0)

    def test_cholesky_reconstructs(self):
        m = [[1, 0.5, 0.2], [0.5, 1, -0.3], [0.2, -0.3, 1]]
        low = cholesky(m)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(sum(low[i][k] * low[j][k] for k in range(3)), m[i][j], places=12)

    def test_impossible_correlations_rejected(self):
        m = correlation_matrix(3, [(0, 1, 0.9), (1, 2, 0.9), (0, 2, -0.9)])
        with self.assertRaises(ValueError):
            cholesky(m)
        with self.assertRaises(ValueError):
            correlation_matrix(2, [(0, 1, 1.0)])
        with self.assertRaises(ValueError):
            correlation_matrix(2, [(0, 0, 0.5)])

    def test_correlate_keeps_values_and_hits_target(self):
        rng = random.Random(3)
        cols = uniforms(3000, 3, rng)
        out = correlate(cols, correlation_matrix(3, [(0, 1, 0.7), (1, 2, -0.5), (0, 2, 0.1)]), rng)
        for a, b in zip(cols, out, strict=True):
            self.assertEqual(sorted(a), sorted(b))
        self.assertAlmostEqual(spearman(out[0], out[1]), 0.7, delta=0.03)
        self.assertAlmostEqual(spearman(out[1], out[2]), -0.5, delta=0.03)
        self.assertAlmostEqual(spearman(out[0], out[2]), 0.1, delta=0.04)


class StatsTests(unittest.TestCase):
    def test_quantile_type7(self):
        v = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(quantile(v, 0.0), 1.0)
        self.assertEqual(quantile(v, 1.0), 4.0)
        self.assertEqual(quantile(v, 0.5), 2.5)
        self.assertAlmostEqual(quantile(v, 0.25), 1.75)
        self.assertEqual(quantile([5.0], 0.3), 5.0)

    def test_quantile_nearest_handles_infinity(self):
        v = [1.0, 2.0, 3.0, math.inf]
        self.assertEqual(quantile_nearest(v, 0.5), 2.0)
        self.assertEqual(quantile_nearest(v, 0.9), math.inf)

    def test_wilson_known_value_and_edges(self):
        lo, hi = wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertEqual(wilson(0, 10)[0], 0.0)
        self.assertEqual(wilson(10, 10)[1], 1.0)
        self.assertGreater(wilson(0, 10)[1], 0.0)

    def test_wilson_coverage(self):
        rng = random.Random(4)
        for p in (0.1, 0.5, 0.9):
            cover = 0
            reps = 3000
            for _ in range(reps):
                k = sum(rng.random() < p for _ in range(60))
                lo, hi = wilson(k, 60)
                cover += lo <= p <= hi
            self.assertGreaterEqual(cover / reps, 0.93)

    def test_quantile_ci_coverage_at_least_nominal(self):
        rng = random.Random(5)
        for q in (0.1, 0.5, 0.9):
            true = q  # uniform(0, 1): the q-quantile is q
            cover = 0
            reps = 1500
            for _ in range(reps):
                s = sorted(rng.random() for _ in range(101))
                lo, hi = quantile_ci(s, q)
                cover += lo <= true <= hi
            self.assertGreaterEqual(cover / reps, 0.945, msg=f"q={q}")

    def test_cvar(self):
        self.assertEqual(cvar(list(range(1, 101)), 0.05), 3.0)
        self.assertEqual(cvar([5.0], 0.05), 5.0)

    def test_mean_se(self):
        m, se = mean_se([2.0, 4.0, 6.0, 8.0])
        self.assertEqual(m, 5.0)
        self.assertAlmostEqual(se, math.sqrt(20 / 3 / 4))

    def test_summarize_fields_consistent(self):
        s = summarize([float(i) for i in range(-50, 150)])
        self.assertLessEqual(s["p05"], s["p10"])
        self.assertLessEqual(s["p10"], s["p50"])
        self.assertLessEqual(s["p50"], s["p90"])
        self.assertLessEqual(s["p50_ci_low"], s["p50"])
        self.assertGreaterEqual(s["p50_ci_high"], s["p50"])
        self.assertAlmostEqual(s["prob_positive"], 149 / 200)
        self.assertLessEqual(s["prob_positive_ci_low"], s["prob_positive"])


if __name__ == "__main__":
    unittest.main()
