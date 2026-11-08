import unittest

from roi_lens import UseCase, point_estimate, rank, simulate

SOLID = UseCase("support-copilot", users=200, hours_saved_per_user_month=(2, 4, 6),
                hourly_cost=40, adoption=(0.5, 0.7, 0.9), monthly_run_cost=3000, build_cost=60000)
WEAK = UseCase("niche-tool", users=10, hours_saved_per_user_month=(0.5, 1, 2),
               hourly_cost=40, adoption=(0.2, 0.4, 0.6), monthly_run_cost=2000, build_cost=60000)


class ROITests(unittest.TestCase):
    def test_point_estimate_matches_hand_math(self):
        # value = 200 * 0.7 * 4 * 40 = 22400 per month; net = (22400 - 3000) * 12 - 60000
        self.assertEqual(point_estimate(SOLID)["net"], 172800.0)

    def test_simulation_is_reproducible(self):
        self.assertEqual(simulate(SOLID, seed=3), simulate(SOLID, seed=3))

    def test_percentiles_are_ordered(self):
        r = simulate(SOLID)
        self.assertLessEqual(r["p10"], r["p50"])
        self.assertLessEqual(r["p50"], r["p90"])

    def test_strong_case_is_almost_surely_positive(self):
        self.assertGreater(simulate(SOLID)["prob_positive"], 0.99)

    def test_weak_case_loses_money(self):
        r = simulate(WEAK)
        self.assertLess(r["p50"], 0)
        self.assertEqual(r["prob_positive"], 0.0)

    def test_rank_puts_strong_case_first(self):
        self.assertEqual([r["name"] for r in rank([WEAK, SOLID])], ["support-copilot", "niche-tool"])

    def test_bad_range_rejected(self):
        bad = UseCase("x", 1, (3, 2, 1), 1, (0.1, 0.2, 0.3), 0, 1)
        with self.assertRaises(ValueError):
            simulate(bad, trials=1)


if __name__ == "__main__":
    unittest.main()
