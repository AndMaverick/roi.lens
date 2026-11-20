import contextlib
import copy
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

import roi_lens
from roi_lens import UseCase, analyze, load, parse, point_estimate, render_markdown, render_text, simulate
from roi_lens.cli import main

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(list(argv))
        except SystemExit as exc:
            code = int(exc.code or 0)
    return code, out.getvalue(), err.getvalue()


def base():
    return json.loads((EX / "support_copilot.json").read_text())


class ScenarioTests(unittest.TestCase):
    def test_example_loads(self):
        sc = load(EX / "support_copilot.json")
        self.assertEqual(len(sc.model.inputs), 9)
        self.assertEqual(len(sc.model.events), 4)
        self.assertIn("min_prob_positive", sc.gates)

    def test_unknown_keys_are_rejected_everywhere(self):
        mutations = [
            lambda d: d.update(horizon="36"),
            lambda d: d["inputs"].update(bad={"dist": "uniform", "low": 1, "high": 2, "typo": 3}),
            lambda d: d["events"][0].update(prob=0.2),
            lambda d: d["gates"].update(min_vibes=1),
            lambda d: d["correlations"][0].update(extra=1),
        ]
        for mutate in mutations:
            d = base()
            mutate(d)
            with self.assertRaises(ValueError):
                parse(d)

    def test_type_and_value_errors(self):
        cases = [("horizon_months", "36"), ("horizon_months", 36.5), ("discount_rate", "ten percent"), ("ramp", "cubic"),
                 ("build_months", -1), ("build_months", 99), ("benefit_decay_per_year", 1.5)]
        for key, value in cases:
            d = base()
            d[key] = value
            with self.assertRaises(ValueError, msg=f"{key}={value!r}"):
                parse(d)

    def test_formula_problems_name_the_formula(self):
        d = base()
        d["benefits"]["x"] = "__import__('os')"
        with self.assertRaises(ValueError) as cm:
            parse(d)
        self.assertIn("benefits.x", str(cm.exception))
        d = base()
        d["run_costs"]["y"] = "unknown_input * 2"
        with self.assertRaises(ValueError) as cm:
            parse(d)
        self.assertIn("unknown_input", str(cm.exception))

    def test_numbers_allowed_as_formulas(self):
        d = base()
        d["run_costs"]["flat"] = 1000
        self.assertIn("flat", parse(d).model.run_costs)

    def test_missing_pieces(self):
        for key in ("name", "inputs", "benefits"):
            d = base()
            del d[key]
            with self.assertRaises(ValueError):
                parse(d)
        with self.assertRaises(ValueError):
            parse([])

    def test_impossible_correlations_fail_at_run_time_not_silently(self):
        d = base()
        d["inputs"] = {"a": 1, "b": {"dist": "uniform", "low": 1, "high": 2}, "c": {"dist": "uniform", "low": 1, "high": 2}}
        d["benefits"] = {"x": "a + b + c"}
        d["run_costs"], d["one_time_costs"], d["events"] = {}, {}, []
        d["correlations"] = [{"a": "a", "b": "b", "rho": 0.9}, {"a": "b", "b": "c", "rho": 0.9}, {"a": "a", "b": "c", "rho": -0.9}]
        with self.assertRaises(ValueError):
            analyze(parse(d), 200)

    def test_fingerprint_depends_on_content_and_settings_but_not_key_order(self):
        d = base()
        a = parse(d).fingerprint(1000, 7, "lhs", "0.2.0")
        reordered = json.loads(json.dumps(d, sort_keys=True))
        self.assertEqual(a, parse(reordered).fingerprint(1000, 7, "lhs", "0.2.0"))
        self.assertNotEqual(a, parse(d).fingerprint(1001, 7, "lhs", "0.2.0"))
        self.assertNotEqual(a, parse(d).fingerprint(1000, 8, "lhs", "0.2.0"))
        self.assertNotEqual(a, parse(d).fingerprint(1000, 7, "random", "0.2.0"))
        self.assertNotEqual(a, parse(d).fingerprint(1000, 7, "lhs", "0.2.1"))
        d2 = copy.deepcopy(d)
        d2["inputs"]["agents"]["high"] = 71
        self.assertNotEqual(a, parse(d2).fingerprint(1000, 7, "lhs", "0.2.0"))
        self.assertEqual(len(a), 64)

    def test_load_reports_the_path(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "bad.json"
            p.write_text("{not json")
            with self.assertRaises(ValueError) as cm:
                load(p)
            self.assertIn("bad.json", str(cm.exception))


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        self.a = analyze(load(EX / "support_copilot.json"), 1500, 5)

    def test_deterministic(self):
        self.assertEqual(self.a, analyze(load(EX / "support_copilot.json"), 1500, 5))

    def test_json_serialisable_and_complete(self):
        text = json.dumps(self.a)
        for key in ("npv", "payback", "roi", "drivers", "tornado", "value_of_information", "breakeven", "gates", "fingerprint"):
            self.assertIn(key, json.loads(text))

    def test_internal_consistency(self):
        n = self.a["npv"]
        self.assertLessEqual(n["p10"], n["p50"])
        self.assertLessEqual(n["p50"], n["p90"])
        self.assertLessEqual(n["prob_positive_ci_low"], n["prob_positive"])
        self.assertGreaterEqual(n["prob_positive_ci_high"], n["prob_positive"])
        self.assertLessEqual(n["cvar05"], n["p05"])
        self.assertAlmostEqual(sum(self.a["events"].values()) > 0 and 1, 1)
        self.assertEqual(self.a["passed"], all(g["passed"] for g in self.a["gates"]))

    def test_renderers_carry_the_headline_numbers_and_no_em_dashes(self):
        text, md = render_text(self.a), render_markdown(self.a)
        for out in (text, md):
            self.assertIn(self.a["fingerprint"], out)
            self.assertNotIn("\u2014", out)
        self.assertIn("Expected NPV", text)
        self.assertIn("| Measure | Value |", md)

    def test_gate_logic(self):
        d = base()
        d["gates"] = {"min_expected_npv": 1e12, "max_payback_p90_months": 100}
        r = analyze(parse(d), 500, 1)
        status = {g["name"]: g["passed"] for g in r["gates"]}
        self.assertFalse(status["min_expected_npv"])
        self.assertTrue(status["max_payback_p90_months"])
        self.assertFalse(r["passed"])

    def test_two_methods_agree_within_error(self):
        a = analyze(load(EX / "support_copilot.json"), 4000, 3, "lhs")["npv"]
        b = analyze(load(EX / "support_copilot.json"), 4000, 3, "random")["npv"]
        self.assertLess(abs(a["mean"] - b["mean"]), 5 * (a["mean_se"] ** 2 + b["mean_se"] ** 2) ** 0.5)


class CliTests(unittest.TestCase):
    def test_passing_example_exits_zero(self):
        code, out, _ = run_cli("analyze", str(EX / "support_copilot.json"), "--trials", "1500")
        self.assertEqual(code, 0)
        self.assertIn("PASS", out)

    def test_failing_example_exits_one(self):
        code, out, _ = run_cli("analyze", str(EX / "marginal_copilot.json"), "--trials", "1500")
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out)

    def test_json_and_markdown_formats(self):
        code, out, _ = run_cli("analyze", str(EX / "support_copilot.json"), "--trials", "800", "--format", "json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["trials"], 800)
        code, out, _ = run_cli("analyze", str(EX / "support_copilot.json"), "--trials", "800", "--format", "markdown")
        self.assertTrue(out.startswith("# "))

    def test_bad_input_exits_two(self):
        self.assertEqual(run_cli("analyze", "/nonexistent.json")[0], 2)
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.json"
            p.write_text('{"name": "x"}')
            code, _, err = run_cli("analyze", str(p))
            self.assertEqual(code, 2)
            self.assertIn("error:", err)
        self.assertEqual(run_cli("analyze", str(EX / "support_copilot.json"), "--trials", "1")[0], 2)

    def test_fingerprint_command_matches_library(self):
        code, out, _ = run_cli("fingerprint", str(EX / "support_copilot.json"), "--trials", "900", "--seed", "4")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), load(EX / "support_copilot.json").fingerprint(900, 4, "lhs", roi_lens.__version__))

    def test_sensitivity_command(self):
        code, out, _ = run_cli("sensitivity", str(EX / "support_copilot.json"), "--trials", "1500", "--pilot-cost", "5000")
        self.assertEqual(code, 0)
        self.assertIn("deflect_rate", out)
        self.assertIn("pilot on this input", out)
        code, out, _ = run_cli("sensitivity", str(EX / "support_copilot.json"), "--trials", "800", "--format", "json")
        self.assertIn("rows", json.loads(out))

    def test_calibrate_command(self):
        rng = random.Random(1)
        rows = [{"low": 0, "high": 10, "truth": rng.gauss(5, 4)} for _ in range(100)]
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "a.json"
            p.write_text(json.dumps({"answers": rows}))
            code, out, _ = run_cli("calibrate", str(p))
            self.assertEqual(code, 0)
            self.assertIn("overconfident", out)
            p.write_text(json.dumps(rows))
            self.assertEqual(run_cli("calibrate", str(p), "--format", "json")[0], 0)

    def test_portfolio_command(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "p.json"
            p.write_text(json.dumps({"budget": 500000, "cases": [
                {"scenario": str(EX / "support_copilot.json"), "commit": 200000},
                {"scenario": str(EX / "marginal_copilot.json"), "commit": 200000}]}))
            code, out, _ = run_cli("portfolio", str(p), "--trials", "800")
            self.assertEqual(code, 0)
            self.assertIn("Best:", out)
            p.write_text(json.dumps({"budget": 1, "cases": [], "colour": 1}))
            self.assertEqual(run_cli("portfolio", str(p))[0], 2)

    def test_version(self):
        code, out, _ = run_cli("--version")
        self.assertEqual(code, 0)
        self.assertIn(roi_lens.__version__, out)


class LegacyApiTests(unittest.TestCase):
    """The 0.1 functions keep their exact behaviour; ``simulate`` dispatches on what it is given."""

    CASE = UseCase("support-copilot", users=200, hours_saved_per_user_month=(2, 4, 6), hourly_cost=40, adoption=(0.5, 0.7, 0.9),
                   monthly_run_cost=3000, build_cost=60000)

    def test_readme_values_unchanged(self):
        self.assertEqual(point_estimate(self.CASE), {"net": 172800.0, "roi": 2.88})
        r = simulate(self.CASE)
        self.assertEqual((r["p10"], r["p50"], r["p90"], r["prob_positive"]), (89602.26, 170260.0, 257907.36, 1.0))

    def test_simulate_dispatches_to_the_new_engine_for_models(self):
        run = simulate(load(EX / "support_copilot.json").model, 200, 1)
        self.assertEqual(run.trials, 200)

    def test_version_string(self):
        self.assertEqual(roi_lens.__version__, "0.2.0")


if __name__ == "__main__":
    unittest.main()
