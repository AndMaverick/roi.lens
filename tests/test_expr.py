import random
import unittest

from roi_lens.expr import ExprError, compile_expr
from roi_lens.synthetic import random_expression


class ExprTests(unittest.TestCase):
    def test_matches_python_eval_on_random_expressions(self):
        rng = random.Random(11)
        names = ["a", "b", "c"]
        checked = errors = 0
        for _ in range(3000):
            src = random_expression(rng, names)
            env = {n: round(rng.uniform(-6, 9), 2) for n in names}
            try:
                want = float(eval(src, {"__builtins__": {}}, {"min": min, "max": max, "abs": abs, **env}))  # noqa: S307 (oracle, test only)
            except ZeroDivisionError:
                with self.assertRaises(ExprError):
                    compile_expr(src)(env)
                errors += 1
                continue
            got = compile_expr(src)(env)
            self.assertAlmostEqual(got, want, delta=1e-9 * max(1.0, abs(want)), msg=src)
            checked += 1
        self.assertGreater(checked, 2500)
        self.assertGreater(errors, 0)

    def test_names_collected(self):
        self.assertEqual(compile_expr("a * b + min(c, 2)").names, {"a", "b", "c"})

    def test_precedence_and_power(self):
        self.assertEqual(compile_expr("2 + 3 * 4 ** 2 / 8")({}), 8.0)
        self.assertEqual(compile_expr("2 ** 3 ** 2")({}), 512.0)
        self.assertEqual(compile_expr("-2 ** 2")({}), -4.0)

    def test_conditionals_and_comparisons(self):
        e = compile_expr("a if b > 1 else c")
        self.assertEqual(e({"a": 1, "b": 2, "c": 9}), 1.0)
        self.assertEqual(e({"a": 1, "b": 0, "c": 9}), 9.0)
        self.assertEqual(compile_expr("1 < a <= 3")({"a": 3}), 1.0)
        self.assertEqual(compile_expr("1 < a <= 3")({"a": 4}), 0.0)

    def test_functions(self):
        self.assertAlmostEqual(compile_expr("sqrt(16) + log(exp(2)) + ceil(2.1) + floor(2.9) + abs(-1)")({}), 4 + 2 + 3 + 2 + 1)

    def test_unsafe_and_unsupported_forms_rejected(self):
        for bad in ("__import__('os').system('x')", "a.b", "a[0]", "lambda: 1", "'text'", "[1, 2]", "open('f')", "a if", "True", "None",
                    "1 and 2", "not a", "a is b", "(x for x in y)", "f(a)", "min(a, key=abs)", "eval('1')", "a := 1", "", "   ", "x" * 500,
                    "1 + (" * 120 + "1" + ")" * 120, "abs()", "sqrt(1, 2)", "{1: 2}", "a @ b", "a << 2", "a & 1", "~a", "f'{a}'"):
            with self.assertRaises(ExprError, msg=bad):
                compile_expr(bad)

    def test_runtime_errors_are_expr_errors(self):
        for src, env in (("1 / a", {"a": 0}), ("a % b", {"a": 1, "b": 0}), ("sqrt(a)", {"a": -1}), ("log(a)", {"a": 0}),
                         ("exp(a)", {"a": 10000}), ("a ** b", {"a": -8, "b": 0.5}), ("x", {})):
            with self.assertRaises(ExprError, msg=src):
                compile_expr(src)(env)


if __name__ == "__main__":
    unittest.main()
