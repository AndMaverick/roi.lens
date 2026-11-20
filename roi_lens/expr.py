"""A small, safe arithmetic language for scenario formulas.

Scenario files are data, so a formula such as ``agents * tickets * deflect_rate * cost_per_ticket`` must never reach ``eval``. This module
parses with ``ast`` and accepts only numbers, named inputs, ``+ - * / // % **``, unary signs, comparisons, ``a if c else b``, and a
short list of functions. Anything else (attribute access, subscripts, lambdas, strings, imports, comprehensions) is rejected before any
evaluation happens."""
from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Mapping

MAX_SOURCE = 400
MAX_NODES = 200


class ExprError(ValueError):
    """The formula is not allowed, or cannot be evaluated for the values given."""


_BIN: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_CMP: dict[type, Callable[[float, float], bool]] = {
    ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
}
_FUNCS: dict[str, Callable[..., float]] = {
    "min": min, "max": max, "abs": abs, "sqrt": math.sqrt, "log": math.log, "exp": math.exp, "ceil": math.ceil, "floor": math.floor,
}
Env = Mapping[str, float]
Node = Callable[[Env], float]


class Expr:
    """A validated formula. ``names`` lists the inputs it reads; ``__call__`` evaluates it for a mapping of input values."""

    def __init__(self, source: str) -> None:
        if not isinstance(source, str) or not source.strip():
            raise ExprError("a formula must be a non-empty string")
        if len(source) > MAX_SOURCE:
            raise ExprError(f"formula is longer than {MAX_SOURCE} characters")
        self.source = source
        try:
            tree = ast.parse(source.strip(), mode="eval")
        except SyntaxError as exc:
            raise ExprError(f"cannot parse {source!r}: {exc.msg}") from exc
        if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
            raise ExprError("formula is too large")
        self.names: set[str] = set()
        self._fn = self._build(tree.body)

    def __call__(self, env: Env) -> float:
        try:
            out = self._fn(env)
        except ZeroDivisionError as exc:
            raise ExprError(f"division by zero in {self.source!r}") from exc
        except OverflowError as exc:
            raise ExprError(f"overflow in {self.source!r}") from exc
        except ValueError as exc:
            raise ExprError(f"{exc} in {self.source!r}") from exc
        if isinstance(out, complex):
            raise ExprError(f"{self.source!r} produced a complex number")
        return float(out)

    def _build(self, node: ast.AST) -> Node:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ExprError("only numbers are allowed as constants")
            value = float(node.value)
            return lambda env: value
        if isinstance(node, ast.Name):
            name = node.id
            self.names.add(name)

            def read(env: Env) -> float:
                if name not in env:
                    raise ExprError(f"unknown name {name!r}")
                return env[name]
            return read
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            inner = self._build(node.operand)
            return (lambda env: -inner(env)) if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right, op = self._build(node.left), self._build(node.right), _BIN[type(node.op)]
            return lambda env: op(left(env), right(env))
        if isinstance(node, ast.Compare):
            parts = [self._build(node.left), *[self._build(c) for c in node.comparators]]
            ops = []
            for o in node.ops:
                if type(o) not in _CMP:
                    raise ExprError("only < <= > >= == != comparisons are allowed")
                ops.append(_CMP[type(o)])

            def compare(env: Env) -> float:
                vals = [p(env) for p in parts]
                return float(all(f(vals[i], vals[i + 1]) for i, f in enumerate(ops)))
            return compare
        if isinstance(node, ast.IfExp):
            test, yes, no = self._build(node.test), self._build(node.body), self._build(node.orelse)
            return lambda env: yes(env) if test(env) else no(env)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                raise ExprError(f"only these functions are allowed: {', '.join(sorted(_FUNCS))}")
            if node.keywords:
                raise ExprError("keyword arguments are not allowed")
            fn = _FUNCS[node.func.id]
            args = [self._build(a) for a in node.args]
            if not args or (node.func.id not in ("min", "max") and len(args) != 1 and not (node.func.id == "log" and len(args) == 2)):
                raise ExprError(f"wrong number of arguments to {node.func.id}")
            return lambda env: float(fn(*[a(env) for a in args]))
        raise ExprError(f"{type(node).__name__} is not allowed in a formula")


def compile_expr(source: str) -> Expr:
    return Expr(source)
