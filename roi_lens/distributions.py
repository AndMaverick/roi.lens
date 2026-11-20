"""Distributions for uncertain inputs. Each has ``ppf`` (the quantile function, which drives every sampler), ``cdf``, ``mean`` and
``var``, so a sample can be checked against the exact answer."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Protocol

_N = NormalDist()
Z90 = _N.inv_cdf(0.95)   # a 90 percent interval runs from the 5th to the 95th percentile, 1.6449 standard deviations either side


class Distribution(Protocol):
    def ppf(self, p: float) -> float: ...
    def cdf(self, x: float) -> float: ...
    def mean(self) -> float: ...
    def var(self) -> float: ...


# regularized incomplete beta function, for the PERT distribution -------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b), the CDF of a Beta(a, b) variable."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def betainc_inv(a: float, b: float, p: float) -> float:
    """The x with I_x(a, b) = p. Newton's method safeguarded by a shrinking bracket, so it converges quickly and, because the CDF is
    monotone, can always fall back to bisection and never fail to converge."""
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    lo, hi = 0.0, 1.0
    x = a / (a + b)
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    for _ in range(200):
        f = betainc(a, b, x) - p
        if f > 0:
            hi = x
        else:
            lo = x
        pdf = math.exp(log_norm + (a - 1) * math.log(x) + (b - 1) * math.log1p(-x)) if 0 < x < 1 else 0.0
        nxt = x - f / pdf if pdf > 0 else (lo + hi) / 2
        if not lo < nxt < hi:
            nxt = (lo + hi) / 2
        if abs(nxt - x) < 1e-15 or hi - lo < 1e-15:
            return nxt
        x = nxt
    return x


# distributions ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Fixed:
    value: float

    def ppf(self, p: float) -> float:
        return self.value

    def cdf(self, x: float) -> float:
        return 1.0 if x >= self.value else 0.0

    def mean(self) -> float:
        return self.value

    def var(self) -> float:
        return 0.0


@dataclass(frozen=True)
class Uniform:
    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.low <= self.high:
            raise ValueError(f"uniform needs low <= high, got {self.low} and {self.high}")

    def ppf(self, p: float) -> float:
        return self.low + p * (self.high - self.low)

    def cdf(self, x: float) -> float:
        if self.high == self.low:
            return 1.0 if x >= self.low else 0.0
        return min(1.0, max(0.0, (x - self.low) / (self.high - self.low)))

    def mean(self) -> float:
        return (self.low + self.high) / 2

    def var(self) -> float:
        return (self.high - self.low) ** 2 / 12


@dataclass(frozen=True)
class Triangular:
    low: float
    mode: float
    high: float

    def __post_init__(self) -> None:
        if not self.low <= self.mode <= self.high:
            raise ValueError(f"triangular needs low <= mode <= high, got {self.low}, {self.mode}, {self.high}")

    def ppf(self, p: float) -> float:
        lo, m, hi = self.low, self.mode, self.high
        if hi == lo:
            return lo
        split = (m - lo) / (hi - lo)
        if p < split:
            return lo + math.sqrt(p * (hi - lo) * (m - lo))
        return hi - math.sqrt((1 - p) * (hi - lo) * (hi - m))

    def cdf(self, x: float) -> float:
        lo, m, hi = self.low, self.mode, self.high
        if x <= lo:
            return 1.0 if hi == lo and x >= lo else 0.0
        if x >= hi:
            return 1.0
        if x <= m:
            return (x - lo) ** 2 / ((hi - lo) * (m - lo))
        return 1 - (hi - x) ** 2 / ((hi - lo) * (hi - m))

    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3

    def var(self) -> float:
        a, b, c = self.low, self.mode, self.high
        return (a * a + b * b + c * c - a * b - a * c - b * c) / 18


@dataclass(frozen=True)
class Pert:
    """The standard estimating distribution: a smooth Beta shaped by a low, a most likely value and a high, with less weight in the tails
    than a triangular. ``lam`` of 4 is the usual PERT; larger values concentrate more around the mode."""

    low: float
    mode: float
    high: float
    lam: float = 4.0

    def __post_init__(self) -> None:
        if not self.low <= self.mode <= self.high:
            raise ValueError(f"pert needs low <= mode <= high, got {self.low}, {self.mode}, {self.high}")
        if self.lam <= 0:
            raise ValueError("pert lam must be positive")

    @property
    def shape(self) -> tuple[float, float]:
        span = self.high - self.low
        return 1 + self.lam * (self.mode - self.low) / span, 1 + self.lam * (self.high - self.mode) / span

    def ppf(self, p: float) -> float:
        if self.high == self.low:
            return self.low
        a, b = self.shape
        return self.low + (self.high - self.low) * betainc_inv(a, b, p)

    def cdf(self, x: float) -> float:
        if self.high == self.low:
            return 1.0 if x >= self.low else 0.0
        a, b = self.shape
        return betainc(a, b, (x - self.low) / (self.high - self.low))

    def mean(self) -> float:
        return (self.low + self.lam * self.mode + self.high) / (self.lam + 2)

    def var(self) -> float:
        if self.high == self.low:
            return 0.0
        a, b = self.shape
        return (self.high - self.low) ** 2 * a * b / ((a + b) ** 2 * (a + b + 1))


@dataclass(frozen=True)
class Normal:
    """A normal, optionally cut off at ``lower`` and ``upper`` (the cut-off distribution is renormalised, not clipped)."""

    mu: float
    sigma: float
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("normal sigma must not be negative")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("normal lower must not exceed upper")
        if self.sigma > 0 and self._mass() <= 0:
            raise ValueError("the bounds leave no probability mass")

    def _bounds(self) -> tuple[float, float]:
        if self.sigma == 0:
            return 0.0, 1.0
        lo = 0.0 if self.lower is None else _N.cdf((self.lower - self.mu) / self.sigma)
        hi = 1.0 if self.upper is None else _N.cdf((self.upper - self.mu) / self.sigma)
        return lo, hi

    def _mass(self) -> float:
        lo, hi = self._bounds()
        return hi - lo

    def ppf(self, p: float) -> float:
        if self.sigma == 0:
            return self.mu
        lo, hi = self._bounds()
        u = lo + min(max(p, 1e-12), 1 - 1e-12) * (hi - lo)
        return self.mu + self.sigma * _N.inv_cdf(u)

    def cdf(self, x: float) -> float:
        if self.sigma == 0:
            return 1.0 if x >= self.mu else 0.0
        lo, hi = self._bounds()
        return min(1.0, max(0.0, (_N.cdf((x - self.mu) / self.sigma) - lo) / (hi - lo)))

    def mean(self) -> float:
        if self.sigma == 0:
            return self.mu
        lo, hi = self._bounds()
        a = -math.inf if self.lower is None else (self.lower - self.mu) / self.sigma
        b = math.inf if self.upper is None else (self.upper - self.mu) / self.sigma
        return self.mu + self.sigma * (_phi(a) - _phi(b)) / (hi - lo)

    def var(self) -> float:
        if self.sigma == 0:
            return 0.0
        lo, hi = self._bounds()
        z = hi - lo
        a = -math.inf if self.lower is None else (self.lower - self.mu) / self.sigma
        b = math.inf if self.upper is None else (self.upper - self.mu) / self.sigma
        ta = 0.0 if a == -math.inf else a * _phi(a)
        tb = 0.0 if b == math.inf else b * _phi(b)
        shift = (_phi(a) - _phi(b)) / z
        return self.sigma ** 2 * (1 + (ta - tb) / z - shift ** 2)


def _phi(x: float) -> float:
    return 0.0 if math.isinf(x) else math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


@dataclass(frozen=True)
class Lognormal:
    """Positive and right-skewed: costs and durations that overrun more than they underrun. ``mu`` and ``sigma`` are of the log."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("lognormal sigma must not be negative")

    def ppf(self, p: float) -> float:
        return math.exp(self.mu) if self.sigma == 0 else math.exp(self.mu + self.sigma * _N.inv_cdf(min(max(p, 1e-12), 1 - 1e-12)))

    def cdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        return (1.0 if x >= math.exp(self.mu) else 0.0) if self.sigma == 0 else _N.cdf((math.log(x) - self.mu) / self.sigma)

    def mean(self) -> float:
        return math.exp(self.mu + self.sigma ** 2 / 2)

    def var(self) -> float:
        return (math.exp(self.sigma ** 2) - 1) * math.exp(2 * self.mu + self.sigma ** 2)


@dataclass(frozen=True)
class Empirical:
    """Your own observations, each equally likely. Use it for measured pilot data."""

    values: tuple[float, ...]

    def __init__(self, values: Sequence[float]) -> None:
        if not values:
            raise ValueError("empirical needs at least one value")
        object.__setattr__(self, "values", tuple(sorted(float(v) for v in values)))

    def ppf(self, p: float) -> float:
        n = len(self.values)
        return self.values[min(n - 1, int(p * n))]

    def cdf(self, x: float) -> float:
        import bisect
        return bisect.bisect_right(self.values, x) / len(self.values)

    def mean(self) -> float:
        return sum(self.values) / len(self.values)

    def var(self) -> float:
        m = self.mean()
        return sum((v - m) ** 2 for v in self.values) / len(self.values)


def from_ci90(low: float, high: float, *, kind: str = "normal", lower: float | None = None, upper: float | None = None) -> Distribution:
    """A distribution from a 90 percent confidence interval: you are 90 percent sure the true value lies between ``low`` and ``high``
    (5 percent chance below, 5 percent above). This is the form that can be calibrated (see ``calibration``). ``kind`` is ``normal`` or
    ``lognormal`` (which needs low > 0 and suits quantities that cannot be negative and skew high). ``lower`` and ``upper`` cut a normal
    off, for example 0 and 1 for a share."""
    if not low < high:
        raise ValueError(f"a 90 percent interval needs low < high, got {low} and {high}")
    if kind == "normal":
        return Normal((low + high) / 2, (high - low) / (2 * Z90), lower, upper)
    if kind == "lognormal":
        if low <= 0:
            raise ValueError("a lognormal interval needs low > 0")
        return Lognormal((math.log(low) + math.log(high)) / 2, (math.log(high) - math.log(low)) / (2 * Z90))
    raise ValueError(f"kind must be normal or lognormal, got {kind!r}")


def from_dict(spec: Any, where: str = "input") -> Distribution:
    """Build a distribution from JSON: a bare number is fixed; ``{"dist": "pert", "low": ..., "mode": ..., "high": ...}`` and so on."""
    if isinstance(spec, bool):
        raise ValueError(f"{where}: expected a number or a distribution object")
    if isinstance(spec, (int, float)):
        return Fixed(float(spec))
    if not isinstance(spec, dict) or "dist" not in spec:
        raise ValueError(f"{where}: expected a number or an object with a 'dist' key")
    kind = spec["dist"]
    shapes = {"fixed": {"value"}, "uniform": {"low", "high"}, "triangular": {"low", "mode", "high"}, "pert": {"low", "mode", "high", "lam"},
              "normal": {"mu", "sigma", "lower", "upper"}, "lognormal": {"mu", "sigma"}, "ci90": {"low", "high", "kind", "lower", "upper"},
              "empirical": {"values"}}
    if kind not in shapes:
        raise ValueError(f"{where}: dist must be one of {', '.join(sorted(shapes))}, got {kind!r}")
    extra = sorted(set(spec) - shapes[kind] - {"dist"})
    if extra:
        raise ValueError(f"{where}: unknown key {extra[0]!r} for {kind} (allowed: {', '.join(sorted(shapes[kind]))})")
    args = {k: v for k, v in spec.items() if k != "dist"}
    try:
        if kind == "fixed":
            return Fixed(float(args["value"]))
        if kind == "ci90":
            return from_ci90(float(args["low"]), float(args["high"]), kind=args.get("kind", "normal"), lower=args.get("lower"), upper=args.get("upper"))
        cls: Any = {"uniform": Uniform, "triangular": Triangular, "pert": Pert, "normal": Normal, "lognormal": Lognormal,
                    "empirical": Empirical}[kind]
        result: Distribution = cls(**args)
        return result
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{where}: {kind} is missing or has a bad value ({exc})") from exc
