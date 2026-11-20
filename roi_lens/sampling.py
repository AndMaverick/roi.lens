"""Sampling: simple random or Latin hypercube uniforms, and rank correlation between inputs.

Latin hypercube sampling cuts [0, 1) into n equal strata and takes one point from each, so every region of every input is covered even with
a few thousand trials. Correlation is imposed by reordering (not by changing values): each input keeps exactly its sampled values, and a
Gaussian copula decides which trial gets which value. That keeps the stratification intact and makes the correlation a rank (Spearman)
correlation, which is what an estimator can actually defend."""
from __future__ import annotations

import math
import random
from collections.abc import Sequence

Matrix = list[list[float]]


def uniforms(n: int, d: int, rng: random.Random, method: str = "lhs") -> Matrix:
    """``d`` columns of ``n`` uniforms in (0, 1). ``method`` is ``lhs`` or ``random``."""
    if n < 1 or d < 0:
        raise ValueError("need n >= 1 and d >= 0")
    if method not in ("lhs", "random"):
        raise ValueError(f"method must be lhs or random, got {method!r}")
    cols: Matrix = []
    for _ in range(d):
        if method == "random":
            cols.append([min(max(rng.random(), 1e-12), 1 - 1e-12) for _ in range(n)])
        else:
            perm = list(range(n))
            rng.shuffle(perm)
            cols.append([(perm[i] + rng.random()) / n for i in range(n)])
    return cols


def cholesky(m: Sequence[Sequence[float]]) -> Matrix:
    """Lower Cholesky factor. Raises ``ValueError`` when the matrix is not positive semi-definite (an impossible set of correlations)."""
    n = len(m)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(low[i][k] * low[j][k] for k in range(j))
            if i == j:
                v = m[i][i] - s
                if v < -1e-9:
                    raise ValueError("the correlation matrix is not positive semi-definite (these correlations cannot all hold together)")
                low[i][j] = math.sqrt(max(v, 0.0))
            else:
                low[i][j] = (m[i][j] - s) / low[j][j] if low[j][j] > 1e-12 else 0.0
    return low


def pearson_for_spearman(rho_s: float) -> float:
    """The Gaussian-copula Pearson correlation that gives Spearman correlation ``rho_s``: 2 sin(pi rho_s / 6)."""
    return 2 * math.sin(math.pi * rho_s / 6)


def spearman_for_pearson(rho: float) -> float:
    """The inverse: the Spearman correlation of a Gaussian copula with Pearson ``rho``: (6 / pi) asin(rho / 2)."""
    return 6 / math.pi * math.asin(rho / 2)


def correlation_matrix(d: int, pairs: Sequence[tuple[int, int, float]]) -> Matrix:
    m = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
    for i, j, r in pairs:
        if i == j:
            raise ValueError("a correlation needs two different inputs")
        if not -1 < r < 1:
            raise ValueError(f"a rank correlation must be strictly between -1 and 1, got {r}")
        m[i][j] = m[j][i] = r
    return m


def ranks(values: Sequence[float]) -> list[float]:
    """Average ranks (1-based), ties share the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = mean_rank
        i = j + 1
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx, ry = ranks(x), ranks(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True)) / math.sqrt(sxx * syy)


def correlate(cols: Matrix, spearman_matrix: Sequence[Sequence[float]], rng: random.Random) -> Matrix:
    """Reorder each column so the columns have (approximately) the requested Spearman correlations. Values are only permuted, so every
    column keeps exactly the same set of values (and any Latin hypercube stratification)."""
    d = len(cols)
    n = len(cols[0]) if cols else 0
    target = [[1.0 if i == j else pearson_for_spearman(spearman_matrix[i][j]) for j in range(d)] for i in range(d)]
    low = cholesky(target)
    g = [[rng.gauss(0.0, 1.0) for _ in range(n)] for _ in range(d)]
    z = [[sum(low[j][k] * g[k][i] for k in range(j + 1)) for i in range(n)] for j in range(d)]
    out: Matrix = []
    for j in range(d):
        rk = ranks(z[j])
        ordered = sorted(cols[j])
        out.append([ordered[int(r) - 1] for r in rk])
    return out
