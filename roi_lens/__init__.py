"""roi.lens: decision-grade ROI for AI projects. Honest uncertainty, tested against closed-form answers, reproducible to the bit."""
from __future__ import annotations

from typing import Any, overload

from ._version import __version__
from .analysis import breakeven, evpi, evppi, pilot_value, rank_drivers, tornado, value_of_information
from .calibration import Answer, calibrate, scale_interval
from .distributions import Empirical, Fixed, Lognormal, Normal, Pert, Triangular, Uniform, from_ci90, from_dict
from .engine import Outcome, Run, evaluate
from .engine import simulate as simulate_model
from .expr import Expr, ExprError, compile_expr
from .model import Event, Model
from .portfolio import Case, plan
from .report import analyze, render_markdown, render_text
from .scenario import Scenario, load, parse
from .simple import UseCase, point_estimate, rank, triangular
from .simple import simulate as simulate_simple


@overload
def simulate(subject: UseCase, months: int = 12, trials: int = 5000, seed: int = 7) -> dict[str, float]: ...
@overload
def simulate(subject: Model, trials: int = 5000, seed: int = 7, method: str = "lhs") -> Run: ...
def simulate(subject: UseCase | Model, *args: Any, **kwargs: Any) -> dict[str, float] | Run:
    """One name for both generations. A ``UseCase`` runs the original 0.1 simulation (unchanged); a ``Model`` runs the full engine."""
    if isinstance(subject, UseCase):
        return simulate_simple(subject, *args, **kwargs)
    return simulate_model(subject, *args, **kwargs)

__all__ = [
    "__version__",
    "Answer", "Case", "Empirical", "Event", "Expr", "ExprError", "Fixed", "Lognormal", "Model", "Normal", "Outcome", "Pert", "Run",
    "Scenario", "Triangular", "Uniform", "UseCase", "analyze", "breakeven", "calibrate", "compile_expr", "evaluate", "evpi", "evppi",
    "from_ci90", "from_dict", "load", "parse", "pilot_value", "plan", "point_estimate", "rank", "rank_drivers", "render_markdown",
    "render_text", "scale_interval", "simulate", "simulate_simple", "tornado", "triangular", "value_of_information",
]
