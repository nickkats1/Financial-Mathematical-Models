"""Financial model implementations.

Modules:
    mpt: Modern Portfolio Theory — max-Sharpe portfolio optimisation.
    risk: Historical Value at Risk and Conditional Value at Risk.
    single_index_model: Sharpe's Single Index Model (alpha, beta, residual risk).
    utility: Mean-variance utility for an investor with risk aversion ``A``.
"""

from portfolio.models.mpt import portfolio_metrics
from portfolio.models.risk import get_cvar, get_var
from portfolio.models.single_index_model import SingleIndexModel
from portfolio.models.utility import get_utility, max_utility

__all__ = [
    "SingleIndexModel",
    "get_cvar",
    "get_utility",
    "get_var",
    "max_utility",
    "portfolio_metrics",
]
