"""Financial model implementations.

Modules:
    mpt: Modern Portfolio Theory — max-Sharpe portfolio optimisation.
    risk: Historical Value at Risk and Conditional Value at Risk.
    single_index_model: Sharpe's Single Index Model (alpha, beta, residual risk).
    utility: Mean-variance utility for an investor with risk aversion ``A``.
"""

from .mpt import portfolio_metrics
from .risk import get_cvar, get_risk_metrics, get_var
from .single_index_model import AssetFit, SingleIndexModel
from .utility import get_utility, max_utility

__all__ = [
    "AssetFit",
    "SingleIndexModel",
    "get_cvar",
    "get_risk_metrics",
    "get_utility",
    "get_var",
    "max_utility",
    "portfolio_metrics",
]
