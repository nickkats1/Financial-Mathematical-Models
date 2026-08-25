"""Mean-variance utility: U = E[r] - c * A * sigma^2 for risk aversion A."""

import pandas as pd
from pypfopt import expected_returns

from ..config import DEFAULT_CONFIG, TRADING_DAYS


def _per_asset_variance(prices: pd.DataFrame) -> pd.Series:
    # Only the diagonal is needed, so this avoids sample_cov's full n x n matrix
    # and its positive-semidefinite repair.
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    return returns.var(ddof=1) * TRADING_DAYS


def get_utility(
    prices: pd.DataFrame,
    risk_aversion: float,
    scaling_factor: float = DEFAULT_CONFIG.utility_scaling_factor,
) -> pd.Series:
    """Per-asset mean-variance utility."""
    er = expected_returns.mean_historical_return(prices)
    var = _per_asset_variance(prices)
    return er - scaling_factor * risk_aversion * var


def max_utility(
    prices: pd.DataFrame,
    risk_aversion: float,
    risk_free_rate: float,
    scaling_factor: float = DEFAULT_CONFIG.utility_scaling_factor,
) -> pd.Series:
    """Per-asset utility at the stationary risky weight y* = (E[r] - rf) / (A * sigma^2).

    Assets with zero variance are excluded from the result — the optimal weight
    is undefined (division by zero) for constant-price series.

    For A > 0 utility is concave in y, so y* is the maximum. For A < 0 (a
    risk-seeking investor) it is convex and y* is the minimum instead — the
    supremum is unbounded. A = 0 has no stationary point at all.
    """
    if risk_aversion == 0:
        raise ValueError("risk_aversion must be non-zero")
    er = expected_returns.mean_historical_return(prices)
    var = _per_asset_variance(prices)
    var_safe = var.where(var > 0)
    y_star = (er - risk_free_rate) / (risk_aversion * var_safe)
    result = (
        risk_free_rate
        + y_star * (er - risk_free_rate)
        - scaling_factor * risk_aversion * y_star ** 2 * var_safe
    )
    return result.dropna()
