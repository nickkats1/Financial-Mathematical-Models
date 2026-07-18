"""Mean-variance utility: U = E[r] - c * A * sigma^2 for risk aversion A."""

import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models


def _per_asset_variance(prices: pd.DataFrame) -> pd.Series:
    cov = risk_models.sample_cov(prices)
    return pd.Series(np.diag(cov.to_numpy()), index=cov.index)


def get_utility(
    prices: pd.DataFrame,
    risk_aversion: float,
    scaling_factor: float = 0.5,
) -> pd.Series:
    """Per-asset mean-variance utility."""
    er = expected_returns.mean_historical_return(prices)
    var = _per_asset_variance(prices)
    return er - scaling_factor * risk_aversion * var


def max_utility(
    prices: pd.DataFrame,
    risk_aversion: float,
    risk_free_rate: float,
    scaling_factor: float = 0.5,
) -> pd.Series:
    """Per-asset maximised utility with optimal risky weight y* = (E[r] - rf) / (A * sigma^2)."""
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    er = expected_returns.mean_historical_return(prices)
    var = _per_asset_variance(prices)
    y_star = (er - risk_free_rate) / (risk_aversion * var)
    return (
        risk_free_rate
        + y_star * (er - risk_free_rate)
        - scaling_factor * risk_aversion * y_star ** 2 * var
    )
