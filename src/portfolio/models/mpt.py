"""Modern Portfolio Theory: max-Sharpe portfolio optimisation via pypfopt."""

from typing import Any

import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models


def portfolio_metrics(prices: pd.DataFrame, risk_free_rate: float) -> dict[str, Any]:
    """Return expected return, volatility, Sharpe ratio, and weights of the max-Sharpe portfolio."""
    if prices is None or prices.empty:
        raise ValueError("prices must contain at least one asset column")
    if prices.shape[0] < 2:
        raise ValueError("prices must contain at least two observations")

    expected_returns_vec = expected_returns.mean_historical_return(prices)
    covariance_matrix = risk_models.sample_cov(prices)

    frontier = EfficientFrontier(expected_returns_vec, covariance_matrix)
    frontier.max_sharpe(risk_free_rate=risk_free_rate)
    weights = frontier.clean_weights()

    annual_return, annual_volatility, sharpe_ratio = frontier.portfolio_performance(
        risk_free_rate=risk_free_rate, verbose=False
    )

    return {
        "expected_annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "weights": weights,
    }
