"""Modern Portfolio Theory: max-Sharpe portfolio optimisation.

Wraps :mod:`pypfopt` to compute the tangency (max-Sharpe) portfolio for
a given set of assets and a risk-free rate.
"""

from typing import Any, Dict

import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models


def portfolio_metrics(prices: pd.DataFrame, risk_free_rate: float) -> Dict[str, Any]:
    """Compute the max-Sharpe portfolio for the given prices.

    The expected returns are estimated from the historical mean and the
    covariance matrix from the sample covariance of returns. The
    :class:`pypfopt.EfficientFrontier` solver then maximises the Sharpe
    ratio subject to long-only, fully-invested constraints.

    Args:
        prices: A DataFrame of price levels indexed by date, with one
            column per asset.
        risk_free_rate: The annualised risk-free rate, expressed as a
            decimal (e.g. ``0.04`` for 4%).

    Returns:
        A dictionary with the following keys:

        - ``expected_annual_return`` (float): annualised expected return.
        - ``annual_volatility`` (float): annualised standard deviation.
        - ``sharpe_ratio`` (float): the maximised Sharpe ratio.
        - ``weights`` (Dict[str, float]): cleaned portfolio weights.
    """
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
