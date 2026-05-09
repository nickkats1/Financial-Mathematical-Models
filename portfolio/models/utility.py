"""Mean-variance utility theory.

Provides utility scores for an investor characterised by a constant
absolute risk-aversion coefficient ``A``. The mean-variance utility of an
asset with expected return ``E[r]`` and variance ``sigma^2`` is::

    U = E[r] - c * A * sigma^2

where ``c`` is a scaling factor (conventionally ``1/2``).

A risk-averse investor has ``A > 0``; a risk-neutral investor has
``A == 0``; a risk-loving investor has ``A < 0``.
"""

import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models


def _per_asset_variance(prices: pd.DataFrame) -> pd.Series:
    """Return the diagonal of the sample covariance matrix as a Series.

    Args:
        prices: A DataFrame of prices indexed by date.

    Returns:
        A Series of per-asset variances, indexed by ticker.
    """
    cov = risk_models.sample_cov(prices)
    return pd.Series(np.diag(cov.to_numpy()), index=cov.index)


def get_utility(
    prices: pd.DataFrame,
    risk_aversion: float,
    scaling_factor: float = 0.5,
) -> pd.Series:
    """Per-asset mean-variance utility ``U = E[r] - c * A * Var[r]``.

    Args:
        prices: A DataFrame of prices indexed by date.
        risk_aversion: The investor's risk aversion ``A``. Positive for
            risk-averse, zero for risk-neutral, negative for risk-loving.
        scaling_factor: The constant ``c`` in front of ``A * Var[r]``.
            Defaults to ``0.5``.

    Returns:
        A Series of utility values, one per asset.
    """
    er = expected_returns.mean_historical_return(prices)
    var = _per_asset_variance(prices)
    return er - scaling_factor * risk_aversion * var


def max_utility(
    prices: pd.DataFrame,
    risk_aversion: float,
    risk_free_rate: float,
    scaling_factor: float = 0.5,
) -> pd.Series:
    """Per-asset maximised utility for an investor allocating between rf and one risky asset.

    The optimal risky-asset weight is::

        y* = (E[r] - rf) / (A * sigma^2)

    and the resulting utility is::

        U* = rf + y* * (E[r] - rf) - c * A * y*^2 * sigma^2

    Args:
        prices: A DataFrame of prices indexed by date.
        risk_aversion: The investor's risk aversion ``A``. Must be positive.
        risk_free_rate: The risk-free rate, expressed in the same period
            convention as the returns implied by ``prices``.
        scaling_factor: The constant ``c`` in front of ``A * Var[r]``.
            Defaults to ``0.5``.

    Returns:
        A Series of maximized utilities, one per asset.

    Raises:
        ValueError: If ``risk_aversion`` is not strictly positive.
    """
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
