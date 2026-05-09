"""Historical Value at Risk and Conditional Value at Risk.

Both :func:`get_var` and :func:`get_cvar` use the empirical (historical)
distribution of returns rather than a parametric assumption. Returns are
flattened across all assets in the input DataFrame before the percentile
is computed.
"""

import numpy as np
import pandas as pd

from portfolio import config


def _flat_returns(prices: pd.DataFrame) -> np.ndarray:
    """Compute simple returns and return them as a flat array of finite values.

    Args:
        prices: A DataFrame of prices indexed by date.

    Returns:
        A 1-D float array containing every finite return across all assets.

    Raises:
        ValueError: If ``prices`` is empty, has too few observations to
            compute returns, or yields no finite returns.
    """
    if prices is None or prices.empty:
        raise ValueError("prices is empty")
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    if returns.empty:
        raise ValueError("not enough observations to compute returns")
    flat = np.asarray(returns.to_numpy(), dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        raise ValueError("no finite returns available")
    return flat


def get_var(prices: pd.DataFrame, confidence: float = config.confidence_level_95) -> float:
    """Historical Value at Risk at the given confidence level.

    Args:
        prices: A DataFrame of prices indexed by date.
        confidence: Confidence level in the open interval ``(0, 1)``. The VaR
            is the ``(1 - confidence)`` percentile of the empirical returns.

    Returns:
        The historical VaR as a (typically negative) float.

    Raises:
        ValueError: If ``confidence`` is not in ``(0, 1)``, or if ``prices``
            does not yield any finite returns.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    flat = _flat_returns(prices)
    return float(np.percentile(flat, (1.0 - confidence) * 100.0))


def get_cvar(prices: pd.DataFrame, confidence: float = config.confidence_level_95) -> float:
    """Historical Conditional Value at Risk (a.k.a. Expected Shortfall).

    CVaR is the mean of the returns that fall at or below the VaR threshold.

    Args:
        prices: A DataFrame of prices indexed by date.
        confidence: Confidence level in the open interval ``(0, 1)``.

    Returns:
        The historical CVaR as a (typically negative) float. Falls back to
        the VaR value if the tail is empty.

    Raises:
        ValueError: If ``confidence`` is not in ``(0, 1)``, or if ``prices``
            does not yield any finite returns.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    flat = _flat_returns(prices)
    var = float(np.percentile(flat, (1.0 - confidence) * 100.0))
    tail = flat[flat <= var]
    if tail.size == 0:
        return var
    return float(np.mean(tail))
