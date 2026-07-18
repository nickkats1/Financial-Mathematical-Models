"""Historical (non-parametric) Value at Risk and Conditional Value at Risk."""

import numpy as np
import pandas as pd

from .. import config


def _flat_returns(prices: pd.DataFrame) -> np.ndarray:
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
    """VaR: the (1 - confidence) percentile of the empirical returns."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    flat = _flat_returns(prices)
    return float(np.percentile(flat, (1.0 - confidence) * 100.0))


def get_cvar(prices: pd.DataFrame, confidence: float = config.confidence_level_95) -> float:
    """CVaR (expected shortfall): mean of returns at or below the VaR threshold."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    flat = _flat_returns(prices)
    var = float(np.percentile(flat, (1.0 - confidence) * 100.0))
    tail = flat[flat <= var]
    if tail.size == 0:  # pragma: no cover - percentile always leaves >=1 point in the tail
        return var
    return float(np.mean(tail))
