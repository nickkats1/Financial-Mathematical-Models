"""Historical (non-parametric) Value at Risk and Conditional Value at Risk."""

import numpy as np
import pandas as pd

from ..config import DEFAULT_CONFIG


def _flat_returns(prices: pd.DataFrame) -> np.ndarray:
    if prices is None or prices.empty:
        raise ValueError("prices is empty")
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    if returns.empty:
        raise ValueError("not enough observations to compute returns")
    flat = returns.to_numpy(dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        raise ValueError("no finite returns available")
    return flat


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")


def _var_cvar(sorted_flat: np.ndarray, confidence: float) -> tuple[float, float]:
    var = float(np.percentile(sorted_flat, (1.0 - confidence) * 100.0))
    # np.percentile always returns a value within [min, max], so the tail slice can
    # never be empty. Sorting first lets it be a view rather than a boolean mask.
    cutoff = int(np.searchsorted(sorted_flat, var, side="right"))
    return var, float(sorted_flat[:cutoff].mean())


def get_var(
    prices: pd.DataFrame,
    confidence: float = DEFAULT_CONFIG.default_confidence,
) -> float:
    """VaR: the (1 - confidence) percentile of the empirical returns."""
    _check_confidence(confidence)
    return _var_cvar(np.sort(_flat_returns(prices)), confidence)[0]


def get_cvar(
    prices: pd.DataFrame,
    confidence: float = DEFAULT_CONFIG.default_confidence,
) -> float:
    """CVaR (expected shortfall): mean of returns at or below the VaR threshold."""
    _check_confidence(confidence)
    return _var_cvar(np.sort(_flat_returns(prices)), confidence)[1]


def get_risk_metrics(
    prices: pd.DataFrame,
    confidences: tuple[float, ...] = DEFAULT_CONFIG.confidence_levels,
) -> dict[float, tuple[float, float]]:
    """Return {confidence: (var, cvar)} for all levels in a single pass over returns.

    Prefer this over calling get_var/get_cvar separately when multiple confidence
    levels are needed for the same price DataFrame.
    """
    for c in confidences:
        _check_confidence(c)
    flat = np.sort(_flat_returns(prices))
    return {c: _var_cvar(flat, c) for c in confidences}
