"""Tests for :mod:`portfolio.models.mpt`."""

import math

import numpy as np
import pandas as pd
import pytest

from portfolio.models.mpt import portfolio_metrics


@pytest.fixture
def synthetic_prices():
    """Three uncorrelated random-walk price series with positive drift."""
    rng = np.random.default_rng(seed=0)
    dates = pd.bdate_range("2023-01-01", periods=252)
    returns = rng.normal(loc=0.0008, scale=0.01, size=(252, 3))
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=["A", "B", "C"])


class TestPortfolioMetrics:
    def test_returns_expected_keys(self, synthetic_prices):
        result = portfolio_metrics(synthetic_prices, risk_free_rate=0.02)
        assert set(result.keys()) == {
            "expected_annual_return",
            "annual_volatility",
            "sharpe_ratio",
            "weights",
        }

    def test_metrics_are_finite(self, synthetic_prices):
        result = portfolio_metrics(synthetic_prices, risk_free_rate=0.02)
        assert math.isfinite(result["expected_annual_return"])
        assert math.isfinite(result["annual_volatility"])
        assert math.isfinite(result["sharpe_ratio"])
        assert result["annual_volatility"] > 0.0

    def test_weights_sum_to_one(self, synthetic_prices):
        result = portfolio_metrics(synthetic_prices, risk_free_rate=0.02)
        weights = result["weights"]
        assert isinstance(weights, dict)
        assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6)

    def test_weights_are_non_negative(self, synthetic_prices):
        """The default EfficientFrontier solver imposes long-only constraints."""
        result = portfolio_metrics(synthetic_prices, risk_free_rate=0.02)
        for ticker, weight in result["weights"].items():
            assert weight >= 0.0, f"{ticker} has negative weight {weight}"
