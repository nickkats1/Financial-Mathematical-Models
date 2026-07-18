"""Tests for :mod:`portfolio.models.risk`."""

import math

import numpy as np
import pandas as pd
import pytest

from portfolio.models.risk import get_cvar, get_var


class TestRisk:
    """Tests for Value at Risk and Conditional Value at Risk."""

    def test_var_is_finite_float(self, fake_prices):
        var = get_var(fake_prices)
        assert isinstance(var, float)
        assert math.isfinite(var)

    def test_cvar_is_at_most_var(self, fake_prices):
        """CVaR is the mean of the left tail, so it must be <= VaR."""
        var = get_var(fake_prices)
        cvar = get_cvar(fake_prices)
        assert cvar <= var

    def test_var_matches_empirical_percentile(self):
        """VaR must equal the (1 - c) percentile of the empirical returns."""
        prices = pd.DataFrame({"A": [100.0, 110.0, 99.0, 103.95, 100.0]})
        returns = prices.pct_change(fill_method=None).dropna().to_numpy().ravel()
        expected = float(np.percentile(returns, 5.0))
        assert get_var(prices, confidence=0.95) == pytest.approx(expected)

    def test_cvar_is_mean_of_tail(self):
        prices = pd.DataFrame({"A": [100.0, 110.0, 99.0, 103.95, 100.0]})
        var = get_var(prices, confidence=0.95)
        returns = prices.pct_change(fill_method=None).dropna().to_numpy().ravel()
        tail = returns[returns <= var]
        assert get_cvar(prices, confidence=0.95) == pytest.approx(float(tail.mean()))

    def test_pools_returns_across_assets(self):
        """A wider frame contributes every asset's returns to the distribution."""
        prices = pd.DataFrame(
            {"A": [100.0, 90.0, 99.0], "B": [50.0, 60.0, 55.0]}
        )
        # -0.10 (A) is the worst pooled return, so the 5% VaR sits near it.
        assert get_var(prices, confidence=0.95) < 0.0

    def test_invalid_confidence_raises(self, fake_prices):
        for bad in (0.0, 1.0, -0.5, 1.5):
            with pytest.raises(ValueError, match="confidence must be in"):
                get_var(fake_prices, confidence=bad)
            with pytest.raises(ValueError, match="confidence must be in"):
                get_cvar(fake_prices, confidence=bad)

    def test_empty_prices_raises(self):
        with pytest.raises(ValueError, match="prices is empty"):
            get_var(pd.DataFrame())
        with pytest.raises(ValueError, match="prices is empty"):
            get_cvar(pd.DataFrame())

    def test_none_prices_raises(self):
        with pytest.raises(ValueError, match="prices is empty"):
            get_var(None)

    def test_too_few_observations_raises(self):
        """A single price row yields no returns after dropping the NaN row."""
        one_row = pd.DataFrame({"A": [100.0]})
        with pytest.raises(ValueError, match="not enough observations"):
            get_var(one_row)

    def test_no_finite_returns_raises(self):
        """A zero price produces an infinite return that is filtered out."""
        prices = pd.DataFrame({"A": [0.0, 5.0]})
        with pytest.raises(ValueError, match="no finite returns"):
            get_var(prices)
