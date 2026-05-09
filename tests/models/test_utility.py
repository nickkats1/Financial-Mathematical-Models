"""Tests for :mod:`portfolio.models.utility`."""

import numpy as np
import pandas as pd
import pytest

from portfolio.models.utility import get_utility, max_utility


@pytest.fixture
def synthetic_prices():
    """Three random-walk price series, used to make the variance non-trivial."""
    rng = np.random.default_rng(seed=42)
    dates = pd.bdate_range("2023-01-01", periods=252)
    returns = rng.normal(loc=0.0005, scale=0.02, size=(252, 3))
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=["A", "B", "C"])


class TestGetUtility:
    def test_returns_one_value_per_asset(self, synthetic_prices):
        utility = get_utility(synthetic_prices, risk_aversion=3.0)
        assert isinstance(utility, pd.Series)
        assert set(utility.index) == set(synthetic_prices.columns)

    def test_higher_risk_aversion_lowers_utility(self, synthetic_prices):
        """For a positive-variance asset, raising A must not raise utility."""
        low_aversion = get_utility(synthetic_prices, risk_aversion=1.0)
        high_aversion = get_utility(synthetic_prices, risk_aversion=5.0)
        assert (high_aversion <= low_aversion).all()

    def test_zero_risk_aversion_equals_expected_return(self, synthetic_prices):
        """With A = 0 the utility collapses to the expected return."""
        from pypfopt import expected_returns

        utility = get_utility(synthetic_prices, risk_aversion=0.0)
        er = expected_returns.mean_historical_return(synthetic_prices)
        pd.testing.assert_series_equal(utility, er, check_names=False)


class TestMaxUtility:
    def test_returns_one_value_per_asset(self, synthetic_prices):
        utility = max_utility(
            synthetic_prices, risk_aversion=3.0, risk_free_rate=0.02
        )
        assert isinstance(utility, pd.Series)
        assert set(utility.index) == set(synthetic_prices.columns)

    def test_rejects_non_positive_risk_aversion(self, synthetic_prices):
        with pytest.raises(ValueError, match="risk_aversion must be positive"):
            max_utility(synthetic_prices, risk_aversion=0.0, risk_free_rate=0.02)
        with pytest.raises(ValueError, match="risk_aversion must be positive"):
            max_utility(synthetic_prices, risk_aversion=-1.0, risk_free_rate=0.02)

    def test_max_utility_at_least_risk_free_rate(self, synthetic_prices):
        """The investor can always set y* = 0 and earn rf, so U* >= rf."""
        rf = 0.02
        utility = max_utility(synthetic_prices, risk_aversion=3.0, risk_free_rate=rf)
        assert (utility >= rf - 1e-12).all()
