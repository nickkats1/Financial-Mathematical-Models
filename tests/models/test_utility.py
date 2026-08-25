"""Tests for :mod:`portfolio.models.utility`."""

import numpy as np
import pandas as pd
import pytest

from portfolio.models.utility import _per_asset_variance, get_utility, max_utility


@pytest.fixture
def synthetic_prices():
    """Three random-walk price series, used to make the variance non-trivial."""
    rng = np.random.default_rng(seed=42)
    dates = pd.bdate_range("2023-01-01", periods=252)
    returns = rng.normal(loc=0.0005, scale=0.02, size=(252, 3))
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=["A", "B", "C"])


class TestPerAssetVariance:
    def test_matches_the_sample_covariance_diagonal(self, synthetic_prices):
        """Computed directly, but must still equal pypfopt's annualised diagonal."""
        from pypfopt import risk_models

        expected = np.diag(risk_models.sample_cov(synthetic_prices).to_numpy())
        assert _per_asset_variance(synthetic_prices).to_numpy() == pytest.approx(
            expected
        )

    def test_constant_prices_have_exactly_zero_variance(self, synthetic_prices):
        """max_utility relies on an exact zero to drop undefined assets."""
        prices = synthetic_prices.assign(FLAT=100.0)
        assert _per_asset_variance(prices)["FLAT"] == 0.0


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

    def test_rejects_zero_risk_aversion(self, synthetic_prices):
        """y* divides by A, so only A = 0 is mathematically out of bounds."""
        with pytest.raises(ValueError, match="risk_aversion must be non-zero"):
            max_utility(synthetic_prices, risk_aversion=0.0, risk_free_rate=0.02)

    def test_accepts_negative_risk_aversion(self, synthetic_prices):
        """A risk-seeking investor still has a stationary point (a minimum)."""
        utility = max_utility(
            synthetic_prices, risk_aversion=-2.0, risk_free_rate=0.02
        )
        assert isinstance(utility, pd.Series)
        assert set(utility.index) == set(synthetic_prices.columns)
        assert (utility <= 0.02 + 1e-12).all()

    def test_max_utility_at_least_risk_free_rate(self, synthetic_prices):
        """The investor can always set y* = 0 and earn rf, so U* >= rf."""
        rf = 0.02
        utility = max_utility(synthetic_prices, risk_aversion=3.0, risk_free_rate=rf)
        assert (utility >= rf - 1e-12).all()

    def test_constant_price_asset_is_excluded(self, synthetic_prices):
        """A flat series has zero variance, so y* is undefined and it is dropped."""
        prices = synthetic_prices.assign(FLAT=100.0)
        utility = max_utility(prices, risk_aversion=3.0, risk_free_rate=0.02)
        assert "FLAT" not in utility.index
        assert set(utility.index) == set(synthetic_prices.columns)
