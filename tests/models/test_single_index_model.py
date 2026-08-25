"""Tests for :class:`portfolio.models.single_index_model.SingleIndexModel`."""

import numpy as np
import pandas as pd
import pytest

from portfolio.config import TRADING_DAYS
from portfolio.models.single_index_model import SingleIndexModel


@pytest.fixture
def returns():
    """Synthetic daily returns for three assets plus a market proxy.

    Each asset is generated as ``alpha + beta * market + noise`` so the
    fitted betas are meaningful (not degenerate) and the risk
    decomposition has a non-trivial systematic component.
    """
    rng = np.random.default_rng(seed=7)
    n = 120
    market = rng.normal(0.0004, 0.009, n)
    dates = pd.bdate_range("2023-01-01", periods=n)
    data = {
        "AAPL": 0.0002 + 1.1 * market + rng.normal(0, 0.006, n),
        "MSFT": 0.0001 + 0.9 * market + rng.normal(0, 0.005, n),
        "KO": 0.0003 + 0.4 * market + rng.normal(0, 0.004, n),
        "^GSPC": market,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def model():
    return SingleIndexModel()


class TestGetModels:
    def test_returns_dict_of_fitted_models(self, model, returns):
        fitted = model.get_models(["AAPL", "MSFT"], "^GSPC", returns)
        assert isinstance(fitted, dict)
        assert set(fitted) == {"AAPL", "MSFT"}
        assert fitted is model.results

    def test_accepts_single_string_ticker(self, model, returns):
        fitted = model.get_models("AAPL", "^GSPC", returns)
        assert set(fitted) == {"AAPL"}

    def test_missing_market_ticker_raises_keyerror(self, model, returns):
        with pytest.raises(KeyError, match="market_ticker"):
            model.get_models(["AAPL"], "NOPE", returns)

    def test_missing_asset_ticker_raises_keyerror(self, model, returns):
        with pytest.raises(KeyError, match="missing from returns"):
            model.get_models(["AAPL", "ZZZZ"], "^GSPC", returns)


class TestRequiresFit:
    """Every accessor must raise before :meth:`get_models` is called."""

    @pytest.mark.parametrize(
        "method",
        [
            "get_betas",
            "get_alphas",
            "get_r_squared",
            "get_residuals",
            "get_market_variance",
            "get_systematic_risks",
            "get_firm_specific_risks",
            "get_total_risks",
        ],
    )
    def test_accessor_before_fit_raises(self, model, method):
        with pytest.raises(RuntimeError, match="call get_models"):
            getattr(model, method)()


class TestModelOutputs:
    @pytest.fixture
    def fitted(self, model, returns):
        model.get_models(["AAPL", "MSFT", "KO"], "^GSPC", returns)
        return model

    def test_betas_are_finite_floats(self, fitted):
        betas = fitted.get_betas()
        assert set(betas) == {"AAPL", "MSFT", "KO"}
        for beta in betas.values():
            assert np.isfinite(beta)

    def test_recovers_approximate_betas(self, fitted):
        """The fitted betas should be close to the true generating betas."""
        betas = fitted.get_betas()
        assert betas["AAPL"] == pytest.approx(1.1, abs=0.25)
        assert betas["MSFT"] == pytest.approx(0.9, abs=0.25)
        assert betas["KO"] == pytest.approx(0.4, abs=0.25)

    def test_alphas_and_residuals(self, fitted, returns):
        alphas = fitted.get_alphas()
        residuals = fitted.get_residuals()
        for ticker in ("AAPL", "MSFT", "KO"):
            assert np.isfinite(alphas[ticker])
            assert isinstance(residuals[ticker], pd.Series)
            assert len(residuals[ticker]) == len(returns)

    def test_market_variance_is_annualised(self, fitted, returns):
        assert fitted.get_market_variance() == pytest.approx(
            float(returns["^GSPC"].var(ddof=1)) * TRADING_DAYS
        )
        assert fitted.get_market_variance() > 0.0

    def test_risk_decomposition_adds_up(self, fitted):
        systematic = fitted.get_systematic_risks()
        firm = fitted.get_firm_specific_risks()
        total = fitted.get_total_risks()
        for ticker in systematic:
            assert systematic[ticker] >= 0.0
            assert firm[ticker] >= 0.0
            assert np.isclose(systematic[ticker] + firm[ticker], total[ticker])

    def test_r_squared_matches_the_fitted_models(self, fitted):
        r_squared = fitted.get_r_squared()
        assert set(r_squared) == {"AAPL", "MSFT", "KO"}
        for ticker, model in fitted.results.items():
            assert r_squared[ticker] == pytest.approx(float(model.rsquared))
            assert 0.0 <= r_squared[ticker] <= 1.0

    def test_getters_are_stable_across_repeated_calls(self, fitted):
        """Estimates are derived once at fit time, so call order must not matter."""
        before = (fitted.get_total_risks(), fitted.get_betas())
        fitted.get_systematic_risks()
        fitted.get_firm_specific_risks()
        assert (fitted.get_total_risks(), fitted.get_betas()) == before

    def test_getters_return_independent_copies(self, fitted):
        """Mutating a returned dict must not corrupt the cached estimates."""
        betas = fitted.get_betas()
        betas["AAPL"] = 999.0
        assert fitted.get_betas()["AAPL"] != 999.0
