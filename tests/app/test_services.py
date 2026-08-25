"""Tests for :mod:`app.services` — the analysis pipeline and SIM adapter."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.forms import AnalysisRequest
from app.services import _fit_single_index_model, run_analysis


def _request(tickers, market_ticker="^GSPC"):
    return AnalysisRequest(
        tickers=tickers,
        start_date="2024-01-01",
        end_date="2024-06-01",
        risk_free_rate=0.04,
        risk_aversion=3.0,
        market_ticker=market_ticker,
        asset_classes=[],
    )


class TestRunAnalysisFailureMapping:
    def test_pypfopt_failure_is_mapped_to_value_error(self, two_asset_prices, patched_download):
        """An exception raised by ``portfolio_metrics`` (e.g. solver failure on
        a singular covariance matrix) must surface as a friendly ``ValueError``
        rather than a 500."""
        with patched_download(two_asset_prices), patch(
            "app.services.portfolio_metrics",
            side_effect=RuntimeError("solver blew up"),
        ), pytest.raises(ValueError, match="max-Sharpe|covariance"):
            run_analysis(_request(["AAPL", "MSFT"]))

    def test_dropped_tickers_recorded_in_result(self, two_asset_prices, patched_download):
        """Tickers that yfinance silently drops should appear in
        ``AnalysisResult.dropped_tickers``."""
        with patched_download(two_asset_prices):
            result = run_analysis(_request(["AAPL", "MSFT", "FAKE1", "FAKE2"]))

        assert set(result.dropped_tickers) == {"FAKE1", "FAKE2"}
        assert result.market_proxy_available is True


class TestRunAnalysisResult:
    def test_full_result_is_well_formed(self, two_asset_prices, patched_download):
        with patched_download(two_asset_prices):
            result = run_analysis(_request(["AAPL", "MSFT"]))

        assert set(result.tickers) == {"AAPL", "MSFT"}
        assert result.market_proxy_available is True
        assert result.dropped_tickers == ()
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(w >= -1e-9 for w in result.weights.values())
        assert result.var_99 <= result.var_95 <= result.var_90
        assert result.cvar_90 <= result.var_90
        assert result.cvar_99 <= result.var_99
        for ticker in result.tickers:
            assert ticker in result.betas
            assert ticker in result.alphas
            assert ticker in result.r_squared
            assert ticker in result.utility
            assert ticker in result.max_utility
        assert result.market_variance > 0.0

    def test_market_proxy_unavailable_skips_sim(self, two_asset_prices, patched_download):
        """When the market proxy returns no data, the SIM is skipped but the
        rest of the analysis still renders."""
        no_market = two_asset_prices.drop(columns="^GSPC")
        with patched_download(no_market):
            result = run_analysis(_request(["AAPL", "MSFT"]))

        assert result.market_proxy_available is False
        assert result.betas == {}
        assert result.alphas == {}
        assert result.r_squared == {}
        assert result.market_variance == 0.0

    def test_market_ticker_already_in_universe_not_refetched(
        self, two_asset_prices, patched_download
    ):
        """If the market proxy is already among the requested tickers it is not
        appended a second time to the fetch list."""
        with patched_download(two_asset_prices):
            result = run_analysis(_request(["AAPL", "MSFT", "^GSPC"]))

        # ^GSPC is both an asset and the market proxy here.
        assert result.market_proxy_available is True
        assert "^GSPC" in result.tickers

    def test_too_few_tickers_after_drop_raises(self, patched_download):
        """If yfinance drops all but one asset, the analysis raises a friendly
        ValueError instead of trying to optimise a single-asset universe."""
        dates = pd.bdate_range("2024-01-01", periods=60)
        rng = np.random.default_rng(seed=2)
        one_asset = pd.DataFrame(
            {
                "AAPL": 100 * np.cumprod(1 + rng.normal(0.0008, 0.012, 60)),
                "^GSPC": 4000 * np.cumprod(1 + rng.normal(0.0005, 0.008, 60)),
            },
            index=dates,
        )
        with patched_download(one_asset), pytest.raises(ValueError, match="Not enough tickers"):
            run_analysis(_request(["AAPL", "FAKE"]))


class TestFitSingleIndexModel:
    def test_empty_when_market_missing_from_prices(self):
        prices = pd.DataFrame({"AAPL": [1.0, 2.0, 3.0], "MSFT": [1.0, 2.0, 3.0]})
        fit = _fit_single_index_model(prices, ["AAPL"], "^GSPC")
        assert fit["betas"] == {}
        assert fit["market_variance"] == 0.0

    def test_empty_when_returns_are_empty(self):
        # A single price row yields no returns after dropping the NaN row.
        prices = pd.DataFrame({"AAPL": [1.0], "^GSPC": [1.0]})
        fit = _fit_single_index_model(prices, ["AAPL"], "^GSPC")
        assert fit["betas"] == {}

    def test_empty_when_no_asset_tickers_in_returns(self):
        prices = pd.DataFrame(
            {"AAPL": [1.0, 1.1, 1.2, 1.3], "^GSPC": [2.0, 2.1, 2.05, 2.2]}
        )
        fit = _fit_single_index_model(prices, ["ZZZZ"], "^GSPC")
        assert fit["betas"] == {}

    def test_populated_when_market_present(self):
        rng = np.random.default_rng(seed=3)
        n = 60
        market = rng.normal(0.0004, 0.009, n)
        prices = pd.DataFrame(
            {
                "AAPL": 100 * np.cumprod(1 + (0.0002 + 1.1 * market)),
                "^GSPC": 4000 * np.cumprod(1 + market),
            }
        )
        fit = _fit_single_index_model(prices, ["AAPL"], "^GSPC")
        assert "AAPL" in fit["betas"]
        assert "AAPL" in fit["r_squared"]
        assert fit["market_variance"] > 0.0
