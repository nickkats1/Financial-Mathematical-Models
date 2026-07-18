"""Tests for the resilience layer added on top of the core models.

Covers:
    - tenacity retry on yfinance transient failures
    - TTL price cache (dedupe + no-cache-on-empty)
    - run_analysis maps pypfopt failures to ValueError
    - SECRET_KEY hardening in create_app
    - Flask-Limiter rate cap on /analyze
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services import AnalysisRequest, run_analysis
from portfolio.data import DataIngestion
from portfolio.data import data_ingestion as di_module


@pytest.fixture(autouse=True)
def _clear_price_cache():
    DataIngestion.clear_price_cache()
    yield
    DataIngestion.clear_price_cache()


@pytest.fixture
def two_asset_prices():
    """Two-asset frame with enough variance for MPT to solve."""
    rng = np.random.default_rng(seed=42)
    dates = pd.bdate_range("2024-01-01", periods=180)
    return pd.DataFrame(
        {
            "AAPL": 100 * np.cumprod(1 + rng.normal(0.0008, 0.012, 180)),
            "MSFT": 200 * np.cumprod(1 + rng.normal(0.0010, 0.015, 180)),
            "^GSPC": 4000 * np.cumprod(1 + rng.normal(0.0005, 0.008, 180)),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# tenacity retry
# ---------------------------------------------------------------------------

class TestRetry:
    def test_yf_download_retries_on_transient_error(self, two_asset_prices):
        """`_yf_download` retries up to 3 times when yfinance raises."""
        mock_dl = MagicMock()
        mock_dl.__getitem__.return_value = two_asset_prices

        side_effects = [
            ConnectionError("transient"),
            ConnectionError("transient"),
            mock_dl,
        ]

        with patch.object(di_module.yf, "download", side_effect=side_effects) as dl:
            di_module._yf_download(["AAPL", "MSFT"], "2024-01-01", "2024-06-01")

        assert dl.call_count == 3

    def test_yf_download_gives_up_after_three_attempts(self):
        with patch.object(
            di_module.yf, "download", side_effect=ConnectionError("dead")
        ) as dl, pytest.raises(ConnectionError):
            di_module._yf_download(["AAPL"], "2024-01-01", "2024-06-01")
        assert dl.call_count == 3


# ---------------------------------------------------------------------------
# TTL price cache
# ---------------------------------------------------------------------------

class TestPriceCache:
    def test_cache_dedupes_across_ticker_orderings(self, two_asset_prices):
        mock_dl = MagicMock()
        mock_dl.__getitem__.return_value = two_asset_prices

        ing = DataIngestion("2024-01-01", "2024-06-01")
        with patch.object(di_module.yf, "download", return_value=mock_dl) as dl:
            ing.fetch_prices(["AAPL", "MSFT"])
            ing.fetch_prices(["MSFT", "AAPL"])
            ing.fetch_prices(["AAPL", "MSFT"])
        assert dl.call_count == 1

    def test_empty_results_are_not_cached(self):
        """An empty response should not poison the cache for the rest of the TTL."""
        empty = MagicMock()
        empty.__getitem__.return_value = pd.DataFrame()

        ing = DataIngestion("2024-01-01", "2024-06-01")
        with patch.object(di_module.yf, "download", return_value=empty) as dl:
            with pytest.raises(ValueError, match="No price data returned"):
                ing.fetch_prices(["FAKE"])
            with pytest.raises(ValueError, match="No price data returned"):
                ing.fetch_prices(["FAKE"])
        # Each call hit yfinance — empty result was not memoised.
        assert dl.call_count == 2


# ---------------------------------------------------------------------------
# run_analysis: friendly MPT failure mapping + dropped tickers
# ---------------------------------------------------------------------------

class TestRunAnalysisFailureMapping:
    def test_pypfopt_failure_is_mapped_to_value_error(self, two_asset_prices):
        """An exception raised by ``portfolio_metrics`` (e.g. solver failure
        on a singular covariance matrix) must surface as a friendly
        ``ValueError`` rather than a 500."""
        mock_dl = MagicMock()
        mock_dl.__getitem__.return_value = two_asset_prices

        request = AnalysisRequest(
            tickers=["AAPL", "MSFT"],
            start_date="2024-01-01",
            end_date="2024-06-01",
            risk_free_rate=0.04,
            risk_aversion=3.0,
            market_ticker="^GSPC",
            asset_classes=[],
        )
        with patch.object(di_module.yf, "download", return_value=mock_dl), \
             patch(
                 "app.services.portfolio_metrics",
                 side_effect=RuntimeError("solver blew up"),
             ), pytest.raises(ValueError, match="max-Sharpe|covariance"):
            run_analysis(request)

    def test_dropped_tickers_recorded_in_result(self, two_asset_prices):
        """Tickers that yfinance silently drops should appear in
        ``AnalysisResult.dropped_tickers``."""
        mock_dl = MagicMock()
        mock_dl.__getitem__.return_value = two_asset_prices  # only AAPL, MSFT, ^GSPC

        request = AnalysisRequest(
            tickers=["AAPL", "MSFT", "FAKE1", "FAKE2"],
            start_date="2024-01-01",
            end_date="2024-06-01",
            risk_free_rate=0.04,
            risk_aversion=3.0,
            market_ticker="^GSPC",
            asset_classes=[],
        )
        with patch.object(di_module.yf, "download", return_value=mock_dl):
            result = run_analysis(request)

        assert set(result.dropped_tickers) == {"FAKE1", "FAKE2"}
        assert result.market_proxy_available is True


# ---------------------------------------------------------------------------
# run_analysis: full result structure + market-proxy / drop edge cases
# ---------------------------------------------------------------------------

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


class TestRunAnalysisResult:
    def test_full_result_is_well_formed(self, two_asset_prices):
        mock_dl = MagicMock()
        mock_dl.__getitem__.return_value = two_asset_prices

        with patch.object(di_module.yf, "download", return_value=mock_dl):
            result = run_analysis(_request(["AAPL", "MSFT"]))

        assert set(result.tickers) == {"AAPL", "MSFT"}
        assert result.market_proxy_available is True
        assert result.dropped_tickers == []
        # MPT weights are long-only and fully invested.
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(w >= -1e-9 for w in result.weights.values())
        # VaR gets more negative as confidence rises; CVaR never exceeds VaR.
        assert result.var_99 <= result.var_95 <= result.var_90
        assert result.cvar_90 <= result.var_90
        assert result.cvar_99 <= result.var_99
        # Single Index Model populated for every asset.
        for ticker in result.tickers:
            assert ticker in result.betas
            assert ticker in result.alphas
            assert ticker in result.r_squared
            assert ticker in result.utility
            assert ticker in result.max_utility
        assert result.market_variance > 0.0

    def test_market_proxy_unavailable_skips_sim(self, two_asset_prices):
        """When the market proxy returns no data, the SIM is skipped but the
        rest of the analysis still renders."""
        no_market = two_asset_prices.drop(columns="^GSPC")
        mock_dl = MagicMock()
        mock_dl.__getitem__.return_value = no_market

        with patch.object(di_module.yf, "download", return_value=mock_dl):
            result = run_analysis(_request(["AAPL", "MSFT"]))

        assert result.market_proxy_available is False
        assert result.betas == {}
        assert result.alphas == {}
        assert result.r_squared == {}
        assert result.market_variance == 0.0

    def test_market_ticker_already_in_universe_not_refetched(self, two_asset_prices):
        """If the market proxy is already among the requested tickers it is not
        appended a second time to the fetch list."""
        mock_dl = MagicMock()
        mock_dl.__getitem__.return_value = two_asset_prices

        with patch.object(di_module.yf, "download", return_value=mock_dl):
            result = run_analysis(_request(["AAPL", "MSFT", "^GSPC"]))

        # ^GSPC is both an asset and the market proxy here.
        assert result.market_proxy_available is True
        assert "^GSPC" in result.tickers

    def test_too_few_tickers_after_drop_raises(self):
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
        mock_dl = MagicMock()
        mock_dl.__getitem__.return_value = one_asset

        with patch.object(
            di_module.yf, "download", return_value=mock_dl
        ), pytest.raises(ValueError, match="Not enough tickers"):
            run_analysis(_request(["AAPL", "FAKE"]))


# ---------------------------------------------------------------------------
# _fit_single_index_model: defensive skip branches
# ---------------------------------------------------------------------------

class TestFitSingleIndexModel:
    def test_empty_when_market_missing_from_prices(self):
        from app.services import _fit_single_index_model

        prices = pd.DataFrame(
            {"AAPL": [1.0, 2.0, 3.0], "MSFT": [1.0, 2.0, 3.0]}
        )
        fit = _fit_single_index_model(prices, ["AAPL"], "^GSPC")
        assert fit["betas"] == {}
        assert fit["market_variance"] == 0.0

    def test_empty_when_returns_are_empty(self):
        from app.services import _fit_single_index_model

        # A single price row yields no returns after dropping the NaN row.
        prices = pd.DataFrame({"AAPL": [1.0], "^GSPC": [1.0]})
        fit = _fit_single_index_model(prices, ["AAPL"], "^GSPC")
        assert fit["betas"] == {}

    def test_empty_when_no_asset_tickers_in_returns(self):
        from app.services import _fit_single_index_model

        prices = pd.DataFrame(
            {"AAPL": [1.0, 1.1, 1.2, 1.3], "^GSPC": [2.0, 2.1, 2.05, 2.2]}
        )
        fit = _fit_single_index_model(prices, ["ZZZZ"], "^GSPC")
        assert fit["betas"] == {}

    def test_populated_when_market_present(self):
        from app.services import _fit_single_index_model

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


# ---------------------------------------------------------------------------
# SECRET_KEY hardening
# ---------------------------------------------------------------------------

class TestSecretKey:
    """Exercise ``_resolve_secret_key`` directly so we never reload the
    ``app`` module — reloading rebinds the limiter and breaks downstream
    rate-limit tests."""

    def test_production_refuses_missing_key(self, monkeypatch):
        from app import _resolve_secret_key

        monkeypatch.setenv("FLASK_ENV", "production")
        monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
        with pytest.raises(RuntimeError, match="FLASK_SECRET_KEY must be set"):
            _resolve_secret_key(test_config=None)

    def test_production_refuses_placeholder_key(self, monkeypatch):
        from app import _resolve_secret_key

        monkeypatch.setenv("FLASK_ENV", "production")
        monkeypatch.setenv("FLASK_SECRET_KEY", "dev-secret-change-me")
        with pytest.raises(RuntimeError, match="FLASK_SECRET_KEY"):
            _resolve_secret_key(test_config=None)

    def test_production_accepts_real_key(self, monkeypatch):
        from app import _resolve_secret_key

        monkeypatch.setenv("FLASK_ENV", "production")
        monkeypatch.setenv("FLASK_SECRET_KEY", "a" * 64)
        assert _resolve_secret_key(test_config=None) == "a" * 64

    def test_dev_tolerates_missing_key(self, monkeypatch):
        from app import _resolve_secret_key

        monkeypatch.delenv("FLASK_ENV", raising=False)
        monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
        assert _resolve_secret_key(test_config=None) == "dev-secret-change-me"

    def test_testing_mode_uses_random_key(self):
        from app import _resolve_secret_key

        key = _resolve_secret_key(test_config={"TESTING": True})
        assert isinstance(key, str) and len(key) >= 32


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimit:
    @pytest.fixture
    def live_app(self, monkeypatch):
        """A non-test Flask app with the limiter active and storage reset.

        CSRF is left enabled; tests fetch a valid token from ``/`` and pass
        it through, so the rate-limit assertion exercises the real
        production middleware order (limiter → CSRF → route).
        """
        monkeypatch.delenv("FLASK_ENV", raising=False)
        monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)

        from app import create_app, limiter

        app = create_app()
        limiter.enabled = True
        limiter.reset()
        yield app
        limiter.reset()
        limiter.enabled = False

    @staticmethod
    def _csrf_token(client) -> str:
        """Pull a fresh CSRF token from a GET on the index page."""
        body = client.get("/").data.decode()
        marker = 'name="csrf_token" value="'
        start = body.index(marker) + len(marker)
        end = body.index('"', start)
        return body[start:end]

    def test_analyze_returns_429_after_ten_requests(self, live_app):
        """The 10/min cap on /analyze must trip after the limit is reached."""
        client = live_app.test_client()
        token = self._csrf_token(client)

        codes = []
        for _ in range(15):
            r = client.post(
                "/analyze",
                data={
                    "csrf_token": token,
                    "tickers": "",  # invalid → 400 fast, no yfinance call
                    "start_date": "2024-01-01",
                    "end_date": "2024-06-01",
                    "risk_free_rate": "0.04",
                    "risk_aversion": "3.0",
                },
            )
            codes.append(r.status_code)
        assert codes.count(429) >= 2, f"expected rate limit to trip, got {codes}"

    def test_healthz_is_exempt(self, live_app):
        """``/healthz`` must not be subject to any rate limit."""
        client = live_app.test_client()
        for _ in range(150):
            assert client.get("/healthz").status_code == 200

    def test_analyze_requires_csrf_token(self, live_app):
        """A POST without a CSRF token should be rejected."""
        client = live_app.test_client()
        r = client.post(
            "/analyze",
            data={"tickers": "AAPL, MSFT"},
        )
        assert r.status_code == 400
