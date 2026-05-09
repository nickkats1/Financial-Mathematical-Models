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
        ) as dl:
            with pytest.raises(ConnectionError):
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
             ):
            with pytest.raises(ValueError, match="max-Sharpe|covariance"):
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
