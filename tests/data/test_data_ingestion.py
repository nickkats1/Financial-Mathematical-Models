"""Tests for :class:`portfolio.data.data_ingestion.DataIngestion`."""

from unittest.mock import patch

import pandas as pd
import pytest

from portfolio.config import DEFAULT_CONFIG, get_asset_class
from portfolio.data import DataIngestion, compute_returns
from portfolio.data import data_ingestion as di_module


@pytest.fixture
def data_ingestion():
    return DataIngestion("2024-01-01", "2024-01-10")


class TestInit:
    def test_defaults_fall_back_to_config_dates(self):
        ing = DataIngestion()
        assert ing.start_date == DEFAULT_CONFIG.start_date
        assert ing.end_date == DEFAULT_CONFIG.end_date

    def test_explicit_dates_are_used(self):
        ing = DataIngestion("2020-01-01", "2020-12-31")
        assert ing.start_date == "2020-01-01"
        assert ing.end_date == "2020-12-31"


class TestFetch:
    def test_fetch_returns_clean_dataframe(self, data_ingestion, fake_prices, patched_download):
        """`_fetch` returns the downloaded frame with NaN and duplicate rows dropped."""
        with patched_download(fake_prices):
            result = data_ingestion._fetch(["AAPL"])

        assert isinstance(result, pd.DataFrame)
        assert not result.isnull().any().any()
        assert not result.duplicated().any()

    def test_fetch_accepts_single_string_ticker(
        self, data_ingestion, fake_prices, patched_download
    ):
        with patched_download(fake_prices):
            result = data_ingestion._fetch("META")
        assert "META" in result.columns

    def test_fetch_public_wrapper(self, data_ingestion, fake_prices, patched_download):
        with patched_download(fake_prices):
            result = data_ingestion.fetch_prices(["AAPL", "MSFT"])
        assert isinstance(result, pd.DataFrame)

    def test_single_ticker_series_is_framed(self, data_ingestion, patched_download):
        """When yfinance returns a Series (single ticker) it is framed."""
        dates = pd.bdate_range("2024-01-01", periods=3)
        series = pd.Series([10.0, 11.0, 12.0], index=dates, name="Close")
        with patched_download(series):
            result = data_ingestion._fetch("AAPL")
        assert list(result.columns) == ["AAPL"]
        assert len(result) == 3

    @pytest.mark.parametrize(
        "tickers, match",
        [(["FAKE"], "No price data returned"),
         ([], "must not be empty"), (["", ""], "must not be empty")],
    )
    def test_fetch_rejects_empty_input_and_empty_results(
        self, data_ingestion, patched_download, tickers, match
    ):
        """Falsy tickers are dropped before a download is attempted."""
        with patched_download(pd.DataFrame()), pytest.raises(ValueError, match=match):
            data_ingestion._fetch(tickers)

    def test_download_error_is_wrapped_as_value_error(self, data_ingestion):
        """A yfinance failure surfaces as a friendly ValueError, not a raw error."""
        patched = patch.object(di_module, "_yf_download", side_effect=ConnectionError("dead"))
        with patched, pytest.raises(ValueError, match="Could not fetch prices"):
            data_ingestion._fetch(["AAPL"])


class TestRetry:
    """Only transport failures are retried — anything else would fail again identically."""

    def test_retries_on_transient_error(self, fake_prices, download_result):
        side_effects = [
            ConnectionError("transient"),
            ConnectionError("transient"),
            download_result(fake_prices),
        ]
        with patch.object(di_module.yf, "download", side_effect=side_effects) as dl:
            di_module._yf_download(["AAPL", "MSFT"], "2024-01-01", "2024-06-01")

        assert dl.call_count == 3

    @pytest.mark.parametrize(
        "side_effect, expected_calls, expected_error",
        [(ConnectionError("dead"), 3, ConnectionError), (ValueError("bad frame"), 1, ValueError)],
    )
    def test_retry_policy(self, side_effect, expected_calls, expected_error):
        patched = patch.object(di_module.yf, "download", side_effect=side_effect)
        with patched as dl, pytest.raises(expected_error):
            di_module._yf_download(["AAPL"], "2024-01-01", "2024-06-01")
        assert dl.call_count == expected_calls


class TestCache:
    def test_cache_returns_isolated_copy(self, data_ingestion, fake_prices, patched_download):
        """Mutating a returned frame must not corrupt the cached copy."""
        with patched_download(fake_prices) as dl:
            first = data_ingestion._fetch(["AAPL", "MSFT"])
            first.iloc[0, 0] = -999.0
            second = data_ingestion._fetch(["AAPL", "MSFT"])

        assert dl.call_count == 1  # second call served from cache
        assert second.iloc[0, 0] != -999.0

    def test_cache_dedupes_across_ticker_orderings(
        self, data_ingestion, fake_prices, patched_download
    ):
        with patched_download(fake_prices) as dl:
            data_ingestion.fetch_prices(["AAPL", "MSFT"])
            data_ingestion.fetch_prices(["MSFT", "AAPL"])
            data_ingestion.fetch_prices(["AAPL", "MSFT"])
        assert dl.call_count == 1

    def test_empty_results_are_not_cached(self, data_ingestion, patched_download):
        """An empty response should not poison the cache for the rest of the TTL."""
        with patched_download(pd.DataFrame()) as dl:
            for _ in range(2):
                with pytest.raises(ValueError, match="No price data returned"):
                    data_ingestion.fetch_prices(["FAKE"])

        assert dl.call_count == 2

    def test_partial_hit_downloads_only_the_new_ticker(
        self, data_ingestion, fake_prices, patched_download
    ):
        """Growing a universe by one symbol refetches only that symbol."""
        with patched_download(fake_prices[["AAPL", "MSFT"]]):
            data_ingestion.fetch_prices(["AAPL", "MSFT"])
        with patched_download(fake_prices[["META"]]) as dl:
            result = data_ingestion.fetch_prices(["AAPL", "MSFT", "META"])

        assert dl.call_args.kwargs["tickers"] == ["META"]
        assert set(result.columns) == {"AAPL", "MSFT", "META"}

    def test_all_nan_columns_are_not_cached(self, data_ingestion, fake_prices, patched_download):
        """yfinance reports a symbol it could not fetch as an all-NaN column."""
        frame = fake_prices[["AAPL", "MSFT"]].copy()
        frame["MSFT"] = float("nan")
        with patched_download(frame) as dl:
            first = data_ingestion.fetch_prices(["AAPL", "MSFT"])
            second = data_ingestion.fetch_prices(["AAPL", "MSFT"])

        assert "MSFT" not in first.columns
        assert "MSFT" not in second.columns
        assert dl.call_count == 2


class TestConvenienceMethods:
    @pytest.mark.parametrize(
        "method, args, expected",
        [
            ("get_market_prices", (), DEFAULT_CONFIG.market_ticker),
            ("get_all_prices", (), list(DEFAULT_CONFIG.all_tickers)),
            *[("get_asset_class_prices", (n,), list(get_asset_class(n).tickers))
              for n in ("stocks", "etfs", "bonds", "crypto")],
        ],
    )
    def test_delegates_to_fetch(self, data_ingestion, method, args, expected):
        sentinel = pd.DataFrame({"X": [1.0]})
        with patch.object(DataIngestion, "_fetch", return_value=sentinel) as fetch:
            result = getattr(data_ingestion, method)(*args)
        fetch.assert_called_once_with(expected)
        assert result is sentinel

    def test_asset_class_prices_rejects_unknown_name(self, data_ingestion):
        with pytest.raises(ValueError, match=r"not found\. Available: "):
            data_ingestion.get_asset_class_prices("equities")


class TestConfigurePriceCache:
    def test_rebuilds_the_cache_with_the_given_sizing(self):
        di_module.configure_price_cache(ttl_seconds=1, max_entries=2)
        try:
            assert di_module._price_cache.maxsize == 2
            assert di_module._price_cache.ttl == 1
        finally:
            di_module.configure_price_cache(
                ttl_seconds=di_module._CACHE_TTL_SECONDS,
                max_entries=di_module._CACHE_MAX_ENTRIES,
            )


class TestComputeReturns:
    def test_compute_returns_drops_the_leading_nan_row(self):
        prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
        returns = compute_returns(prices)
        assert returns["A"].tolist() == pytest.approx([0.1, -0.1])
        assert not returns.isnull().any().any()
