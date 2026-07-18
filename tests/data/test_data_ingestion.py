"""Tests for :class:`portfolio.data.data_ingestion.DataIngestion`."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from portfolio import config
from portfolio.data import DataIngestion
from portfolio.data import data_ingestion as di_module


@pytest.fixture(autouse=True)
def _clear_price_cache():
    """Isolate the module-level TTL cache between tests."""
    DataIngestion.clear_price_cache()
    yield
    DataIngestion.clear_price_cache()


@pytest.fixture
def data_ingestion():
    return DataIngestion("2024-01-01", "2024-01-10")


def _mock_download(frame):
    """Build a MagicMock standing in for a yfinance download result."""
    mock_dl = MagicMock()
    mock_dl.__getitem__.return_value = frame
    return mock_dl


class TestInit:
    def test_defaults_fall_back_to_config_dates(self):
        ing = DataIngestion()
        assert ing.start_date == config.start_date
        assert ing.end_date == config.end_date

    def test_explicit_dates_are_used(self):
        ing = DataIngestion("2020-01-01", "2020-12-31")
        assert ing.start_date == "2020-01-01"
        assert ing.end_date == "2020-12-31"


class TestFetch:
    def test_fetch_returns_clean_dataframe(self, data_ingestion, fake_prices):
        """`_fetch` returns a clean DataFrame with the expected columns."""
        with patch(
            "portfolio.data.data_ingestion.yf.download",
            return_value=_mock_download(fake_prices),
        ):
            result = data_ingestion._fetch(["AAPL"])

        assert isinstance(result, pd.DataFrame)
        assert "META" in result.columns
        assert "QQQ" in result.columns
        assert not result.isnull().any().any()
        assert not result.duplicated().any()

    def test_fetch_accepts_single_string_ticker(self, data_ingestion, fake_prices):
        with patch(
            "portfolio.data.data_ingestion.yf.download",
            return_value=_mock_download(fake_prices),
        ):
            result = data_ingestion._fetch("META")
        assert "META" in result.columns

    def test_fetch_public_wrapper(self, data_ingestion, fake_prices):
        with patch(
            "portfolio.data.data_ingestion.yf.download",
            return_value=_mock_download(fake_prices),
        ):
            result = data_ingestion.fetch_prices(["AAPL", "MSFT"])
        assert isinstance(result, pd.DataFrame)

    def test_single_ticker_series_is_framed(self, data_ingestion):
        """When yfinance returns a Series (single ticker) it is framed."""
        dates = pd.bdate_range("2024-01-01", periods=3)
        series = pd.Series([10.0, 11.0, 12.0], index=dates, name="Close")
        with patch(
            "portfolio.data.data_ingestion.yf.download",
            return_value=_mock_download(series),
        ):
            result = data_ingestion._fetch("AAPL")
        assert list(result.columns) == ["AAPL"]
        assert len(result) == 3

    def test_fetch_raises_when_no_data(self, data_ingestion):
        """`_fetch` raises ValueError when yfinance returns an empty frame."""
        with patch(
            "portfolio.data.data_ingestion.yf.download",
            return_value=_mock_download(pd.DataFrame()),
        ), pytest.raises(ValueError, match="No price data returned"):
            data_ingestion._fetch(["FAKE"])

    def test_fetch_raises_when_tickers_empty(self, data_ingestion):
        """`_fetch` raises ValueError when given no tickers."""
        with pytest.raises(ValueError, match="tickers must not be empty"):
            data_ingestion._fetch([])

    def test_fetch_filters_falsy_tickers(self, data_ingestion):
        """Empty-string tickers are dropped; an all-empty list still raises."""
        with pytest.raises(ValueError, match="tickers must not be empty"):
            data_ingestion._fetch(["", ""])

    def test_download_error_is_wrapped_as_value_error(self, data_ingestion):
        """A yfinance failure surfaces as a friendly ValueError, not a raw error."""
        with patch.object(
            di_module, "_yf_download", side_effect=ConnectionError("dead")
        ), pytest.raises(ValueError, match="Could not fetch prices"):
            data_ingestion._fetch(["AAPL"])


class TestCache:
    def test_cache_returns_isolated_copy(self, data_ingestion, fake_prices):
        """Mutating a returned frame must not corrupt the cached copy."""
        with patch(
            "portfolio.data.data_ingestion.yf.download",
            return_value=_mock_download(fake_prices),
        ) as dl:
            first = data_ingestion._fetch(["AAPL", "MSFT"])
            first.iloc[0, 0] = -999.0
            second = data_ingestion._fetch(["AAPL", "MSFT"])

        assert dl.call_count == 1  # second call served from cache
        assert second.iloc[0, 0] != -999.0


class TestConvenienceMethods:
    @pytest.mark.parametrize(
        "method, expected",
        [
            ("get_stock_prices", config.stock_tickers),
            ("get_etf_prices", config.etf_tickers),
            ("get_bond_prices", config.bond_tickers),
            ("get_crypto_prices", config.crypto_tickers),
            ("get_sp500_prices", config.sp500_ticker),
            ("get_all_prices", config.all_tickers),
        ],
    )
    def test_convenience_method_delegates_to_fetch(
        self, data_ingestion, method, expected
    ):
        sentinel = pd.DataFrame({"X": [1.0]})
        with patch.object(
            DataIngestion, "_fetch", return_value=sentinel
        ) as fetch:
            result = getattr(data_ingestion, method)()
        fetch.assert_called_once_with(expected)
        assert result is sentinel


class TestComputeReturns:
    def test_compute_returns_drops_first_row(self, fake_prices):
        """`compute_returns` drops the leading NaN row from `pct_change`."""
        returns = DataIngestion.compute_returns(fake_prices)
        assert len(returns) == len(fake_prices) - 1
        assert not returns.isnull().any().any()

    def test_compute_returns_values(self):
        prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
        returns = DataIngestion.compute_returns(prices)
        assert returns["A"].tolist() == pytest.approx([0.1, -0.1])
