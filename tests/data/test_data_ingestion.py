"""Tests for :class:`portfolio.data.data_ingestion.DataIngestion`."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from portfolio.data import DataIngestion


@pytest.fixture
def data_ingestion():
    return DataIngestion()


class TestDataIngestion:
    def test_fetch_returns_dataframe(self, data_ingestion, fake_prices):
        """`_fetch` returns a clean DataFrame with the expected columns."""
        mock_dl = MagicMock()
        mock_dl.__getitem__.return_value = fake_prices

        with patch("portfolio.data.data_ingestion.yf.download", return_value=mock_dl):
            result = data_ingestion._fetch(["AAPL"])

        assert isinstance(result, pd.DataFrame)
        assert "META" in result.columns
        assert "QQQ" in result.columns
        assert not result.isnull().any().any()
        assert not result.duplicated().any()

    def test_fetch_raises_when_no_data(self, data_ingestion):
        """`_fetch` raises ValueError when yfinance returns an empty frame."""
        mock_dl = MagicMock()
        mock_dl.__getitem__.return_value = pd.DataFrame()

        with patch("portfolio.data.data_ingestion.yf.download", return_value=mock_dl):
            with pytest.raises(ValueError, match="No price data returned"):
                data_ingestion._fetch(["FAKE"])

    def test_fetch_raises_when_tickers_empty(self, data_ingestion):
        """`_fetch` raises ValueError when given no tickers."""
        with pytest.raises(ValueError, match="tickers must not be empty"):
            data_ingestion._fetch([])

    def test_compute_returns_drops_first_row(self, data_ingestion, fake_prices):
        """`compute_returns` drops the leading NaN row from `pct_change`."""
        returns = data_ingestion.compute_returns(fake_prices)
        assert len(returns) == len(fake_prices) - 1
        assert not returns.isnull().any().any()
