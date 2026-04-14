import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from portfolio.data.data_ingestion import DataIngestion


@pytest.fixture
def data_ingestion():
    return DataIngestion()




class TestDataIngestion:
    def test_fetch(self, data_ingestion, fake_prices):
        """test fetch returns correct DataFrame"""
        mock_dl = MagicMock()
        mock_dl.__getitem__.return_value = fake_prices

        with patch("portfolio.data.data_ingestion.yf.download", return_value=mock_dl):
            result = data_ingestion._fetch(["AAPL"])

        assert isinstance(result, pd.DataFrame)
        assert "META" in result.columns
        assert "QQQ" in result.columns
        assert not result.isnull().any().any()
        assert not result.duplicated().any()
        
    def test_raises_value_error(self, data_ingestion):
        """test ValueError is raised when yfinance returns no data"""
        mock_dl = MagicMock()
        mock_dl.__getitem__.return_value = pd.DataFrame() 
        with patch("portfolio.data.data_ingestion.yf.download", return_value=mock_dl):
            with pytest.raises(ValueError, match="No price data returned"):
                data_ingestion._fetch(["FAKE"])
                

        
        
    