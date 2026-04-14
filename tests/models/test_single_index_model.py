import pytest
from unittest.mock import Mock, patch
import statsmodels.api as sm
import pandas as pd
import numpy as np

from portfolio.models.single_index_model import SingleIndexModel


@pytest.fixture
def single_index_model():
    return SingleIndexModel()


class TestSingleIndexModel:
    """Test Single Index Model"""
    
    def test_get_models(self, fake_prices, single_index_model, dummy_config):
        """Test get models method"""
        get_models = single_index_model.get_models(
            tickers=dummy_config["all_tickers"],
            market_ticker=dummy_config["sp500_ticker"],
            returns=fake_prices
        )
        
        assert get_models is not None
        
        assert isinstance(get_models, dict)
        
        assert "^GSPC" in get_models.keys()
        assert "NVDA" in get_models.keys()
        
        expected_tickers = [
            t for t in dummy_config['all_tickers'] if t in fake_prices.columns
        ]
        
        assert set(expected_tickers) <= set(get_models.keys())
        
        
    def test_get_model_methods(self, fake_prices, single_index_model, dummy_config):
        """test methods derived from get_model"""
        available_tickers = [
            t for t in dummy_config['all_tickers'] if t in fake_prices.columns
        ]
    
        # instance of get models.

        single_index_model.get_models(
            tickers=available_tickers,
            market_ticker=dummy_config['sp500_ticker'],
            returns=fake_prices
        )
        
        # betas, alphas, residuals for testing
        betas = single_index_model.get_betas()
        alphas = single_index_model.get_alphas()
        residuals = single_index_model.get_residuals()
        systematic_risks = single_index_model.get_systematic_risks()
        firm_specific_risk = single_index_model.get_firm_specific_risks()
        total_risk = single_index_model.get_total_risks()
        
        assert isinstance(betas, dict)
        assert isinstance(alphas, dict)
        assert isinstance(residuals, dict)
        assert isinstance(systematic_risks, dict)
        assert isinstance(firm_specific_risk, dict)
        assert isinstance(total_risk, dict)
        
        
        
        for ticker in available_tickers:
            assert ticker in betas
            assert ticker in alphas
            assert ticker in residuals
            assert ticker in systematic_risks
            assert ticker in firm_specific_risk
            assert ticker in total_risk
            assert isinstance(betas[ticker], np.floating)
            assert isinstance(alphas[ticker], np.floating)
            assert isinstance(residuals[ticker], pd.Series)
            assert len(residuals[ticker]) == len(fake_prices)
            assert len(residuals[ticker]) == len(fake_prices)
            assert not np.isnan(betas[ticker])
            assert not np.isinf(betas[ticker])
            assert set(betas.keys()) == set(available_tickers)
            assert np.isclose(
                systematic_risks[ticker] + firm_specific_risk[ticker],
                total_risk[ticker]
            )
            
        
        

