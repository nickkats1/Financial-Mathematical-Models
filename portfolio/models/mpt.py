from pypfopt import risk_models, EfficientFrontier, expected_returns
import numpy as np
import pandas as pd

from typing import Dict, Any

def portfolio_metrics(prices: pd.DataFrame, risk_free_rate: float) -> Dict[str, Any]:
    """Get Portfolio metrics given input prices.
    
    Args:
        prices: input prices of portfolio.
        risk_free_rate: The cushion of the portfolio.
        
    Returns:
        dictionary of expected returns, volatility, sharpe ratio, and weights.
    """
    
    # mu is expected returns
    
    mu = expected_returns.mean_historical_return(prices)
    
    # S is Volatility.
    
    s = risk_models.sample_cov(prices)
    
    # Efficient Frontier
    
    ef = EfficientFrontier(mu, s)
    
    weights = ef.max_sharpe(risk_free_rate)
    weights = ef.clean_weights()
    
    
    expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)
    
    return {
        "expected annual return": expected_annual_return,
        "annual volatility": annual_volatility,
        "sharpe ratio": sharpe_ratio,
        "weights": weights
    }
    
    
    
    
