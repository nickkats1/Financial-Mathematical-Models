from pypfopt import expected_returns, risk_models, EfficientFrontier
import numpy as np
import pandas as pd


# --- Get expected returns, Volatility, Sharpe Ratio ---

# --- Expected returns ---
def get_expected_returns(dataframe: pd.DataFrame) -> pd.Series:
    """Get Expected Returns from dataframe"""
    return expected_returns.mean_historical_return(dataframe)


# --- Volatility ---

def get_volatility(dataframe: pd.DataFrame) -> pd.Series:
    """Get Volatility from dataframe"""
    return risk_models.sample_cov(dataframe)


# --- get efficient frontier ---


def get_efficient_frontier(dataframe: pd.DataFrame) -> EfficientFrontier:
    """get efficient frontier from dataframe."""
    
    # mu is expected returns
    
    mu = get_expected_returns(dataframe)
    
    # S is volatility
    
    S = get_volatility(dataframe)
    
    return EfficientFrontier(mu, S)


# --- Get Portfolio Weights ---

def get_weights(ef: EfficientFrontier, risk_free_rate: float) -> pd.Series:
    """Get Weights from optimized portfolio and return them."""
    
    ef = EfficientFrontier(ef)
    
    weights = ef.max_sharpe(risk_free_rate)
    
    # clean weights
    
    weights = ef.clean_weights()
    
    return weights
    

# --- Sharpe Ratio ---

def sharpe_ratio(
    mu: expected_returns,
    S: risk_models,
    risk_free_rate: float
) -> float:
    """Get Sharpe Ratio manually"""
    
    return (mu - risk_free_rate) / S









    

    
