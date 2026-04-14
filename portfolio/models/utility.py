import pandas as pd
from pypfopt import expected_returns, risk_models


def get_utility(dataframe: pd.DataFrame, risk_aversion: float, scaling_factor: float = 0.5):
    """Get utility of investor based on level of risk aversion.
    
    Args:
        dataframe: dataframe of prices from yfinance.
        risk_aversion: Level of risk aversion from investor.
        scaling_factor: 0.5, for: U = Expected Returns - 0.5 - variance of returns.
    
    Returns:
        Utility (U): The investors level of Utility based on level of risk aversion.
    """
    

    
    er = expected_returns.mean_historical_return(dataframe)
    variance_of_returns = risk_models.sample_cov(dataframe)
    u = er - scaling_factor * risk_aversion * variance_of_returns
    return u
