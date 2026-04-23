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



# --- Maximum Utility ---

def max_utility(
    dataframe: pd.DataFrame,
    risk_aversion: float,
    risk_free_rate: float,
    scaling_factor: float = 0.5
) -> pd.DataFrame:
    """Maximize utility of investors returns.
    
    Args:
        dataframe: insert dataframe with prices to be used through pyportfolio.
        risk_aversion: level of risk investor is willing to take on.
        risk_free_rate: The 'cushion' of the portfolio.
        scaling_factor: constant of 0.5.
        
        
    Returns:
        max_u: Utility maximized.
    """
    # er is expected returns
    
    er = expected_returns.mean_historical_return(dataframe)
    
    variance_of_returns = risk_models.sample_cov(dataframe)
    
    y_star = (er - risk_free_rate) / (risk_aversion * variance_of_returns)
    
    max_u = (
        risk_free_rate + y_star * (er - risk_free_rate) - 
        (0.5) * risk_aversion * y_star ** 2 * variance_of_returns
    )
    return max_u
        
