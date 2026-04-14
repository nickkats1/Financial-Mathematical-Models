import numpy as np
import pandas as pd


from portfolio import config


# --- Value at Risk ---

def get_var(dataframe: pd.DataFrame) -> float:
    """Get Value At Risk at 95%"""
    returns = dataframe.pct_change().dropna().drop_duplicates()
    return np.percentile(returns, (1 - config.var_95) * 100)



# --- Conditional Value At Risk ---


def get_cvar(dataframe: pd.DataFrame) -> float:
    """get Conditional Value at Risk."""
    
    returns = dataframe.pct_change().dropna().drop_duplicates()
    
    var = get_var(dataframe)
 
    tail_risk = returns[returns < var]
    
    return np.mean(tail_risk)




