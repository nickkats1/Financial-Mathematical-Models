import numpy as np
import pandas as pd
import statsmodels.api as sm

class SingleIndexModel:
    """
    Single Index Model (SIM) implementation.

    Regresses each asset's returns against the market (S&P 500)
    to decompose risk into systematic and firm-specific components.
    """

    def __init__(self):
        self.results = {}
        self._market_returns = None

    def get_models(
        self,
        tickers: list[str] | str,
        market_ticker: str,
        returns: pd.DataFrame
    ) -> dict[str, sm.regression.linear_model.RegressionResultsWrapper]:
        """
        Get OLS models for each ticker regressed on the market.
        """
        self._market_returns = returns[market_ticker]
        X = sm.add_constant(self._market_returns)
        models = {}
        for ticker in tickers:
            y = returns[ticker]
            model = sm.OLS(y, X).fit()
            models[ticker] = model
        self.results = models
        return models

    def get_betas(self) -> dict[str, float]:
        """Raw OLS betas for each stock."""
        return {t: model.params.iloc[1] for t, model in self.results.items()}

    def get_alphas(self) -> dict[str, float]:
        """OLS intercepts (alphas) for each stock."""
        return {t: model.params.iloc[0] for t, model in self.results.items()}

    def get_residuals(self) -> dict[str, pd.Series]:
        """OLS residuals (error terms) for each stock."""
        return {t: model.resid for t, model in self.results.items()}

    def get_market_variance(self) -> float:
        """Variance of the market returns."""
        return float(np.var(self._market_returns))

    def get_systematic_risks(self) -> dict[str, float]:
        """Systematic risk: β² × σ²_m."""
        betas = self.get_betas()
        market_var = self.get_market_variance()
        return {t: beta ** 2 * market_var for t, beta in betas.items()}

    def get_firm_specific_risks(self) -> dict[str, float]:
        """Firm-specific risk: variance of residuals."""
        return {t: float(np.var(model.resid)) for t, model in self.results.items()}

    def get_total_risks(self) -> dict[str, float]:
        """Total risk: systematic + firm-specific."""
        systematic = self.get_systematic_risks()
        firm_specific = self.get_firm_specific_risks()
        return {t: systematic[t] + firm_specific[t] for t in systematic}