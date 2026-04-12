import numpy as np
import pandas as pd
import statsmodels.api as sm


from src import config
from src.data.data_ingestion import DataIngestion


class SingleIndexModel:
    """Single Index Model (SIM) implementation.

    Regresses each asset's returns against the market (S&P 500)
    to decompose risk into systematic and firm-specific components.
    """

    def __init__(self):
        self.config = config
        self.ingestion = DataIngestion()

        returns = self.ingestion.get_all_prices()
        market_returns = self.ingestion.get_sp500_prices().squeeze()
        shared_index = returns.index.intersection(market_returns.index)

        self._stock_returns = returns.loc[shared_index]
        self._market_returns = market_returns.loc[shared_index]

        self._models: dict[str, sm.regression.linear_model.RegressionResultsWrapper] = {}
        self._fit_all()


    def _fit_all(self) -> None:
        """Run OLS for each stock against the market."""
        X = sm.add_constant(self._market_returns)
        for ticker in self._stock_returns.columns:
            self._models[ticker] = sm.OLS(
                endog=self._stock_returns[ticker],
                exog=X,
            ).fit()


    def get_betas(self) -> dict[str, float]:
        """Raw OLS betas for each stock."""
        return {t: model.params.iloc[1] for t, model in self._models.items()}


    def get_alphas(self) -> dict[str, float]:
        """OLS intercepts (alphas) for each stock."""
        return {t: model.params.iloc[0] for t, model in self._models.items()}


    def get_residuals(self) -> dict[str, pd.Series]:
        """OLS residuals (error terms) for each stock."""
        return {t: model.resid for t, model in self._models.items()}


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
        return {t: float(np.var(model.resid)) for t, model in self._models.items()}

    def get_total_risks(self) -> dict[str, float]:
        """Total risk: systematic + firm-specific."""
        systematic = self.get_systematic_risks()
        firm_specific = self.get_firm_specific_risks()
        return {t: systematic[t] + firm_specific[t] for t in systematic}