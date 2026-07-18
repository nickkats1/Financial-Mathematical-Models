"""Sharpe's Single Index Model: R_i = alpha_i + beta_i * R_m + epsilon_i."""

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


class SingleIndexModel:
    """OLS of each asset's returns on a market proxy; exposes alpha, beta, and risk splits."""

    def __init__(self) -> None:
        self.results: dict[str, Any] = {}
        self._market_returns: pd.Series | None = None

    def _require_fit(self) -> None:
        if not self.results or self._market_returns is None:
            raise RuntimeError("call get_models(...) before accessing model outputs")

    def get_models(
        self,
        tickers: list[str] | str,
        market_ticker: str,
        returns: pd.DataFrame,
    ) -> dict[str, Any]:
        if isinstance(tickers, str):
            tickers = [tickers]
        if market_ticker not in returns.columns:
            raise KeyError(f"market_ticker {market_ticker!r} not in returns columns")
        missing = [t for t in tickers if t not in returns.columns]
        if missing:
            raise KeyError(f"tickers missing from returns: {missing}")

        self._market_returns = returns[market_ticker]
        market_with_const = sm.add_constant(self._market_returns)
        models: dict[str, Any] = {
            ticker: sm.OLS(returns[ticker], market_with_const).fit()
            for ticker in tickers
        }
        self.results = models
        return models

    def get_betas(self) -> dict[str, float]:
        self._require_fit()
        return {ticker: model.params.iloc[1] for ticker, model in self.results.items()}

    def get_alphas(self) -> dict[str, float]:
        self._require_fit()
        return {ticker: model.params.iloc[0] for ticker, model in self.results.items()}

    def get_residuals(self) -> dict[str, pd.Series]:
        self._require_fit()
        return {ticker: model.resid for ticker, model in self.results.items()}

    def get_market_variance(self) -> float:
        self._require_fit()
        assert self._market_returns is not None
        return float(self._market_returns.var(ddof=1))

    def get_systematic_risks(self) -> dict[str, float]:
        betas = self.get_betas()
        market_var = self.get_market_variance()
        return {ticker: beta ** 2 * market_var for ticker, beta in betas.items()}

    def get_firm_specific_risks(self) -> dict[str, float]:
        self._require_fit()
        return {
            ticker: float(np.var(model.resid, ddof=1))
            for ticker, model in self.results.items()
        }

    def get_total_risks(self) -> dict[str, float]:
        systematic = self.get_systematic_risks()
        firm_specific = self.get_firm_specific_risks()
        return {ticker: systematic[ticker] + firm_specific[ticker] for ticker in systematic}
