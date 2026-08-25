"""Sharpe's Single Index Model: R_i = alpha_i + beta_i * R_m + epsilon_i."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS


@dataclass(frozen=True, slots=True)
class AssetFit:
    """One asset's fit against the market proxy."""

    alpha: float
    beta: float
    firm_specific: float
    rsquared: float
    resid: pd.Series


class SingleIndexModel:
    """OLS of each asset's returns on a market proxy; exposes alpha, beta, and risk splits.

    ``results`` maps each ticker to an :class:`AssetFit`. These are plain records,
    not regression objects: standard errors, p-values and confidence intervals are
    not available, because nothing in this project consumes them.

    Variances (market, systematic, firm-specific, total) are annualised by
    :data:`~portfolio.config.TRADING_DAYS`, matching :mod:`portfolio.models.mpt`
    and :mod:`portfolio.models.utility`. Alpha stays a daily intercept.
    """

    def __init__(self) -> None:
        self.results: dict[str, AssetFit] = {}
        self._market_variance = 0.0

    def _require_fit(self) -> None:
        if not self.results:
            raise RuntimeError("call get_models(...) before accessing model outputs")

    def get_models(
        self,
        tickers: list[str] | str,
        market_ticker: str,
        returns: pd.DataFrame,
    ) -> dict[str, AssetFit]:
        """Fit every ticker against ``market_ticker`` and cache the estimates."""
        if isinstance(tickers, str):
            tickers = [tickers]
        if market_ticker not in returns.columns:
            raise KeyError(f"market_ticker {market_ticker!r} not in returns columns")
        missing = [t for t in tickers if t not in returns.columns]
        if missing:
            raise KeyError(f"tickers missing from returns: {missing}")

        market_returns = returns[market_ticker]
        self._fit(tickers, market_returns, returns)
        return self.results

    def _fit(
        self,
        tickers: list[str],
        market_returns: pd.Series,
        returns: pd.DataFrame,
    ) -> None:
        # Every per-asset regression shares the design matrix [1, R_m], so the whole
        # panel collapses to beta = cov(R_i, R_m) / var(R_m) — one pass over a matrix
        # instead of N separate fits.
        assets = returns[tickers].to_numpy(dtype=float)
        market = market_returns.to_numpy(dtype=float)
        market_centred = market - market.mean()
        assets_centred = assets - assets.mean(axis=0)

        betas = (market_centred @ assets_centred) / (market_centred @ market_centred)
        alphas = assets.mean(axis=0) - betas * market.mean()
        residuals = assets_centred - np.outer(market_centred, betas)
        firm_specific = residuals.var(axis=0, ddof=1) * TRADING_DAYS
        total_ss = (assets_centred ** 2).sum(axis=0)
        # A constant asset has zero total sum of squares. Report NaN, as the previous
        # statsmodels implementation did, rather than warning on the request path.
        with np.errstate(divide="ignore", invalid="ignore"):
            r_squared = 1.0 - (residuals ** 2).sum(axis=0) / total_ss

        self.results = {
            ticker: AssetFit(
                alpha=float(alphas[i]),
                beta=float(betas[i]),
                firm_specific=float(firm_specific[i]),
                rsquared=float(r_squared[i]),
                resid=pd.Series(residuals[:, i], index=returns.index, name=ticker),
            )
            for i, ticker in enumerate(tickers)
        }
        self._market_variance = float(market_returns.var(ddof=1) * TRADING_DAYS)

    def get_betas(self) -> dict[str, float]:
        self._require_fit()
        return {ticker: fit.beta for ticker, fit in self.results.items()}

    def get_alphas(self) -> dict[str, float]:
        self._require_fit()
        return {ticker: fit.alpha for ticker, fit in self.results.items()}

    def get_r_squared(self) -> dict[str, float]:
        self._require_fit()
        return {ticker: fit.rsquared for ticker, fit in self.results.items()}

    def get_residuals(self) -> dict[str, pd.Series]:
        self._require_fit()
        return {ticker: fit.resid for ticker, fit in self.results.items()}

    def get_market_variance(self) -> float:
        self._require_fit()
        return self._market_variance

    def get_systematic_risks(self) -> dict[str, float]:
        self._require_fit()
        return {
            ticker: fit.beta ** 2 * self._market_variance
            for ticker, fit in self.results.items()
        }

    def get_firm_specific_risks(self) -> dict[str, float]:
        self._require_fit()
        return {ticker: fit.firm_specific for ticker, fit in self.results.items()}

    def get_total_risks(self) -> dict[str, float]:
        return {
            ticker: systematic + self.results[ticker].firm_specific
            for ticker, systematic in self.get_systematic_risks().items()
        }
