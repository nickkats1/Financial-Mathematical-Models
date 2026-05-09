"""Sharpe's Single Index Model.

Decomposes each asset's risk into a systematic component (driven by the
market) and a firm-specific component (idiosyncratic residual variance).

The model for asset ``i`` is::

    R_i = alpha_i + beta_i * R_m + epsilon_i

where ``R_m`` is the return of a market proxy (typically the S&P 500) and
``epsilon_i`` is a zero-mean firm-specific shock.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

RegressionResult = sm.regression.linear_model.RegressionResultsWrapper


class SingleIndexModel:
    """Fit and inspect Single Index Model regressions for a set of assets.

    Usage:
        >>> sim = SingleIndexModel()
        >>> sim.get_models(tickers, market_ticker, returns)
        >>> betas = sim.get_betas()
        >>> systematic = sim.get_systematic_risks()
    """

    def __init__(self) -> None:
        self.results: Dict[str, RegressionResult] = {}
        self._market_returns: Optional[pd.Series] = None

    def _require_fit(self) -> None:
        """Raise ``RuntimeError`` if :meth:`get_models` has not been called yet."""
        if not self.results or self._market_returns is None:
            raise RuntimeError("call get_models(...) before accessing model outputs")

    def get_models(
        self,
        tickers: Union[List[str], str],
        market_ticker: str,
        returns: pd.DataFrame,
    ) -> Dict[str, RegressionResult]:
        """Fit OLS of each ticker's returns on the market returns.

        Args:
            tickers: A single ticker or a list of tickers to regress.
            market_ticker: The column name in ``returns`` to use as the market
                proxy (e.g. ``"^GSPC"``).
            returns: A DataFrame of returns indexed by date, with one column
                per ticker.

        Returns:
            A dict mapping each ticker to its fitted statsmodels
            ``RegressionResultsWrapper``. The result is also cached on
            ``self.results``.

        Raises:
            KeyError: If ``market_ticker`` or any of ``tickers`` is missing
                from ``returns.columns``.
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        if market_ticker not in returns.columns:
            raise KeyError(f"market_ticker {market_ticker!r} not in returns columns")
        missing = [t for t in tickers if t not in returns.columns]
        if missing:
            raise KeyError(f"tickers missing from returns: {missing}")

        self._market_returns = returns[market_ticker]
        market_with_const = sm.add_constant(self._market_returns)
        models: Dict[str, RegressionResult] = {
            ticker: sm.OLS(returns[ticker], market_with_const).fit()
            for ticker in tickers
        }
        self.results = models
        return models

    def get_betas(self) -> Dict[str, float]:
        """Return the fitted beta (market sensitivity) for each ticker."""
        self._require_fit()
        return {ticker: model.params.iloc[1] for ticker, model in self.results.items()}

    def get_alphas(self) -> Dict[str, float]:
        """Return the fitted alpha (intercept) for each ticker."""
        self._require_fit()
        return {ticker: model.params.iloc[0] for ticker, model in self.results.items()}

    def get_residuals(self) -> Dict[str, pd.Series]:
        """Return the regression residuals (idiosyncratic shocks) for each ticker."""
        self._require_fit()
        return {ticker: model.resid for ticker, model in self.results.items()}

    def get_market_variance(self) -> float:
        """Return the sample variance (``ddof=1``) of the market return series."""
        self._require_fit()
        assert self._market_returns is not None
        return float(self._market_returns.var(ddof=1))

    def get_systematic_risks(self) -> Dict[str, float]:
        """Return the systematic risk (``beta**2 * sigma_m**2``) for each ticker."""
        betas = self.get_betas()
        market_var = self.get_market_variance()
        return {ticker: beta ** 2 * market_var for ticker, beta in betas.items()}

    def get_firm_specific_risks(self) -> Dict[str, float]:
        """Return the firm-specific risk (sample variance of residuals) per ticker."""
        self._require_fit()
        return {
            ticker: float(np.var(model.resid, ddof=1))
            for ticker, model in self.results.items()
        }

    def get_total_risks(self) -> Dict[str, float]:
        """Return total risk (systematic + firm-specific) for each ticker."""
        systematic = self.get_systematic_risks()
        firm_specific = self.get_firm_specific_risks()
        return {ticker: systematic[ticker] + firm_specific[ticker] for ticker in systematic}
