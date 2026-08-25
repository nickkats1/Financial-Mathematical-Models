"""Service layer that wires a validated request to the portfolio models."""

from dataclasses import dataclass
from typing import TypedDict

import pandas as pd

from app.config import DEFAULT_APP_CONFIG, AppConfig
from app.forms import AnalysisRequest
from portfolio.config import DEFAULT_CONFIG
from portfolio.data import DataIngestion, compute_returns
from portfolio.models import (
    SingleIndexModel,
    get_risk_metrics,
    get_utility,
    max_utility,
    portfolio_metrics,
)


@dataclass(frozen=True)
class AnalysisResult:
    """Aggregate analytics computed for an :class:`AnalysisRequest`."""

    tickers: tuple[str, ...]
    start_date: str
    end_date: str
    risk_free_rate: float
    risk_aversion: float
    market_ticker: str
    asset_classes: tuple[str, ...]
    dropped_tickers: tuple[str, ...]
    market_proxy_available: bool
    expected_annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    weights: dict[str, float]
    var_90: float
    cvar_90: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    utility: dict[str, float]
    max_utility: dict[str, float]
    market_variance: float
    alphas: dict[str, float]
    betas: dict[str, float]
    systematic_risks: dict[str, float]
    firm_specific_risks: dict[str, float]
    total_risks: dict[str, float]
    r_squared: dict[str, float]


class SingleIndexFit(TypedDict):
    """Per-asset Single Index Model decomposition returned by the service."""

    alphas: dict[str, float]
    betas: dict[str, float]
    systematic: dict[str, float]
    firm_specific: dict[str, float]
    total: dict[str, float]
    r_squared: dict[str, float]
    market_variance: float


def _fit_single_index_model(
    prices: pd.DataFrame,
    asset_tickers: list[str],
    market_ticker: str,
) -> SingleIndexFit:
    """Fit the Single Index Model; returns an empty result if the market proxy has no data."""
    empty: SingleIndexFit = {
        "alphas": {},
        "betas": {},
        "systematic": {},
        "firm_specific": {},
        "total": {},
        "r_squared": {},
        "market_variance": 0.0,
    }
    if market_ticker not in prices.columns:
        return empty

    returns = compute_returns(prices)
    if returns.empty:
        return empty

    fit_tickers = [t for t in asset_tickers if t in returns.columns]
    if not fit_tickers:
        return empty

    sim = SingleIndexModel()
    sim.get_models(fit_tickers, market_ticker, returns)

    return {
        "alphas": sim.get_alphas(),
        "betas": sim.get_betas(),
        "systematic": sim.get_systematic_risks(),
        "firm_specific": sim.get_firm_specific_risks(),
        "total": sim.get_total_risks(),
        "r_squared": sim.get_r_squared(),
        "market_variance": sim.get_market_variance(),
    }


def run_analysis(
    request: AnalysisRequest,
    config: AppConfig = DEFAULT_APP_CONFIG,
) -> AnalysisResult:
    """Fetch prices and compute MPT, VaR/CVaR, utility, and Single Index Model figures."""
    ingestion = DataIngestion(
        start_date=request.start_date,
        end_date=request.end_date,
    )

    fetch_tickers = list(dict.fromkeys([*request.tickers, request.market_ticker]))
    all_prices = ingestion.fetch_prices(fetch_tickers)

    asset_tickers = [t for t in request.tickers if t in all_prices.columns]
    dropped = [t for t in request.tickers if t not in all_prices.columns]
    if len(asset_tickers) < config.min_tickers:
        raise ValueError(
            "Not enough tickers returned price data for the chosen window. "
            f"Dropped by yfinance: {', '.join(dropped) or '(none)'}. "
            "Try widening the date range or using more liquid symbols."
        )
    asset_prices = all_prices[asset_tickers]
    market_available = request.market_ticker in all_prices.columns

    try:
        mpt = portfolio_metrics(asset_prices, risk_free_rate=request.risk_free_rate)
    except Exception as exc:  # pypfopt raises a wide range of types
        raise ValueError(
            "Could not solve the max-Sharpe portfolio for this universe — "
            "the covariance matrix is likely singular or poorly conditioned. "
            "Try a longer date window, fewer tickers, or a more diverse "
            f"universe ({exc.__class__.__name__})."
        ) from exc

    utility = get_utility(asset_prices, risk_aversion=request.risk_aversion)
    utility_max = max_utility(
        asset_prices,
        risk_aversion=request.risk_aversion,
        risk_free_rate=request.risk_free_rate,
    )
    risk = get_risk_metrics(asset_prices)
    # AnalysisResult's flat var_/cvar_ fields mirror these three levels, in order.
    low, mid, high = DEFAULT_CONFIG.confidence_levels
    sim = _fit_single_index_model(all_prices, asset_tickers, request.market_ticker)

    return AnalysisResult(
        tickers=tuple(asset_prices.columns),
        start_date=request.start_date,
        end_date=request.end_date,
        risk_free_rate=request.risk_free_rate,
        risk_aversion=request.risk_aversion,
        market_ticker=request.market_ticker,
        asset_classes=tuple(request.asset_classes),
        dropped_tickers=tuple(dropped),
        market_proxy_available=market_available,
        expected_annual_return=mpt["expected_annual_return"],
        annual_volatility=mpt["annual_volatility"],
        sharpe_ratio=mpt["sharpe_ratio"],
        weights=dict(mpt["weights"]),
        var_90=risk[low][0],
        cvar_90=risk[low][1],
        var_95=risk[mid][0],
        cvar_95=risk[mid][1],
        var_99=risk[high][0],
        cvar_99=risk[high][1],
        utility=_series_to_dict(utility),
        max_utility=_series_to_dict(utility_max),
        market_variance=sim["market_variance"],
        alphas=sim["alphas"],
        betas=sim["betas"],
        systematic_risks=sim["systematic"],
        firm_specific_risks=sim["firm_specific"],
        total_risks=sim["total"],
        r_squared=sim["r_squared"],
    )


def _series_to_dict(series: pd.Series) -> dict[str, float]:
    """Convert a numeric Series to a plain ``{ticker: value}`` dict."""
    return {str(idx): float(value) for idx, value in series.items()}
