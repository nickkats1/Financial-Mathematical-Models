"""Service layer that wires the form input to the portfolio models."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import TypedDict

import pandas as pd

from portfolio import config
from portfolio.data import DataIngestion
from portfolio.models import (
    SingleIndexModel,
    get_cvar,
    get_utility,
    get_var,
    max_utility,
    portfolio_metrics,
)

MAX_TICKER_INPUT_LENGTH = 1000
MIN_TICKERS = 2
MAX_TICKERS = 200
DEFAULT_MARKET_TICKER = "^GSPC"

# Yahoo Finance symbols: letters/digits plus . - = ^ (e.g. BRK.B, BTC-USD, ^GSPC, CL=F).
_TICKER_RE = re.compile(r"^[A-Z0-9.^=-]{1,20}$")

ASSET_CLASS_TICKERS: dict[str, list[str]] = {
    key: ac.tickers for key, ac in config.ASSET_CLASSES.items()
}
ASSET_CLASS_LABELS: dict[str, str] = {
    key: ac.label for key, ac in config.ASSET_CLASSES.items()
}


@dataclass(frozen=True)
class AnalysisRequest:
    """Validated, immutable inputs from the analysis form."""

    tickers: list[str]
    start_date: str
    end_date: str
    risk_free_rate: float
    risk_aversion: float
    market_ticker: str
    asset_classes: list[str]


@dataclass(frozen=True)
class AnalysisResult:
    """Aggregate analytics computed for an :class:`AnalysisRequest`."""

    tickers: list[str]
    start_date: str
    end_date: str
    risk_free_rate: float
    risk_aversion: float
    market_ticker: str
    asset_classes: list[str]
    dropped_tickers: list[str]
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


def parse_tickers(raw: str) -> list[str]:
    """Parse comma/whitespace-separated tickers into a de-duplicated upper-case list.

    May return empty — parse_form enforces MIN_TICKERS after asset-class presets merge in.
    """
    if len(raw) > MAX_TICKER_INPUT_LENGTH:
        raise ValueError(
            f"Ticker input too long (max {MAX_TICKER_INPUT_LENGTH} characters)."
        )

    tokens = [token.strip().upper() for token in raw.replace(",", " ").split()]
    seen: set = set()
    tickers: list[str] = []
    for token in tokens:
        if not token or token in seen:
            continue
        if not _TICKER_RE.match(token):
            raise ValueError(f"Invalid ticker symbol: {token!r}.")
        seen.add(token)
        tickers.append(token)

    if len(tickers) > MAX_TICKERS:
        raise ValueError(f"Please supply at most {MAX_TICKERS} tickers.")
    return tickers


def _parse_asset_classes(raw: list[str]) -> list[str]:
    """Validate asset-class keys, returning them de-duplicated in canonical order."""
    seen: set = set()
    cleaned: list[str] = []
    for value in raw:
        key = (value or "").strip().lower()
        if not key or key in seen:
            continue
        if key not in ASSET_CLASS_TICKERS:
            raise ValueError(f"Unknown asset class: {value!r}.")
        seen.add(key)
        cleaned.append(key)
    return [k for k in ASSET_CLASS_TICKERS if k in seen]


def _merge_universe(
    typed: list[str],
    classes: list[str],
) -> list[str]:
    """Merge user-typed tickers with the selected asset-class presets."""
    seen: set = set()
    merged: list[str] = []
    for ticker in typed:
        if ticker not in seen:
            seen.add(ticker)
            merged.append(ticker)
    for key in classes:
        for ticker in ASSET_CLASS_TICKERS[key]:
            upper = ticker.upper()
            if upper not in seen:
                seen.add(upper)
                merged.append(upper)
    return merged


def _parse_market_ticker(raw: str) -> str:
    """Normalise the market-proxy ticker, falling back to the default."""
    cleaned = (raw or "").strip().upper()
    if not cleaned:
        return DEFAULT_MARKET_TICKER
    if not _TICKER_RE.match(cleaned):
        raise ValueError("Market ticker must be a single valid symbol.")
    return cleaned


def _parse_iso_date(value: str, field: str) -> date:
    """Parse an ISO YYYY-MM-DD date or raise ValueError with a friendly message."""
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be in YYYY-MM-DD format.") from exc


def _parse_float(value: str, field: str) -> float:
    """Parse a float or raise ValueError with a friendly message."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number.") from exc


def parse_form(form: Mapping[str, str]) -> AnalysisRequest:
    """Validate raw form fields into an AnalysisRequest; raises ValueError on bad input."""
    typed_tickers = parse_tickers(form.get("tickers", ""))

    raw_classes = (
        form.getlist("asset_classes")
        if hasattr(form, "getlist")
        else form.get("asset_classes", "").split(",") if form.get("asset_classes") else []
    )
    asset_classes = _parse_asset_classes(raw_classes)

    universe = _merge_universe(typed_tickers, asset_classes)
    if len(universe) < MIN_TICKERS:
        raise ValueError(
            f"Please supply at least {MIN_TICKERS} tickers — type symbols, "
            "tick one or more asset classes, or both."
        )
    if len(universe) > MAX_TICKERS:
        raise ValueError(
            f"Combined universe is too large (max {MAX_TICKERS} tickers)."
        )

    start = _parse_iso_date(form.get("start_date", ""), "Start date")
    end = _parse_iso_date(form.get("end_date", ""), "End date")
    if start >= end:
        raise ValueError("Start date must be earlier than end date.")

    risk_free_rate = _parse_float(form.get("risk_free_rate", ""), "Risk-free rate")
    if not 0.0 <= risk_free_rate <= 1.0:
        raise ValueError("Risk-free rate must be between 0 and 1.")

    risk_aversion = _parse_float(form.get("risk_aversion", ""), "Risk aversion")
    if risk_aversion <= 0:
        raise ValueError(
            "Risk aversion must be positive (max-utility allocation requires A > 0)."
        )

    market_ticker = _parse_market_ticker(form.get("market_ticker", ""))

    return AnalysisRequest(
        tickers=universe,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        risk_free_rate=risk_free_rate,
        risk_aversion=risk_aversion,
        market_ticker=market_ticker,
        asset_classes=asset_classes,
    )


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

    returns = DataIngestion.compute_returns(prices)
    if market_ticker not in returns.columns or returns.empty:
        return empty

    fit_tickers = [t for t in asset_tickers if t in returns.columns]
    if not fit_tickers:
        return empty

    sim = SingleIndexModel()
    sim.get_models(fit_tickers, market_ticker, returns)

    return {
        "alphas": {t: float(v) for t, v in sim.get_alphas().items()},
        "betas": {t: float(v) for t, v in sim.get_betas().items()},
        "systematic": {t: float(v) for t, v in sim.get_systematic_risks().items()},
        "firm_specific": {t: float(v) for t, v in sim.get_firm_specific_risks().items()},
        "total": {t: float(v) for t, v in sim.get_total_risks().items()},
        "r_squared": {
            t: float(model.rsquared) for t, model in sim.results.items()
        },
        "market_variance": sim.get_market_variance(),
    }


def run_analysis(request: AnalysisRequest) -> AnalysisResult:
    """Fetch prices and compute MPT, VaR/CVaR, utility, and Single Index Model figures."""
    ingestion = DataIngestion(
        start_date=request.start_date,
        end_date=request.end_date,
    )

    fetch_tickers = list(request.tickers)
    if request.market_ticker not in fetch_tickers:
        fetch_tickers.append(request.market_ticker)
    all_prices = ingestion.fetch_prices(fetch_tickers)

    asset_tickers = [t for t in request.tickers if t in all_prices.columns]
    dropped = [t for t in request.tickers if t not in all_prices.columns]
    if len(asset_tickers) < MIN_TICKERS:
        raise ValueError(
            "Not enough tickers returned price data for the chosen window. "
            f"Dropped by yfinance: {', '.join(dropped) or '(none)'}. "
            "Try widening the date range or using more liquid symbols."
        )
    asset_prices = all_prices[asset_tickers]
    market_available = request.market_ticker in all_prices.columns

    try:
        mpt = portfolio_metrics(asset_prices, risk_free_rate=request.risk_free_rate)
    except Exception as exc:  # noqa: BLE001 — pypfopt raises a wide range of types
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
    sim = _fit_single_index_model(all_prices, asset_tickers, request.market_ticker)

    return AnalysisResult(
        tickers=list(asset_prices.columns),
        start_date=request.start_date,
        end_date=request.end_date,
        risk_free_rate=request.risk_free_rate,
        risk_aversion=request.risk_aversion,
        market_ticker=request.market_ticker,
        asset_classes=list(request.asset_classes),
        dropped_tickers=dropped,
        market_proxy_available=market_available,
        expected_annual_return=mpt["expected_annual_return"],
        annual_volatility=mpt["annual_volatility"],
        sharpe_ratio=mpt["sharpe_ratio"],
        weights=dict(mpt["weights"]),
        var_90=get_var(asset_prices, confidence=0.90),
        cvar_90=get_cvar(asset_prices, confidence=0.90),
        var_95=get_var(asset_prices, confidence=0.95),
        cvar_95=get_cvar(asset_prices, confidence=0.95),
        var_99=get_var(asset_prices, confidence=0.99),
        cvar_99=get_cvar(asset_prices, confidence=0.99),
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
