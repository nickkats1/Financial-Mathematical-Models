"""Form parsing and validation for the analysis request.

Every bound the web form enforces lives here — ticker syntax, universe size,
date ordering, rate and aversion ranges. Raising :class:`ValueError` is the
contract; ``app/routes.py`` turns that into a 400 re-render of the form.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from app.config import DEFAULT_APP_CONFIG, AppConfig
from portfolio.config import ASSET_CLASSES, DEFAULT_CONFIG, MIN_RISK_AVERSION

# Yahoo Finance symbols: letters/digits plus . - = ^ (e.g. BRK.B, BTC-USD, ^GSPC, CL=F).
_TICKER_RE = re.compile(r"^[A-Z0-9.^=-]{1,20}$")

ASSET_CLASS_TICKERS: dict[str, tuple[str, ...]] = {
    name: ac.tickers for name, ac in ASSET_CLASSES.items()
}
ASSET_CLASS_LABELS: dict[str, str] = {
    name: ac.label for name, ac in ASSET_CLASSES.items()
}


@dataclass(frozen=True)
class AnalysisRequest:
    """Validated, immutable inputs from the analysis form."""

    tickers: tuple[str, ...]
    start_date: str
    end_date: str
    risk_free_rate: float
    risk_aversion: float
    market_ticker: str
    asset_classes: tuple[str, ...]


def parse_tickers(raw: str, config: AppConfig = DEFAULT_APP_CONFIG) -> list[str]:
    """Parse comma/whitespace-separated tickers into a de-duplicated upper-case list.

    May return empty — parse_form enforces the minimum after asset-class presets merge in.
    """
    if len(raw) > config.max_ticker_input_length:
        raise ValueError(
            f"Ticker input too long (max {config.max_ticker_input_length} characters)."
        )

    tokens = [token.upper() for token in raw.replace(",", " ").split()]
    seen: set = set()
    tickers: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        if not _TICKER_RE.match(token):
            raise ValueError(f"Invalid ticker symbol: {token!r}.")
        seen.add(token)
        tickers.append(token)

    if len(tickers) > config.max_tickers:
        raise ValueError(f"Please supply at most {config.max_tickers} tickers.")
    return tickers


def _selected_asset_classes(form) -> list:
    """Return the asset-class checkbox values, robust to plain dicts in tests."""
    if hasattr(form, "getlist"):
        return form.getlist("asset_classes")
    return [value for value in form.get("asset_classes", "").split(",") if value]


def _parse_asset_classes(raw: list[str]) -> list[str]:
    """Validate asset-class keys, returning them de-duplicated in canonical order."""
    seen: set[str] = set()
    for value in raw:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        if key not in ASSET_CLASS_TICKERS:
            raise ValueError(f"Unknown asset class: {value!r}.")
        seen.add(key)
    return [k for k in ASSET_CLASS_TICKERS if k in seen]


def _merge_universe(
    typed: list[str],
    classes: list[str],
) -> list[str]:
    """Merge user-typed tickers with the selected asset-class presets."""
    preset = (
        ticker.upper() for key in classes for ticker in ASSET_CLASS_TICKERS[key]
    )
    return list(dict.fromkeys([*typed, *preset]))


def _parse_market_ticker(raw: str) -> str:
    """Normalise the market-proxy ticker, falling back to the default."""
    cleaned = raw.strip().upper()
    if not cleaned:
        return DEFAULT_CONFIG.market_ticker
    if not _TICKER_RE.match(cleaned):
        raise ValueError("Market ticker must be a single valid symbol.")
    return cleaned


def _parse_iso_date(value: str, field: str) -> date:
    """Parse an ISO YYYY-MM-DD date or raise ValueError with a friendly message."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be in YYYY-MM-DD format.") from exc


def _parse_float(value: str, field: str) -> float:
    """Parse a float or raise ValueError with a friendly message."""
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a number.") from exc


def parse_form(
    form: Mapping[str, str],
    config: AppConfig = DEFAULT_APP_CONFIG,
) -> AnalysisRequest:
    """Validate raw form fields into an AnalysisRequest; raises ValueError on bad input."""
    typed_tickers = parse_tickers(form.get("tickers", ""), config)

    asset_classes = _parse_asset_classes(_selected_asset_classes(form))

    universe = _merge_universe(typed_tickers, asset_classes)
    if len(universe) < config.min_tickers:
        raise ValueError(
            f"Please supply at least {config.min_tickers} tickers — type symbols, "
            "tick one or more asset classes, or both."
        )
    if len(universe) > config.max_tickers:
        raise ValueError(
            f"Combined universe is too large (max {config.max_tickers} tickers)."
        )

    start = _parse_iso_date(form.get("start_date", ""), "Start date")
    end = _parse_iso_date(form.get("end_date", ""), "End date")
    if start >= end:
        raise ValueError("Start date must be earlier than end date.")

    risk_free_rate = _parse_float(form.get("risk_free_rate", ""), "Risk-free rate")
    if not 0.0 <= risk_free_rate <= 1.0:
        raise ValueError("Risk-free rate must be between 0 and 1.")

    risk_aversion = _parse_float(form.get("risk_aversion", ""), "Risk aversion")
    if risk_aversion == 0 or risk_aversion < MIN_RISK_AVERSION:
        raise ValueError(
            f"Risk aversion must be non-zero and at least {MIN_RISK_AVERSION}. "
            "Negative A describes a risk-seeking investor; A = 0 leaves the "
            "max-utility allocation undefined."
        )

    market_ticker = _parse_market_ticker(form.get("market_ticker", ""))

    return AnalysisRequest(
        tickers=tuple(universe),
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        risk_free_rate=risk_free_rate,
        risk_aversion=risk_aversion,
        market_ticker=market_ticker,
        asset_classes=tuple(asset_classes),
    )
