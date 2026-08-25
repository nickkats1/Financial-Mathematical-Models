"""Domain configuration for the portfolio models.

Two immutable value objects:

* :class:`AssetClass` — a named preset universe of tickers, registered in
  :data:`ASSET_CLASSES` and looked up with :func:`get_asset_class`.
* :class:`PortfolioConfig` — model-level defaults (market proxy, date window,
  confidence levels, rate and aversion defaults).

Nothing here reads the environment or imports Flask. Deployment-specific knobs
(secret key, rate limits, cache sizing) live in :mod:`app.config`.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from itertools import chain
from types import MappingProxyType

TRADING_DAYS = 252

# Negative A describes a risk-seeking investor; the floor keeps the utility
# figures interpretable rather than letting variance dominate without limit.
MIN_RISK_AVERSION = -5.0


@dataclass(frozen=True, slots=True)
class AssetClass:
    """A named preset universe of tickers."""

    name: str
    label: str
    tickers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Asset class name must not be empty.")
        if not self.tickers:
            raise ValueError(f"Asset class {self.name!r} has no tickers.")
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError(f"Asset class {self.name!r} has duplicate tickers.")


_PRESETS: tuple[AssetClass, ...] = (
    AssetClass(
        name="stocks",
        label="Stocks",
        tickers=(
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "AMD", "INTC", "MU",
            "NFLX", "ORCL", "CRM", "ADBE", "AVGO",
            "JPM", "BAC", "WFC", "GS", "MS",
            "V", "MA", "PYPL", "SOFI", "HOOD",
            "WMT", "TGT", "COST", "MCD", "NKE",
            "KO", "PEP", "PG", "JNJ", "PFE",
            "T", "VZ", "DIS", "F", "GM",
            "BA", "CAT", "GE", "XOM", "CVX",
            "PLTR", "SNAP", "AAL", "CCL", "RKT",
            "SMCI", "MARA",
        ),
    ),
    AssetClass(
        name="etfs",
        label="ETFs",
        tickers=(
            "SPY", "QQQ", "IWM", "DIA", "VTI",
            "VOO", "VEA", "VWO", "EFA", "EEM",
            "AGG", "BND", "TLT", "IEF", "SHY",
            "LQD", "HYG", "TIP",
            "GLD", "SLV", "USO", "UNG",
            "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB",
            "ARKK", "FNGU",
        ),
    ),
    AssetClass(
        name="bonds",
        label="Treasury bonds",
        tickers=("^IRX", "^FVX", "^TNX", "^TYX"),
    ),
    AssetClass(
        name="crypto",
        label="Crypto",
        tickers=(
            "BTC-USD", "ETH-USD", "USDT-USD", "USDC-USD",
            "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
            "DOGE-USD", "TRX-USD", "LINK-USD", "ZEC-USD",
        ),
    ),
)


def _build_registry(presets: tuple[AssetClass, ...]) -> Mapping[str, AssetClass]:
    """Index presets by name, rejecting duplicates."""
    registry: dict[str, AssetClass] = {}
    for preset in presets:
        if preset.name in registry:
            raise ValueError(f"Duplicate asset class name: {preset.name!r}.")
        registry[preset.name] = preset
    return MappingProxyType(registry)


ASSET_CLASSES: Mapping[str, AssetClass] = _build_registry(_PRESETS)


def get_asset_class(name: str) -> AssetClass:
    """Return the preset registered under ``name``, or raise listing what exists."""
    try:
        return ASSET_CLASSES[name]
    except KeyError:
        raise ValueError(
            f"Asset class {name!r} not found. Available: {sorted(ASSET_CLASSES)}"
        ) from None


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    """Model-level defaults. Derive variants with :func:`dataclasses.replace`."""

    market_ticker: str = "^GSPC"
    start_date: str = "2022-12-01"
    end_date: str = "2026-04-30"
    confidence_levels: tuple[float, ...] = (0.90, 0.95, 0.99)
    default_confidence: float = 0.95
    risk_free_rate: float = 0.04
    risk_aversion: float = 3.0
    utility_scaling_factor: float = 0.5

    def __post_init__(self) -> None:
        self.validate_confidence_levels()
        self.validate_dates()
        self.validate_rates()

    def validate_confidence_levels(self) -> None:
        """Every level lies in (0, 1) and the default is one of them."""
        if not self.confidence_levels:
            raise ValueError("confidence_levels must not be empty.")
        for level in self.confidence_levels:
            if not 0.0 < level < 1.0:
                raise ValueError(f"Confidence level must be in (0, 1), got {level}.")
        if self.default_confidence not in self.confidence_levels:
            raise ValueError(
                f"default_confidence {self.default_confidence} must be one of "
                f"confidence_levels {self.confidence_levels}."
            )

    def validate_dates(self) -> None:
        """Both dates parse as ISO YYYY-MM-DD and start precedes end."""
        for field_name in ("start_date", "end_date"):
            value = getattr(self, field_name)
            try:
                date.fromisoformat(value)
            except ValueError:
                raise ValueError(
                    f"{field_name} must be in YYYY-MM-DD format, got {value!r}."
                ) from None
        if date.fromisoformat(self.start_date) >= date.fromisoformat(self.end_date):
            raise ValueError("start_date must be earlier than end_date.")

    def validate_rates(self) -> None:
        """Rate and aversion defaults match the bounds the web form enforces."""
        if not 0.0 <= self.risk_free_rate <= 1.0:
            raise ValueError(
                f"risk_free_rate must be between 0 and 1, got {self.risk_free_rate}."
            )
        if self.risk_aversion == 0 or self.risk_aversion < MIN_RISK_AVERSION:
            raise ValueError(
                f"risk_aversion must be non-zero and at least "
                f"{MIN_RISK_AVERSION}, got {self.risk_aversion}."
            )
        if self.utility_scaling_factor <= 0:
            raise ValueError(
                "utility_scaling_factor must be positive, got "
                f"{self.utility_scaling_factor}."
            )

    @property
    def all_tickers(self) -> tuple[str, ...]:
        """Every preset ticker plus the market proxy, de-duplicated, order preserved."""
        preset_tickers = chain.from_iterable(
            asset_class.tickers for asset_class in ASSET_CLASSES.values()
        )
        return tuple(dict.fromkeys([*preset_tickers, self.market_ticker]))


DEFAULT_CONFIG = PortfolioConfig()
