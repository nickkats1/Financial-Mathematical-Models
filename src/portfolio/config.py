"""Default tickers, date range, and confidence levels for the portfolio models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetClass:
    label: str
    tickers: list[str]


ASSET_CLASSES: dict[str, AssetClass] = {
    "stocks": AssetClass(
        "Stocks",
        [
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
        ],
    ),
    "etfs": AssetClass(
        "ETFs",
        [
            "SPY", "QQQ", "IWM", "DIA", "VTI",
            "VOO", "VEA", "VWO", "EFA", "EEM",
            "AGG", "BND", "TLT", "IEF", "SHY",
            "LQD", "HYG", "TIP",
            "GLD", "SLV", "USO", "UNG",
            "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB",
            "ARKK", "FNGU",
        ],
    ),
    "bonds": AssetClass(
        "Treasury bonds",
        ["^IRX", "^FVX", "^TNX", "^TYX"],
    ),
    "crypto": AssetClass(
        "Crypto",
        [
            "BTC-USD", "ETH-USD", "USDT-USD", "USDC-USD",
            "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
            "DOGE-USD", "TRX-USD", "LINK-USD", "ZEC-USD",
        ],
    ),
}

sp500_ticker: str = "^GSPC"

stock_tickers: list[str] = ASSET_CLASSES["stocks"].tickers
etf_tickers: list[str] = ASSET_CLASSES["etfs"].tickers
bond_tickers: list[str] = ASSET_CLASSES["bonds"].tickers
crypto_tickers: list[str] = ASSET_CLASSES["crypto"].tickers

all_tickers: list[str] = [
    ticker for asset_class in ASSET_CLASSES.values() for ticker in asset_class.tickers
] + [sp500_ticker]

# Notebook-level defaults only; the Flask app always supplies an explicit range.
start_date: str = "2022-12-01"
end_date: str = "2026-04-30"

confidence_level_95: float = 0.95
