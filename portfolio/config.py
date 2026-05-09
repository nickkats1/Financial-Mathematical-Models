"""Project configuration: tickers, date range, and default confidence levels.

All values defined here are module-level defaults consumed by the
``portfolio.data`` and ``portfolio.models`` subpackages. They can be
imported and overridden in notebooks or downstream scripts.

Ticker-list policy
------------------
Every symbol in this file is fetched through :func:`yfinance.download`.
Unknown or illiquid symbols are silently dropped by yfinance and bloat
the optimisation universe, so the lists below are kept to recognisable,
high-liquidity instruments. Add to them as needed for your own work.
"""

from typing import List


# --- Equity tickers -----------------------------------------------------------

stock_tickers: List[str] = [
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
]


# --- ETF tickers --------------------------------------------------------------

etf_tickers: List[str] = [
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    "VOO", "VEA", "VWO", "EFA", "EEM",
    "AGG", "BND", "TLT", "IEF", "SHY",
    "LQD", "HYG", "TIP",
    "GLD", "SLV", "USO", "UNG",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB",
    "ARKK", "FNGU",
]


# --- Cryptocurrency tickers ---------------------------------------------------

crypto_tickers: List[str] = [
    "BTC-USD", "ETH-USD", "USDT-USD", "USDC-USD",
    "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "TRX-USD", "LINK-USD", "ZEC-USD",
]


# --- Treasury bond tickers ----------------------------------------------------

bond_tickers: List[str] = [
    "^IRX", "^FVX", "^TNX", "^TYX",
]


# --- Market index ticker (used as the market proxy in the Single Index Model)

sp500_ticker: str = "^GSPC"


# --- Combined universe --------------------------------------------------------

all_tickers: List[str] = [
    *stock_tickers,
    *etf_tickers,
    *bond_tickers,
    *crypto_tickers,
    sp500_ticker,
]


# --- Date range used for all yfinance fetches ---------------------------------
# These are notebook-level defaults only. The Flask app always supplies an
# explicit user-picked range and never falls through to these values.

start_date: str = "2022-12-01"
end_date: str = "2026-04-30"


# --- Default confidence levels for VaR / CVaR ---------------------------------

confidence_level_95: float = 0.95
confidence_level_99: float = 0.99
