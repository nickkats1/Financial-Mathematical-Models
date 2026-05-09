"""Price and return data ingestion via the yfinance API.

The :class:`DataIngestion` class wraps :func:`yfinance.download` and provides
typed convenience methods for each asset class declared in
:mod:`portfolio.config`. It also exposes :meth:`fetch_prices` for arbitrary
user-supplied tickers and date ranges (used by the Flask web app).

Resilience:
    - :func:`yfinance.download` is wrapped in a bounded exponential-backoff
      retry via :mod:`tenacity` so transient network / rate-limit blips do
      not turn into HTTP 500s for the user.
    - A small in-memory :class:`cachetools.TTLCache` deduplicates identical
      ``(tickers, start, end)`` requests across users — useful when the
      same form is submitted in quick succession.
"""

from threading import RLock
from typing import List, Optional, Tuple, Union

import pandas as pd
import yfinance as yf
from cachetools import TTLCache
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from portfolio import config

_CACHE_TTL_SECONDS = 300
_CACHE_MAX_ENTRIES = 64

_price_cache: TTLCache = TTLCache(maxsize=_CACHE_MAX_ENTRIES, ttl=_CACHE_TTL_SECONDS)
_cache_lock = RLock()


def _cache_key(tickers: Tuple[str, ...], start: str, end: str) -> Tuple:
    """Stable cache key: tickers are already sorted by the caller."""
    return (tickers, start, end)


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
    reraise=True,
)
def _yf_download(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Call :func:`yfinance.download` with exponential-backoff retries.

    yfinance is unofficial and occasionally raises transient network or
    HTTP errors. Three attempts with 0.5–4 s backoff is enough to ride
    out the typical hiccup. Empty frames are *not* retried — yfinance
    returns an empty frame when a symbol does not exist, and retrying
    will not change the result; the caller validates downstream.
    """
    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )


def _cached_close(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Fetch close prices for ``tickers`` with a small TTL cache.

    Empty / failed responses are not cached, so a transient yfinance
    outage does not poison the cache for the rest of the TTL window.
    """
    key = _cache_key(tickers, start, end)
    with _cache_lock:
        cached_frame = _price_cache.get(key)
    if cached_frame is not None:
        return cached_frame.copy()

    raw = _yf_download(list(tickers), start, end)
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close = close.dropna(how="all").drop_duplicates()

    if not close.empty:
        with _cache_lock:
            _price_cache[key] = close.copy()
    return close


class DataIngestion:
    """Fetch closing prices and compute returns from yfinance.

    Args:
        start_date: Inclusive start date in ``YYYY-MM-DD`` format. If omitted,
            falls back to ``portfolio.config.start_date`` for notebook use;
            the Flask app always supplies an explicit value.
        end_date: Exclusive end date in ``YYYY-MM-DD`` format. If omitted,
            falls back to ``portfolio.config.end_date`` for notebook use;
            the Flask app always supplies an explicit value.
    """

    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        self.start_date = start_date if start_date is not None else config.start_date
        self.end_date = end_date if end_date is not None else config.end_date

    def _fetch(self, tickers: Union[List[str], str]) -> pd.DataFrame:
        """Fetch close prices for one or more tickers.

        Args:
            tickers: A single ticker symbol or a list of ticker symbols.

        Returns:
            A DataFrame of closing prices indexed by date, with one column
            per ticker, sorted alphabetically. Columns whose entire series
            is NaN (typical for symbols yfinance does not recognise) are
            dropped.

        Raises:
            ValueError: If ``tickers`` is empty or yfinance returns no
                usable data for the requested universe and date range.
        """
        if isinstance(tickers, str):
            tickers_list: List[str] = [tickers]
        else:
            tickers_list = sorted({t for t in tickers if t})

        if not tickers_list:
            raise ValueError("tickers must not be empty")

        try:
            close = _cached_close(
                tuple(tickers_list), self.start_date, self.end_date
            )
        except Exception as exc:  # noqa: BLE001 — surface as ValueError
            raise ValueError(
                f"Could not fetch prices from yfinance for tickers="
                f"{tickers_list} between {self.start_date} and "
                f"{self.end_date}: {exc.__class__.__name__}"
            ) from exc

        close = close.dropna(axis=1, how="all").dropna(how="all")
        if close.empty or close.shape[1] == 0:
            raise ValueError(
                f"No price data returned for tickers={tickers_list} "
                f"between {self.start_date} and {self.end_date}"
            )

        return close.dropna().drop_duplicates()

    def fetch_prices(self, tickers: Union[List[str], str]) -> pd.DataFrame:
        """Public wrapper around :meth:`_fetch` for arbitrary ticker lists."""
        return self._fetch(tickers)

    def get_stock_prices(self) -> pd.DataFrame:
        """Return closing prices for every ticker in ``config.stock_tickers``."""
        return self._fetch(config.stock_tickers)

    def get_etf_prices(self) -> pd.DataFrame:
        """Return closing prices for every ticker in ``config.etf_tickers``."""
        return self._fetch(config.etf_tickers)

    def get_bond_prices(self) -> pd.DataFrame:
        """Return closing prices for every ticker in ``config.bond_tickers``."""
        return self._fetch(config.bond_tickers)

    def get_crypto_prices(self) -> pd.DataFrame:
        """Return closing prices for every ticker in ``config.crypto_tickers``."""
        return self._fetch(config.crypto_tickers)

    def get_sp500_prices(self) -> pd.DataFrame:
        """Return closing prices for the S&P 500 index (``config.sp500_ticker``)."""
        return self._fetch(config.sp500_ticker)

    def get_all_prices(self) -> pd.DataFrame:
        """Return closing prices for every ticker in ``config.all_tickers``."""
        return self._fetch(config.all_tickers)

    @staticmethod
    def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Compute simple period-over-period returns from a price DataFrame.

        Args:
            prices: A DataFrame of prices indexed by date.

        Returns:
            A DataFrame of returns of the same shape as ``prices``, with the
            first row dropped because no prior price is available.
        """
        return prices.pct_change(fill_method=None).dropna()

    @staticmethod
    def clear_price_cache() -> None:
        """Drop every cached price frame — used by tests and admin tooling."""
        with _cache_lock:
            _price_cache.clear()
