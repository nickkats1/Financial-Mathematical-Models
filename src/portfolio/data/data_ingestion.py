"""Price and return data ingestion via yfinance, with retries and a TTL cache."""

from threading import RLock

import pandas as pd
import yfinance as yf
from cachetools import TTLCache
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .. import config

_CACHE_TTL_SECONDS = 300
_CACHE_MAX_ENTRIES = 64

_price_cache: TTLCache = TTLCache(maxsize=_CACHE_MAX_ENTRIES, ttl=_CACHE_TTL_SECONDS)
_cache_lock = RLock()


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
    reraise=True,
)
def _yf_download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    # Empty frames are not retried: yfinance returns an empty frame for unknown
    # symbols, and retrying will not change that; the caller validates downstream.
    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )


def _cached_close(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    key = (tickers, start, end)
    with _cache_lock:
        cached_frame = _price_cache.get(key)
    if cached_frame is not None:
        return cached_frame.copy()

    raw = _yf_download(list(tickers), start, end)
    raw_close = raw["Close"]
    close: pd.DataFrame = (
        raw_close.to_frame(name=tickers[0])
        if isinstance(raw_close, pd.Series)
        else raw_close
    )
    close = close.dropna(how="all").drop_duplicates()

    # Empty / failed responses are not cached, so a transient outage does not
    # poison the cache for the rest of the TTL window.
    if not close.empty:
        with _cache_lock:
            _price_cache[key] = close.copy()
    return close


class DataIngestion:
    """Fetch closing prices and compute returns from yfinance."""

    def __init__(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        self.start_date = start_date if start_date is not None else config.start_date
        self.end_date = end_date if end_date is not None else config.end_date

    def _fetch(self, tickers: list[str] | str) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers_list: list[str] = [tickers]
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

    def fetch_prices(self, tickers: list[str] | str) -> pd.DataFrame:
        return self._fetch(tickers)

    def get_stock_prices(self) -> pd.DataFrame:
        return self._fetch(config.stock_tickers)

    def get_etf_prices(self) -> pd.DataFrame:
        return self._fetch(config.etf_tickers)

    def get_bond_prices(self) -> pd.DataFrame:
        return self._fetch(config.bond_tickers)

    def get_crypto_prices(self) -> pd.DataFrame:
        return self._fetch(config.crypto_tickers)

    def get_sp500_prices(self) -> pd.DataFrame:
        return self._fetch(config.sp500_ticker)

    def get_all_prices(self) -> pd.DataFrame:
        return self._fetch(config.all_tickers)

    @staticmethod
    def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
        return prices.pct_change(fill_method=None).dropna()

    @staticmethod
    def clear_price_cache() -> None:
        with _cache_lock:
            _price_cache.clear()
