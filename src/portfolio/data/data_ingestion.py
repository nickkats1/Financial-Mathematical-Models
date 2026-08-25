"""Price and return data ingestion via yfinance, with retries and a TTL cache."""

from threading import Lock

import pandas as pd
import yfinance as yf
from cachetools import TTLCache
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import DEFAULT_CONFIG, get_asset_class

# Mirrors price_cache_* in app/config.py, which overrides these at startup; the
# library cannot import app.config without inverting the dependency direction.
_CACHE_TTL_SECONDS = 300
_CACHE_MAX_ENTRIES = 1024

_price_cache: TTLCache = TTLCache(maxsize=_CACHE_MAX_ENTRIES, ttl=_CACHE_TTL_SECONDS)
_cache_lock = Lock()


def configure_price_cache(ttl_seconds: int, max_entries: int) -> None:
    """Rebuild the module-level price cache; called once at application startup."""
    global _price_cache
    with _cache_lock:
        _price_cache = TTLCache(maxsize=max_entries, ttl=ttl_seconds)


@retry(
    retry=retry_if_exception_type(OSError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
    reraise=True,
)
def _yf_download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    # Only transport failures are retried. yfinance swallows its own per-ticker
    # errors and hands back an empty column, so anything that does escape is
    # deterministic and would fail again on a second attempt.
    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )


def _downloaded_close(tickers: list[str], start: str, end: str) -> dict[str, pd.Series]:
    """Download closing prices and split them into one Series per ticker."""
    raw = _yf_download(tickers, start, end)
    raw_close = raw["Close"]
    close: pd.DataFrame = (
        raw_close.to_frame(name=tickers[0])
        if isinstance(raw_close, pd.Series)
        else raw_close
    )

    # A ticker yfinance failed to fetch comes back as an all-NaN column rather than
    # an error, so dropping empty columns here is what keeps a transient outage out
    # of the cache for the rest of the TTL window.
    return {
        str(ticker): series
        for ticker in close.columns
        if not (series := close[ticker].dropna()).empty
    }


def _cached_close(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    with _cache_lock:
        have = {
            ticker: series
            for ticker in tickers
            if (series := _price_cache.get((ticker, start, end))) is not None
        }

    missing = [ticker for ticker in tickers if ticker not in have]
    if missing:
        fetched = _downloaded_close(missing, start, end)
        with _cache_lock:
            for ticker, series in fetched.items():
                # setdefault re-checks under the lock: another thread may have
                # populated the entry while we were downloading. The copy keeps the
                # cached Series from pinning the whole downloaded frame alive.
                _price_cache.setdefault((ticker, start, end), series.copy())
        have.update(fetched)

    ordered = [have[ticker] for ticker in tickers if ticker in have]
    if not ordered:
        return pd.DataFrame()
    return pd.concat(ordered, axis=1).copy()


class DataIngestion:
    """Fetch closing prices and compute returns from yfinance."""

    def __init__(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        self.start_date = DEFAULT_CONFIG.start_date if start_date is None else start_date
        self.end_date = DEFAULT_CONFIG.end_date if end_date is None else end_date

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
        except Exception as exc:
            raise ValueError(
                f"Could not fetch prices from yfinance for tickers="
                f"{tickers_list} between {self.start_date} and "
                f"{self.end_date}: {exc.__class__.__name__}"
            ) from exc

        if close.empty:
            raise ValueError(
                f"No price data returned for tickers={tickers_list} "
                f"between {self.start_date} and {self.end_date}"
            )


        return close.dropna()

    def fetch_prices(self, tickers: list[str] | str) -> pd.DataFrame:
        return self._fetch(tickers)

    def get_asset_class_prices(self, name: str) -> pd.DataFrame:
        """Fetch closing prices for a named preset — "stocks", "etfs", "bonds", "crypto"."""
        return self._fetch(list(get_asset_class(name).tickers))

    def get_market_prices(self) -> pd.DataFrame:
        return self._fetch(DEFAULT_CONFIG.market_ticker)

    def get_all_prices(self) -> pd.DataFrame:
        return self._fetch(list(DEFAULT_CONFIG.all_tickers))


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns, with the leading NaN row dropped."""
    return prices.pct_change(fill_method=None).dropna()


def clear_price_cache() -> None:
    """Empty the module-level price cache."""
    with _cache_lock:
        _price_cache.clear()
