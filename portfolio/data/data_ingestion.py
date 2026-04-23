import pandas as pd
import yfinance as yf
from portfolio import config

from typing import List

class DataIngestion:
    """Fetch interface to yfinance for price and return data."""

    def __init__(self) -> None:
        self.config = config

    def _fetch(self, tickers: List[str] | str) -> pd.DataFrame:
        """Fetch prices from yfinance"""
        tickers_list = sorted(tickers) if isinstance(tickers, list) else [tickers]

        prices = yf.download(
            tickers=tickers_list,
            start=config.start_date,
            end=config.end_date,
        )["Close"]

        if prices is None or prices.empty:
            raise ValueError(
                f"No price data returned for tickers={tickers_list} "
                f"between {config.start_date} and {config.end_date}"
            )

        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers_list[0])

        return prices.dropna().drop_duplicates()

    def get_stock_prices(self) -> pd.DataFrame:
        """Fetch stock prices from yfinance"""
        return self._fetch(config.stock_tickers)

    def get_etf_prices(self) -> pd.DataFrame:
        """Fetch ETF prices from yfinance"""
        return self._fetch(config.etf_tickers)

    def get_bond_prices(self) -> pd.DataFrame:
        """Fetch bond prices from yfinance"""
        return self._fetch(config.bond_tickers)

    def get_crypto_prices(self) -> pd.DataFrame:
        """Fetch crypto prices from yfinance"""
        return self._fetch(config.crypto_tickers)

    def get_sp500_prices(self) -> pd.DataFrame:
        """Get SP&500 prices from yfinance"""
        return self._fetch(config.sp500_ticker)

    def get_all_prices(self) -> pd.DataFrame:
        """Get all prices from yfinance"""
        return self._fetch(config.all_tickers)

    @staticmethod
    def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Computer returns from yfinance"""
        return prices.pct_change().dropna()