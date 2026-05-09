"""Shared pytest fixtures used across the test suite."""

import pandas as pd
import pytest


@pytest.fixture
def dummy_config():
    """Return a small in-memory config dict mirroring :mod:`portfolio.config`."""
    config = {
        "start_date": "2024-01-01",
        "end_date": "2024-01-05",
        "stock_tickers": ["META", "NVDA", "MCD"],
        "etf_tickers": ["BEG", "QQQ"],
        "bond_tickers": ["^TNX", "^IRX"],
        "crypto_tickers": ["BTC-USD", "ETH-USD"],
        "sp500_ticker": "^GSPC",
        "risk_free_rate": 0.001,
        "risk_aversion": 4.0,
    }
    config["all_tickers"] = (
        config["stock_tickers"]
        + config["etf_tickers"]
        + config["bond_tickers"]
        + config["crypto_tickers"]
        + [config["sp500_ticker"]]
    )
    return config


@pytest.fixture
def fake_prices():
    """Return a small synthetic price DataFrame for fast offline tests."""
    dates = pd.bdate_range("2024-01-01", periods=3)
    return pd.DataFrame(
        {
            "META": [100.0, 101.0, 102.0],
            "NVDA": [12.01, 30.01, 2.50],
            "MCD": [100.02, 200.10, 600.50],
            "BEG": [500.03, 500.04, 500.06],
            "^GSPC": [2600.0, 2006.0, 2001.0],
            "QQQ": [10.0, 11.0, 12.0],
            "BTC-USD": [10.0, 12.0, 33.0],
            "ETH-USD": [100.0, 200.0, 300.0],
            "^TNX": [10.02, 11.11, 12.01],
            "^IRX": [100.0, 101.0, 102.0],
        },
        index=dates,
    )
