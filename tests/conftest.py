"""Shared pytest fixtures used across the test suite."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app import create_app
from portfolio.data import clear_price_cache
from portfolio.data import data_ingestion as di_module


@pytest.fixture(autouse=True)
def _clear_price_cache():
    """Isolate the module-level TTL cache between tests."""
    clear_price_cache()
    yield
    clear_price_cache()


@pytest.fixture
def download_result():
    """Build a stand-in for a yfinance download result; the code does ``raw["Close"]``."""

    def _make(frame):
        download = MagicMock()
        download.__getitem__.return_value = frame
        return download

    return _make


@pytest.fixture
def patched_download(download_result):
    """Patch yfinance for the duration of a block so a download yields ``frame``."""

    @contextmanager
    def _patch(frame):
        target = download_result(frame)
        with patch.object(di_module.yf, "download", return_value=target) as mock:
            yield mock

    return _patch


@pytest.fixture
def fake_prices():
    """Return a small synthetic price DataFrame for fast offline tests."""
    dates = pd.bdate_range("2024-01-01", periods=3)
    return pd.DataFrame(
        {
            "AAPL": [150.0, 151.0, 152.0],
            "MSFT": [250.0, 252.0, 251.0],
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


@pytest.fixture
def two_asset_prices():
    """Two-asset frame with enough variance for MPT to solve."""
    rng = np.random.default_rng(seed=42)
    dates = pd.bdate_range("2024-01-01", periods=180)
    return pd.DataFrame(
        {
            "AAPL": 100 * np.cumprod(1 + rng.normal(0.0008, 0.012, 180)),
            "MSFT": 200 * np.cumprod(1 + rng.normal(0.0010, 0.015, 180)),
            "^GSPC": 4000 * np.cumprod(1 + rng.normal(0.0005, 0.008, 180)),
        },
        index=dates,
    )


@pytest.fixture
def client():
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_form():
    return {
        "tickers": "AAPL, MSFT",
        "start_date": "2024-01-01",
        "end_date": "2024-06-01",
        "risk_free_rate": "0.04",
        "risk_aversion": "3.0",
    }
