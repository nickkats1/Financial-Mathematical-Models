import pytest
import pandas as pd


@pytest.fixture
def dummy_config():
    """dummy config for project"""
    config = {
        "start_date": "2024-01-01",
        "end_date": "2024-01-05",
        "stock_tickers": ["META", "NVDA", "MCD"],
        "etf_tickers": ["BEG", "QQQ"],
        "bond_tickers": ["^TNX", "^IRX"],
        "crypto_tickers": ["BTC-USD", "ETH-USD"],
        "sp500_ticker": "^GSPC",
        "RISK_FREE_RATE": 0.001,
        "A": 4.0
    }
    config["all_tickers"] = (
        config["stock_tickers"] +
        config["etf_tickers"] +
        config["bond_tickers"] +
        config["crypto_tickers"] +
        [config["sp500_ticker"]]
    )
    return config



@pytest.fixture
def fake_prices():
    dates = pd.bdate_range("2024-01-01", periods=3)
    return pd.DataFrame({
        "META": [100.0, 101.0, 102.0],
        "NVDA": [12.01, 30.01, 2.50],
        "MCD": [100.02, 200.10, 600.50],
        "BEG": [500.03, 500.04, 500.06],
        "^GSPC": [2600, 2006, 2001],
        "QQQ": [10,11, 12],
        "BTC-USD": [10, 12, 33],
        "ETH-USD": [100, 200, 300],
        "^TNX": [10.02, 11.11, 12.01],
        "^IRX": [100, 101, 102]
    }, index=dates)
    
    
