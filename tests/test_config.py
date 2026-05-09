"""Tests for the values declared in :mod:`portfolio.config`."""

from portfolio import config


class TestConfig:
    """Smoke tests that the expected ticker symbols and dates are present."""

    def test_stock_tickers(self):
        assert "NVDA" in config.stock_tickers
        assert "META" in config.stock_tickers

    def test_etf_tickers(self):
        assert "SPY" in config.etf_tickers
        assert "QQQ" in config.etf_tickers

    def test_crypto_tickers(self):
        assert "BTC-USD" in config.crypto_tickers

    def test_bond_tickers(self):
        assert "^IRX" in config.bond_tickers

    def test_sp500_ticker(self):
        assert config.sp500_ticker == "^GSPC"

    def test_all_tickers(self):
        all_tickers = config.all_tickers
        assert "NVDA" in all_tickers
        assert "BTC-USD" in all_tickers
        assert "^IRX" in all_tickers
        assert "QQQ" in all_tickers

    def test_start_date(self):
        assert config.start_date == "2022-12-01"

    def test_end_date(self):
        assert config.end_date == "2026-04-30"

    def test_confidence_levels(self):
        assert config.confidence_level_95 == 0.95
        assert config.confidence_level_99 == 0.99
