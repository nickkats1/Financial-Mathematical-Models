"""Tests for the asset-class registry and PortfolioConfig in :mod:`portfolio.config`."""

from dataclasses import replace

import pytest

from portfolio.config import (
    ASSET_CLASSES,
    DEFAULT_CONFIG,
    AssetClass,
    PortfolioConfig,
    _build_registry,
    get_asset_class,
)


class TestAssetClass:
    def test_rejects_empty_tickers(self):
        with pytest.raises(ValueError, match="has no tickers"):
            AssetClass(name="empty", label="Empty", tickers=())

    def test_rejects_duplicate_tickers(self):
        with pytest.raises(ValueError, match="duplicate tickers"):
            AssetClass(name="dupes", label="Dupes", tickers=("AAPL", "AAPL"))

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match="name must not be empty"):
            AssetClass(name="", label="Nameless", tickers=("AAPL",))


class TestAssetClassRegistry:
    def test_registry_names(self):
        assert set(ASSET_CLASSES) == {"stocks", "etfs", "bonds", "crypto"}

    def test_preset_carries_its_own_name(self):
        for name, asset_class in ASSET_CLASSES.items():
            assert asset_class.name == name

    def test_labels(self):
        assert get_asset_class("stocks").label == "Stocks"
        assert get_asset_class("etfs").label == "ETFs"
        assert get_asset_class("bonds").label == "Treasury bonds"
        assert get_asset_class("crypto").label == "Crypto"

    @pytest.mark.parametrize(
        ("name", "ticker"),
        [
            ("stocks", "NVDA"),
            ("stocks", "META"),
            ("etfs", "SPY"),
            ("etfs", "QQQ"),
            ("bonds", "^IRX"),
            ("crypto", "BTC-USD"),
        ],
    )
    def test_expected_tickers_present(self, name, ticker):
        assert ticker in get_asset_class(name).tickers

    def test_unknown_name_lists_what_is_available(self):
        with pytest.raises(ValueError, match=r"not found\. Available: ") as exc:
            get_asset_class("equities")
        assert "stocks" in str(exc.value)

    def test_duplicate_preset_names_are_rejected(self):
        preset = get_asset_class("stocks")
        with pytest.raises(ValueError, match="Duplicate asset class name"):
            _build_registry((preset, preset))


class TestPortfolioConfig:
    def test_defaults(self):
        assert DEFAULT_CONFIG.market_ticker == "^GSPC"
        assert DEFAULT_CONFIG.start_date == "2022-12-01"
        assert DEFAULT_CONFIG.end_date == "2026-04-30"
        assert DEFAULT_CONFIG.confidence_levels == (0.90, 0.95, 0.99)
        assert DEFAULT_CONFIG.default_confidence == 0.95

    def test_all_tickers_spans_every_preset_and_the_market_proxy(self):
        all_tickers = DEFAULT_CONFIG.all_tickers
        for ticker in ("NVDA", "QQQ", "^IRX", "BTC-USD", "^GSPC"):
            assert ticker in all_tickers

    def test_all_tickers_has_no_duplicates(self):
        all_tickers = DEFAULT_CONFIG.all_tickers
        assert len(set(all_tickers)) == len(all_tickers)

    def test_default_confidence_must_be_one_of_the_levels(self):
        with pytest.raises(ValueError, match="must be one of"):
            replace(DEFAULT_CONFIG, default_confidence=0.5)

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
    def test_confidence_levels_must_be_in_open_unit_interval(self, level):
        with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
            PortfolioConfig(confidence_levels=(level,), default_confidence=level)

    def test_empty_confidence_levels_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            PortfolioConfig(confidence_levels=())

    def test_dates_must_be_iso(self):
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            replace(DEFAULT_CONFIG, start_date="12/01/2022")

    def test_dates_must_be_ordered(self):
        with pytest.raises(ValueError, match="earlier than end_date"):
            replace(DEFAULT_CONFIG, start_date="2026-01-01", end_date="2025-01-01")

    @pytest.mark.parametrize("rate", [-0.01, 1.01])
    def test_risk_free_rate_bounds(self, rate):
        with pytest.raises(ValueError, match="risk_free_rate"):
            replace(DEFAULT_CONFIG, risk_free_rate=rate)

    @pytest.mark.parametrize("aversion", [0.0, -6.0])
    def test_risk_aversion_rejects_zero_and_below_floor(self, aversion):
        with pytest.raises(ValueError, match="risk_aversion must be non-zero"):
            replace(DEFAULT_CONFIG, risk_aversion=aversion)

    @pytest.mark.parametrize("aversion", [-5.0, -3.0, 3.25])
    def test_risk_aversion_accepts_negative_and_fractional(self, aversion):
        assert replace(DEFAULT_CONFIG, risk_aversion=aversion).risk_aversion == aversion

    def test_utility_scaling_factor_must_be_positive(self):
        with pytest.raises(ValueError, match="utility_scaling_factor"):
            replace(DEFAULT_CONFIG, utility_scaling_factor=0.0)
