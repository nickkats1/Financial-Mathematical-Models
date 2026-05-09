"""Tests for the Flask application factory, routes, and service layer."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app import create_app
from app.services import (
    AnalysisRequest,
    parse_form,
    parse_tickers,
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


# ---------------------------------------------------------------------------
# parse_tickers
# ---------------------------------------------------------------------------

class TestParseTickers:
    def test_parses_comma_separated(self):
        assert parse_tickers("aapl, msft, googl") == ["AAPL", "MSFT", "GOOGL"]

    def test_parses_whitespace_separated(self):
        assert parse_tickers("AAPL MSFT GOOGL") == ["AAPL", "MSFT", "GOOGL"]

    def test_deduplicates_preserving_order(self):
        assert parse_tickers("AAPL, msft, AAPL") == ["AAPL", "MSFT"]

    def test_allows_zero_or_one_ticker(self):
        """parse_tickers no longer enforces MIN_TICKERS — parse_form does,
        because the universe can also come from asset-class presets."""
        assert parse_tickers("") == []
        assert parse_tickers("AAPL") == ["AAPL"]

    def test_rejects_overlong_input(self):
        with pytest.raises(ValueError, match="too long"):
            parse_tickers("A," * 1000)


# ---------------------------------------------------------------------------
# parse_form
# ---------------------------------------------------------------------------

class TestParseForm:
    def test_returns_validated_request(self, valid_form):
        request = parse_form(valid_form)
        assert isinstance(request, AnalysisRequest)
        assert request.tickers == ["AAPL", "MSFT"]
        assert request.start_date == "2024-01-01"
        assert request.risk_free_rate == 0.04
        assert request.risk_aversion == 3.0

    def test_request_is_immutable(self, valid_form):
        """Frozen dataclass — runtime mutation should fail."""
        request = parse_form(valid_form)
        with pytest.raises((AttributeError, Exception)):
            request.risk_free_rate = 0.5  # type: ignore[misc]

    def test_rejects_bad_date(self, valid_form):
        valid_form["start_date"] = "not-a-date"
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            parse_form(valid_form)

    def test_rejects_inverted_date_range(self, valid_form):
        valid_form["start_date"] = "2024-06-01"
        valid_form["end_date"] = "2024-01-01"
        with pytest.raises(ValueError, match="earlier than"):
            parse_form(valid_form)

    def test_rejects_out_of_range_risk_free_rate(self, valid_form):
        valid_form["risk_free_rate"] = "1.5"
        with pytest.raises(ValueError, match="between 0 and 1"):
            parse_form(valid_form)

    def test_rejects_non_positive_risk_aversion(self, valid_form):
        valid_form["risk_aversion"] = "0"
        with pytest.raises(ValueError, match="must be positive"):
            parse_form(valid_form)

    def test_rejects_non_numeric_risk_free_rate(self, valid_form):
        valid_form["risk_free_rate"] = "abc"
        with pytest.raises(ValueError, match="must be a number"):
            parse_form(valid_form)

    def test_rejects_empty_universe(self, valid_form):
        valid_form["tickers"] = ""
        with pytest.raises(ValueError, match="at least 2"):
            parse_form(valid_form)

    def test_merges_asset_classes_with_typed_tickers(self, valid_form):
        """An asset-class preset should expand the universe even if no
        symbols are typed."""
        from werkzeug.datastructures import MultiDict

        form = MultiDict(valid_form)
        form["tickers"] = ""
        form.setlist("asset_classes", ["bonds"])
        request = parse_form(form)
        assert len(request.tickers) >= 2
        assert request.asset_classes == ["bonds"]

    def test_rejects_unknown_asset_class(self, valid_form):
        from werkzeug.datastructures import MultiDict

        form = MultiDict(valid_form)
        form.setlist("asset_classes", ["commodities"])
        with pytest.raises(ValueError, match="Unknown asset class"):
            parse_form(form)

    def test_market_ticker_defaults_when_blank(self, valid_form):
        request = parse_form(valid_form)
        assert request.market_ticker == "^GSPC"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

class TestRoutes:
    def test_index_renders_form(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"<form" in response.data
        assert b"tickers" in response.data

    def test_healthz(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}

    def test_analyze_with_invalid_tickers_renders_error(self, client):
        response = client.post(
            "/analyze",
            data={
                "tickers": "ONLYONE",
                "start_date": "2024-01-01",
                "end_date": "2024-06-01",
                "risk_free_rate": "0.04",
                "risk_aversion": "3.0",
            },
        )
        assert response.status_code == 400
        assert b"at least 2" in response.data

    def test_analyze_with_inverted_dates_renders_error(self, client):
        response = client.post(
            "/analyze",
            data={
                "tickers": "AAPL, MSFT",
                "start_date": "2024-06-01",
                "end_date": "2024-01-01",
                "risk_free_rate": "0.04",
                "risk_aversion": "3.0",
            },
        )
        assert response.status_code == 400
        assert b"earlier than" in response.data

    def test_analyze_happy_path(self, client):
        """End-to-end: mock yfinance and render the results page."""
        dates = pd.bdate_range("2024-01-01", periods=120)
        prices = pd.DataFrame(
            {"AAPL": range(100, 220), "MSFT": range(200, 320)},
            index=dates,
            dtype=float,
        )

        mock_dl = MagicMock()
        mock_dl.__getitem__.return_value = prices

        with patch("portfolio.data.data_ingestion.yf.download", return_value=mock_dl):
            response = client.post(
                "/analyze",
                data={
                    "tickers": "AAPL, MSFT",
                    "start_date": "2024-01-01",
                    "end_date": "2024-06-01",
                    "risk_free_rate": "0.04",
                    "risk_aversion": "3.0",
                },
            )

        assert response.status_code == 200
        assert b"Analysis results" in response.data
        assert b"Sharpe ratio" in response.data
