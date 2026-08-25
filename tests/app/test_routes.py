"""Tests for the HTTP routes in :mod:`app.routes`."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
from werkzeug.datastructures import MultiDict

from app.config import DEFAULT_APP_CONFIG
from app.forms import parse_form
from app.routes import (
    DEFAULT_LOOKBACK_YEARS,
    _default_form_values,
    _selected_asset_classes,
)

FORM = {
    "tickers": "AAPL, MSFT",
    "start_date": "2024-01-01",
    "end_date": "2024-06-01",
    "risk_free_rate": "0.04",
    "risk_aversion": "3.0",
}


class TestSelectedAssetClasses:
    """`_selected_asset_classes` must handle both MultiDict and plain dict."""

    def test_multidict_uses_getlist(self):
        form = MultiDict()
        form.setlist("asset_classes", ["stocks", "bonds"])
        assert _selected_asset_classes(form) == ["stocks", "bonds"]

    def test_plain_dict_splits_comma_separated(self):
        assert _selected_asset_classes({"asset_classes": "stocks,bonds"}) == [
            "stocks",
            "bonds",
        ]

    def test_plain_dict_without_key(self):
        assert _selected_asset_classes({}) == []


class TestIndex:
    def test_index_renders_form(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"<form" in response.data
        assert b"tickers" in response.data


class TestHealthz:
    def test_healthz(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}


class TestAnalyze:
    def test_invalid_tickers_renders_error(self, client):
        response = client.post("/analyze", data={**FORM, "tickers": "ONLYONE"})
        assert response.status_code == 400
        assert b"at least 2" in response.data

    def test_inverted_dates_render_error(self, client):
        response = client.post(
            "/analyze",
            data={**FORM, "start_date": "2024-06-01", "end_date": "2024-01-01"},
        )
        assert response.status_code == 400
        assert b"earlier than" in response.data

    def test_happy_path(self, client):
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
            response = client.post("/analyze", data=FORM)

        assert response.status_code == 200
        assert b"Analysis results" in response.data
        assert b"Sharpe ratio" in response.data

    def test_negative_risk_aversion_renders_convexity_note(self, client):
        """A risk-seeking A must run end-to-end and warn that U* is a minimum."""
        dates = pd.bdate_range("2024-01-01", periods=120)
        prices = pd.DataFrame(
            {"AAPL": range(100, 220), "MSFT": range(200, 320)},
            index=dates,
            dtype=float,
        )

        mock_dl = MagicMock()
        mock_dl.__getitem__.return_value = prices

        with patch("portfolio.data.data_ingestion.yf.download", return_value=mock_dl):
            response = client.post("/analyze", data={**FORM, "risk_aversion": "-2.5"})

        assert response.status_code == 200
        assert b"-2.50" in response.data
        assert b"not a maximum" in response.data

    def test_zero_risk_aversion_renders_error(self, client):
        response = client.post("/analyze", data={**FORM, "risk_aversion": "0"})
        assert response.status_code == 400
        assert b"must be non-zero and at least" in response.data

    def test_unexpected_error_returns_500(self, client):
        """A non-ValueError from the pipeline is logged and rendered as 500."""
        with patch("app.routes.run_analysis", side_effect=RuntimeError("boom")):
            response = client.post("/analyze", data=FORM)
        assert response.status_code == 500
        assert b"Unexpected error" in response.data

    def test_value_error_from_pipeline_returns_400(self, client):
        with patch(
            "app.routes.run_analysis", side_effect=ValueError("no data for window")
        ):
            response = client.post("/analyze", data=FORM)
        assert response.status_code == 400
        assert b"no data for window" in response.data


class TestDefaultFormValues:
    """The landing page must prefill a form that `parse_form` will accept."""

    def test_defaults_are_accepted_by_parse_form(self):
        request = parse_form(_default_form_values())
        assert len(request.tickers) >= DEFAULT_APP_CONFIG.min_tickers
        assert request.start_date < request.end_date

    def test_default_window_is_three_years(self):
        values = _default_form_values()
        start = date.fromisoformat(values["start_date"])
        end = date.fromisoformat(values["end_date"])
        assert (end - start).days == DEFAULT_LOOKBACK_YEARS * 365
