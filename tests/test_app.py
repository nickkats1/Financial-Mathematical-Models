"""Tests for the Flask application factory, routes, and service layer."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from werkzeug.datastructures import MultiDict

from app import create_app
from app.routes import _selected_asset_classes
from app.services import (
    MAX_TICKERS,
    AnalysisRequest,
    parse_form,
    parse_tickers,
)


def _two_char_symbols(count: int) -> list[str]:
    """Return ``count`` distinct 2-letter ticker symbols (AA, AB, ...)."""
    import string

    letters = string.ascii_uppercase
    symbols = [a + b for a in letters for b in letters]
    assert count <= len(symbols)
    return symbols[:count]


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

    def test_rejects_too_many_tickers(self):
        # Two-char symbols keep the input under the 1000-char length cap so the
        # count check (not the length check) is the one that trips.
        many = " ".join(_two_char_symbols(MAX_TICKERS + 1))
        with pytest.raises(ValueError, match="at most 200"):
            parse_tickers(many)

    def test_accepts_exactly_max_tickers(self):
        many = " ".join(_two_char_symbols(MAX_TICKERS))
        assert len(parse_tickers(many)) == MAX_TICKERS

    def test_accepts_yahoo_symbol_punctuation(self):
        assert parse_tickers("BRK.B BTC-USD ^GSPC CL=F") == [
            "BRK.B", "BTC-USD", "^GSPC", "CL=F",
        ]

    @pytest.mark.parametrize("bad", ["AAPL;rm", "<script>", "A/B", "..%2F", "A" * 21])
    def test_rejects_invalid_symbols(self, bad):
        with pytest.raises(ValueError, match="Invalid ticker symbol"):
            parse_tickers(bad)


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
        form = MultiDict(valid_form)
        form["tickers"] = ""
        form.setlist("asset_classes", ["bonds"])
        request = parse_form(form)
        assert len(request.tickers) >= 2
        assert request.asset_classes == ["bonds"]

    def test_rejects_unknown_asset_class(self, valid_form):
        form = MultiDict(valid_form)
        form.setlist("asset_classes", ["commodities"])
        with pytest.raises(ValueError, match="Unknown asset class"):
            parse_form(form)

    def test_asset_classes_returned_in_canonical_order(self, valid_form):
        form = MultiDict(valid_form)
        form["tickers"] = ""
        form.setlist("asset_classes", ["crypto", "stocks"])
        request = parse_form(form)
        assert request.asset_classes == ["stocks", "crypto"]

    def test_duplicate_asset_classes_are_deduped(self, valid_form):
        form = MultiDict(valid_form)
        form.setlist("asset_classes", ["bonds", "bonds"])
        request = parse_form(form)
        assert request.asset_classes == ["bonds"]

    def test_asset_classes_from_plain_dict(self, valid_form):
        """A plain dict (no ``getlist``) parses comma-separated classes."""
        form = dict(valid_form)
        form["tickers"] = ""
        form["asset_classes"] = "bonds,stocks"
        request = parse_form(form)
        assert request.asset_classes == ["stocks", "bonds"]

    def test_rejects_combined_universe_too_large(self, valid_form):
        typed = " ".join(_two_char_symbols(MAX_TICKERS - 1))
        form = MultiDict(valid_form)
        form["tickers"] = typed
        form.setlist("asset_classes", ["stocks"])
        with pytest.raises(ValueError, match="Combined universe is too large"):
            parse_form(form)

    def test_market_ticker_defaults_when_blank(self, valid_form):
        request = parse_form(valid_form)
        assert request.market_ticker == "^GSPC"

    def test_market_ticker_normalised_to_upper(self, valid_form):
        valid_form["market_ticker"] = "spy"
        assert parse_form(valid_form).market_ticker == "SPY"

    def test_rejects_multi_symbol_market_ticker(self, valid_form):
        valid_form["market_ticker"] = "SPX ^GSPC"
        with pytest.raises(ValueError, match="single valid symbol"):
            parse_form(valid_form)

    def test_rejects_bad_end_date(self, valid_form):
        valid_form["end_date"] = "nope"
        with pytest.raises(ValueError, match="End date must be"):
            parse_form(valid_form)

    def test_rejects_negative_risk_free_rate(self, valid_form):
        valid_form["risk_free_rate"] = "-0.1"
        with pytest.raises(ValueError, match="between 0 and 1"):
            parse_form(valid_form)


class TestMergeUniverse:
    """`_merge_universe` de-dupes typed symbols against each other and against
    the expanded asset-class presets."""

    def test_typed_tickers_are_deduped(self):
        from app.services import _merge_universe

        assert _merge_universe(["AAPL", "AAPL", "MSFT"], []) == ["AAPL", "MSFT"]

    def test_class_tickers_appended_and_uppercased(self):
        from app.services import _merge_universe

        merged = _merge_universe(["AAPL"], ["bonds"])
        assert merged[0] == "AAPL"
        assert all(t == t.upper() for t in merged)
        assert len(merged) > 1

    def test_class_ticker_already_typed_is_not_duplicated(self):
        from app.services import ASSET_CLASS_TICKERS, _merge_universe

        first_bond = ASSET_CLASS_TICKERS["bonds"][0].upper()
        merged = _merge_universe([first_bond], ["bonds"])
        assert merged.count(first_bond) == 1


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

    def test_security_headers_present(self, client):
        response = client.get("/")
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Referrer-Policy"] == "no-referrer"

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

    def test_analyze_unexpected_error_returns_500(self, client):
        """A non-ValueError from the pipeline is logged and rendered as 500."""
        with patch(
            "app.routes.run_analysis", side_effect=RuntimeError("boom")
        ):
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
        assert response.status_code == 500
        assert b"Unexpected error" in response.data

    def test_analyze_value_error_from_pipeline_returns_400(self, client):
        with patch(
            "app.routes.run_analysis", side_effect=ValueError("no data for window")
        ):
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
        assert response.status_code == 400
        assert b"no data for window" in response.data
