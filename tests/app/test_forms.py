"""Tests for :mod:`app.forms` — ticker parsing and analysis-form validation."""

import dataclasses
import string

import pytest
from werkzeug.datastructures import MultiDict

from app.config import DEFAULT_APP_CONFIG
from app.forms import (
    ASSET_CLASS_TICKERS,
    AnalysisRequest,
    _merge_universe,
    parse_form,
    parse_tickers,
)

MAX_TICKERS = DEFAULT_APP_CONFIG.max_tickers


def _two_char_symbols(count: int) -> list[str]:
    """Return ``count`` distinct 2-letter ticker symbols (AA, AB, ...)."""
    letters = string.ascii_uppercase
    return [a + b for a in letters for b in letters][:count]


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


class TestParseForm:
    def test_returns_validated_request(self, valid_form):
        request = parse_form(valid_form)
        assert isinstance(request, AnalysisRequest)
        assert request.tickers == ("AAPL", "MSFT")
        assert request.start_date == "2024-01-01"
        assert request.risk_free_rate == 0.04
        assert request.risk_aversion == 3.0

    def test_request_is_immutable(self, valid_form):
        """AnalysisRequest is frozen; coverage cannot catch this regressing."""
        request = parse_form(valid_form)
        with pytest.raises(dataclasses.FrozenInstanceError):
            request.risk_free_rate = 0.5  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("start_date", "not-a-date", "YYYY-MM-DD"),
            ("end_date", "nope", "End date must be"),
            ("risk_free_rate", "1.5", "between 0 and 1"),
            ("risk_free_rate", "-0.1", "between 0 and 1"),
            ("risk_free_rate", "abc", "must be a number"),
            ("market_ticker", "SPX ^GSPC", "single valid symbol"),
            ("tickers", "", "at least 2"),
        ],
    )
    def test_rejects_invalid_field(self, valid_form, field, value, message):
        valid_form[field] = value
        with pytest.raises(ValueError, match=message):
            parse_form(valid_form)

    def test_rejects_inverted_date_range(self, valid_form):
        valid_form["start_date"] = "2024-06-01"
        valid_form["end_date"] = "2024-01-01"
        with pytest.raises(ValueError, match="earlier than"):
            parse_form(valid_form)

    @pytest.mark.parametrize("aversion", ["0", "-6"])
    def test_rejects_zero_and_below_floor_risk_aversion(self, valid_form, aversion):
        valid_form["risk_aversion"] = aversion
        with pytest.raises(ValueError, match="must be non-zero and at least"):
            parse_form(valid_form)

    @pytest.mark.parametrize("aversion", ["-3.5", "3.25"])
    def test_accepts_negative_and_fractional_risk_aversion(self, valid_form, aversion):
        valid_form["risk_aversion"] = aversion
        assert parse_form(valid_form).risk_aversion == float(aversion)

    def test_merges_asset_classes_with_typed_tickers(self, valid_form):
        """An asset-class preset should expand the universe even if no
        symbols are typed."""
        form = MultiDict(valid_form)
        form["tickers"] = ""
        form.setlist("asset_classes", ["bonds"])
        request = parse_form(form)
        assert len(request.tickers) >= 2
        assert request.asset_classes == ("bonds",)

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
        assert request.asset_classes == ("stocks", "crypto")

    def test_duplicate_asset_classes_are_deduped(self, valid_form):
        form = MultiDict(valid_form)
        form.setlist("asset_classes", ["bonds", "bonds"])
        request = parse_form(form)
        assert request.asset_classes == ("bonds",)

    def test_asset_classes_from_plain_dict(self, valid_form):
        """A plain dict (no ``getlist``) parses comma-separated classes."""
        form = dict(valid_form)
        form["tickers"] = ""
        form["asset_classes"] = "bonds,stocks"
        request = parse_form(form)
        assert request.asset_classes == ("stocks", "bonds")

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


class TestMergeUniverse:
    """`_merge_universe` de-dupes typed symbols against each other and against
    the expanded asset-class presets."""

    def test_typed_tickers_are_deduped(self):
        assert _merge_universe(["AAPL", "AAPL", "MSFT"], []) == ["AAPL", "MSFT"]

    def test_class_tickers_appended_and_uppercased(self):
        merged = _merge_universe(["AAPL"], ["bonds"])
        assert merged[0] == "AAPL"
        assert all(t == t.upper() for t in merged)
        assert len(merged) > 1

    def test_class_ticker_already_typed_is_not_duplicated(self):
        first_bond = ASSET_CLASS_TICKERS["bonds"][0].upper()
        merged = _merge_universe([first_bond], ["bonds"])
        assert merged.count(first_bond) == 1

