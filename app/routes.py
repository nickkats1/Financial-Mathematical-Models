"""HTTP routes for the portfolio analytics web app."""

import logging
from datetime import date, timedelta

from flask import Blueprint, render_template, request

from app import limiter
from app.services import ASSET_CLASS_LABELS, parse_form, run_analysis

bp = Blueprint("main", __name__)
logger = logging.getLogger(__name__)


def _default_form_values() -> dict:
    """Return the default form values shown on the landing page."""
    today = date.today()
    return {
        "tickers": "AAPL, MSFT, GOOGL, AMZN, NVDA",
        "start_date": (today - timedelta(days=3 * 365)).isoformat(),
        "end_date": today.isoformat(),
        "risk_free_rate": "0.04",
        "risk_aversion": "3.0",
        "market_ticker": "^GSPC",
    }


def _selected_asset_classes(form) -> list:
    """Return the asset-class checkbox values, robust to plain dicts in tests."""
    if hasattr(form, "getlist"):
        return form.getlist("asset_classes")
    raw = form.get("asset_classes", "")
    return [value for value in raw.split(",") if value] if raw else []


def _render_index(form_defaults, error: str | None, status: int = 200):
    """Render the index page, threading checkbox state and labels through."""
    return (
        render_template(
            "index.html",
            defaults=form_defaults,
            asset_class_labels=ASSET_CLASS_LABELS,
            selected_asset_classes=_selected_asset_classes(form_defaults),
            error=error,
        ),
        status,
    )


@bp.route("/", methods=["GET"])
def index():
    """Render the analysis form with sensible default values."""
    body, status = _render_index(_default_form_values(), error=None)
    return body, status


@bp.route("/analyze", methods=["POST"])
@limiter.limit("10 per minute")
def analyze():
    """Validate the form, run the analysis, and render the results page."""
    form = request.form

    try:
        analysis_request = parse_form(form)
    except ValueError as exc:
        return _render_index(form, error=str(exc), status=400)

    try:
        result = run_analysis(analysis_request)
    except ValueError as exc:
        return _render_index(form, error=str(exc), status=400)
    except Exception as exc:  # noqa: BLE001 — log and surface a generic message
        logger.exception("Unexpected error while running analysis")
        return _render_index(
            form,
            error=f"Unexpected error: {exc.__class__.__name__}",
            status=500,
        )

    return render_template(
        "results.html",
        result=result,
        asset_class_labels=ASSET_CLASS_LABELS,
    )


@bp.route("/healthz", methods=["GET"])
@limiter.exempt
def healthz():
    """Liveness probe used by the Docker healthcheck."""
    return {"status": "ok"}, 200
