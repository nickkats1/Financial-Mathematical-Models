"""HTTP routes for the portfolio analytics web app."""

import logging
from datetime import date, timedelta

from flask import Blueprint, current_app, render_template, request

from app import limiter
from app.config import DEFAULT_APP_CONFIG
from app.forms import ASSET_CLASS_LABELS, _selected_asset_classes, parse_form
from app.services import run_analysis
from portfolio.config import DEFAULT_CONFIG, MIN_RISK_AVERSION, get_asset_class

bp = Blueprint("main", __name__)
logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_YEARS = 3


def _default_form_values() -> dict:
    """Return the default form values shown on the landing page."""
    today = date.today()
    lookback = timedelta(days=DEFAULT_LOOKBACK_YEARS * 365)
    return {
        "tickers": ", ".join(get_asset_class("stocks").tickers[:5]),
        "start_date": (today - lookback).isoformat(),
        "end_date": today.isoformat(),
        "risk_free_rate": str(DEFAULT_CONFIG.risk_free_rate),
        "risk_aversion": str(DEFAULT_CONFIG.risk_aversion),
        "market_ticker": DEFAULT_CONFIG.market_ticker,
    }


def _render_index(form_defaults, error: str | None, status: int = 200):
    """Render the index page, threading checkbox state and labels through."""
    return (
        render_template(
            "index.html",
            defaults=form_defaults,
            asset_class_labels=ASSET_CLASS_LABELS,
            selected_asset_classes=_selected_asset_classes(form_defaults),
            min_risk_aversion=MIN_RISK_AVERSION,
            error=error,
        ),
        status,
    )


@bp.route("/", methods=["GET"])
def index():
    """Render the analysis form with sensible default values."""
    return _render_index(_default_form_values(), error=None)


@bp.route("/analyze", methods=["POST"])
@limiter.limit(lambda: current_app.config["RATELIMIT_ANALYZE"])
def analyze():
    """Validate the form, run the analysis, and render the results page."""
    form = request.form
    config = current_app.config.get("APP_CONFIG", DEFAULT_APP_CONFIG)

    try:
        analysis_request = parse_form(form, config)
    except ValueError as exc:
        return _render_index(form, error=str(exc), status=400)

    try:
        result = run_analysis(analysis_request, config)
    except ValueError as exc:
        return _render_index(form, error=str(exc), status=400)
    except Exception as exc:
        logger.exception("Unexpected error while running analysis")
        return _render_index(
            form,
            error=f"Unexpected error: {exc.__class__.__name__}",
            status=500,
        )

    return render_template(
        "results.html",
        result=result,
    )


@bp.route("/healthz", methods=["GET"])
@limiter.exempt
def healthz():
    """Liveness probe used by the Docker healthcheck."""
    return {"status": "ok"}, 200
