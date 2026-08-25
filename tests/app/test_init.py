"""Tests for the application factory in :mod:`app` — ``create_app``.

Covers the wiring create_app owns: security headers, the ephemeral testing
secret key, and the Flask-Limiter / CSRF middleware. Route behaviour lives in
``test_routes.py``; the settings object itself lives in ``test_config.py``.
"""

import pytest

from app import create_app, limiter
from app.config import DEV_SECRET_PLACEHOLDER, AppConfig


def _csrf_token(client) -> str:
    """Pull a fresh CSRF token from a GET on the index page."""
    body = client.get("/").data.decode()
    marker = 'name="csrf_token" value="'
    start = body.index(marker) + len(marker)
    end = body.index('"', start)
    return body[start:end]


@pytest.fixture
def live_app():
    """A non-test Flask app with the limiter active and storage reset.

    CSRF is left enabled; tests fetch a valid token from ``/`` and pass it
    through, so the rate-limit assertion exercises the real production
    middleware order (limiter → CSRF → route).
    """
    app = create_app(config=AppConfig())
    limiter.enabled = True
    limiter.reset()
    yield app
    limiter.reset()
    limiter.enabled = False


class TestSecurityHeaders:
    def test_security_headers_present(self, client):
        response = client.get("/")
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Referrer-Policy"] == "no-referrer"


class TestSecretKey:
    def test_testing_mode_uses_random_key(self):
        app = create_app({"TESTING": True})
        assert app.config["SECRET_KEY"] != DEV_SECRET_PLACEHOLDER
        assert len(app.config["SECRET_KEY"]) >= 32


class TestRateLimit:
    def test_analyze_returns_429_after_ten_requests(self, live_app):
        """The 10/min cap on /analyze must trip after the limit is reached."""
        client = live_app.test_client()
        token = _csrf_token(client)

        codes = []
        for _ in range(15):
            response = client.post(
                "/analyze",
                data={
                    "csrf_token": token,
                    "tickers": "",  # invalid → 400 fast, no yfinance call
                    "start_date": "2024-01-01",
                    "end_date": "2024-06-01",
                    "risk_free_rate": "0.04",
                    "risk_aversion": "3.0",
                },
            )
            codes.append(response.status_code)
        assert codes.count(429) >= 2, f"expected rate limit to trip, got {codes}"

    def test_healthz_is_exempt(self, live_app):
        """``/healthz`` must not be subject to any rate limit."""
        client = live_app.test_client()
        for _ in range(150):
            assert client.get("/healthz").status_code == 200

    def test_analyze_requires_csrf_token(self, live_app):
        """A POST without a CSRF token should be rejected."""
        client = live_app.test_client()
        response = client.post("/analyze", data={"tickers": "AAPL, MSFT"})
        assert response.status_code == 400
