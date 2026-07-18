"""Flask application factory for the portfolio analytics web app."""

import os
import secrets

from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect

DEV_SECRET_PLACEHOLDER = "dev-secret-change-me"

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["120 per minute"],
    storage_uri="memory://",
)
csrf = CSRFProtect()


def _resolve_secret_key(test_config: dict | None) -> str:
    """Ephemeral key for tests; require FLASK_SECRET_KEY in production; else dev placeholder."""
    if test_config is not None and test_config.get("TESTING"):
        return secrets.token_hex(32)

    env_key = os.environ.get("FLASK_SECRET_KEY", "").strip()
    is_production = os.environ.get("FLASK_ENV", "").lower() == "production"

    if is_production:
        if not env_key or env_key == DEV_SECRET_PLACEHOLDER:
            raise RuntimeError(
                "FLASK_SECRET_KEY must be set to a non-default value when "
                "FLASK_ENV=production. Generate one with:\n"
                "    python -c 'import secrets; print(secrets.token_hex(32))'"
            )
        return env_key

    return env_key or DEV_SECRET_PLACEHOLDER


def create_app(test_config: dict | None = None) -> Flask:
    """Create and configure a Flask application instance."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_mapping(
        SECRET_KEY=_resolve_secret_key(test_config),
        WTF_CSRF_TIME_LIMIT=None,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("FLASK_ENV", "").lower() == "production",
    )
    if test_config is not None:
        app.config.update(test_config)

    limiter.init_app(app)
    if app.config.get("TESTING"):
        limiter.enabled = False
        app.config["WTF_CSRF_ENABLED"] = False

    csrf.init_app(app)

    @app.after_request
    def set_security_headers(response):
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' https://cdn.jsdelivr.net; "
            "style-src 'self'; "
            "img-src 'self' data:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'",
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    from app.routes import bp as main_bp

    app.register_blueprint(main_bp)
    return app
