"""Flask application factory for the portfolio analytics web app."""

import secrets

from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect

from app.config import AppConfig
from portfolio.data.data_ingestion import configure_price_cache

_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}

limiter = Limiter(key_func=get_remote_address)
csrf = CSRFProtect()


def create_app(
    test_config: dict | None = None,
    *,
    config: AppConfig | None = None,
) -> Flask:
    """Create and configure a Flask application instance.

    ``config`` carries this project's own settings; ``test_config`` is the
    raw-Flask-keys escape hatch (``TESTING``, ``WTF_CSRF_ENABLED``) applied on top.
    """
    testing = bool(test_config and test_config.get("TESTING"))
    if config is None:
        # Tests get an ephemeral key so they never depend on the ambient environment.
        config = (
            AppConfig(secret_key=secrets.token_hex(32))
            if testing
            else AppConfig.from_env()
        )

    app = Flask(__name__, instance_relative_config=False)
    app.config.from_mapping(config.to_flask_mapping())
    if test_config is not None:
        app.config.update(test_config)

    # Set rather than defaulted: the Limiter is a module-level singleton whose
    # init_app returns early while disabled, so a testing app would otherwise
    # leave it disabled — and storage-less — for every app built afterwards.
    app.config["RATELIMIT_ENABLED"] = not app.config.get("TESTING", False)
    if app.config.get("TESTING"):
        app.config["WTF_CSRF_ENABLED"] = False

    app.logger.setLevel(config.log_level)
    configure_price_cache(
        config.price_cache_ttl_seconds,
        config.price_cache_max_entries,
    )

    limiter.init_app(app)
    csrf.init_app(app)

    @app.after_request
    def set_security_headers(response):
        for header, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response

    from app.routes import bp as main_bp

    app.register_blueprint(main_bp)
    return app
