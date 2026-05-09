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
    """Pick a SECRET_KEY appropriate for the current environment.

    - Tests / explicit ``TESTING`` config: any value is fine; we generate
      an ephemeral one.
    - Production (``FLASK_ENV=production``): refuse to start if
      ``FLASK_SECRET_KEY`` is unset or still the dev placeholder.
    - Otherwise: fall back to the env var, or to a noisy dev placeholder
      so local development still works.
    """
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
    """Create and configure a Flask application instance.

    Args:
        test_config: Optional config overrides for testing.

    Returns:
        A configured :class:`flask.Flask` instance.
    """
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_mapping(
        SECRET_KEY=_resolve_secret_key(test_config),
        WTF_CSRF_TIME_LIMIT=None,
    )
    if test_config is not None:
        app.config.update(test_config)

    limiter.init_app(app)
    if app.config.get("TESTING"):
        limiter.enabled = False
        app.config["WTF_CSRF_ENABLED"] = False

    csrf.init_app(app)

    from app.routes import bp as main_bp

    app.register_blueprint(main_bp)
    return app
