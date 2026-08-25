"""Runtime configuration for the Flask app, sourced from the environment.

:class:`AppConfig` holds the knobs that vary per deployment — secret key, rate
limits, price-cache sizing, request bounds. Domain constants (tickers, market
proxy, confidence levels) live in :mod:`portfolio.config` instead.

Two environment names are unprefixed because they predate this module and are
referenced by ``docker/docker-compose.yml``: ``FLASK_SECRET_KEY`` and
``FLASK_ENV``. Everything else uses an ``FMM_`` prefix so it cannot collide
with Flask's own configuration namespace.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEV_SECRET_PLACEHOLDER = "dev-secret-change-me"

_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"})


def _env_str(env: Mapping[str, str], name: str) -> str | None:
    """Return a stripped env value, or None when unset or blank."""
    return env.get(name, "").strip() or None


def _env_int(env: Mapping[str, str], name: str) -> int | None:
    """Return an env value parsed as int, or None when unset or blank."""
    raw = _env_str(env, name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from None


def _require_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}.")


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Runtime configuration. Derive variants with :func:`dataclasses.replace`."""

    secret_key: str = DEV_SECRET_PLACEHOLDER
    env: str = "development"
    rate_limit_default: str = "120 per minute"
    rate_limit_analyze: str = "10 per minute"
    rate_limit_storage_uri: str = "memory://"
    # Mirrors _CACHE_TTL_SECONDS / _CACHE_MAX_ENTRIES in portfolio/data/data_ingestion.py.
    price_cache_ttl_seconds: int = 300
    price_cache_max_entries: int = 1024
    max_ticker_input_length: int = 1000
    min_tickers: int = 2
    max_tickers: int = 200
    port: int = 8080
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        # Frozen dataclasses have no other hook for normalisation.
        object.__setattr__(self, "log_level", self.log_level.strip().upper())
        self.validate_secret_key()
        self.validate_ticker_bounds()
        self.validate_cache()
        self.validate_server()

    @property
    def is_production(self) -> bool:
        return self.env.strip().lower() == "production"

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        **overrides: Any,
    ) -> "AppConfig":
        """Build from ``env`` (defaults to ``os.environ``); overrides win over it."""
        source = os.environ if env is None else env
        from_environment: dict[str, Any] = {
            "secret_key": _env_str(source, "FLASK_SECRET_KEY"),
            "env": _env_str(source, "FLASK_ENV"),
            "rate_limit_default": _env_str(source, "FMM_RATE_LIMIT_DEFAULT"),
            "rate_limit_analyze": _env_str(source, "FMM_RATE_LIMIT_ANALYZE"),
            "rate_limit_storage_uri": _env_str(source, "FMM_RATE_LIMIT_STORAGE_URI"),
            "price_cache_ttl_seconds": _env_int(source, "FMM_PRICE_CACHE_TTL_SECONDS"),
            "price_cache_max_entries": _env_int(source, "FMM_PRICE_CACHE_MAX_ENTRIES"),
            "max_ticker_input_length": _env_int(source, "FMM_MAX_TICKER_INPUT_LENGTH"),
            "min_tickers": _env_int(source, "FMM_MIN_TICKERS"),
            "max_tickers": _env_int(source, "FMM_MAX_TICKERS"),
            "port": _env_int(source, "FMM_PORT"),
            "log_level": _env_str(source, "FMM_LOG_LEVEL"),
        }
        present = {k: v for k, v in from_environment.items() if v is not None}
        return cls(**{**present, **overrides})

    def validate_secret_key(self) -> None:
        """Production must carry a real key; development tolerates the placeholder."""
        if not self.is_production:
            return
        if not self.secret_key or self.secret_key == DEV_SECRET_PLACEHOLDER:
            raise RuntimeError(
                "FLASK_SECRET_KEY must be set to a non-default value when "
                "FLASK_ENV=production. Generate one with:\n"
                "    python -c 'import secrets; print(secrets.token_hex(32))'"
            )

    def validate_ticker_bounds(self) -> None:
        """Universe bounds must be orderable and leave room for a covariance matrix."""
        if self.min_tickers < 2:
            raise ValueError(
                f"min_tickers must be at least 2, got {self.min_tickers}."
            )
        if self.max_tickers < self.min_tickers:
            raise ValueError(
                f"max_tickers ({self.max_tickers}) must be at least "
                f"min_tickers ({self.min_tickers})."
            )
        _require_positive("max_ticker_input_length", self.max_ticker_input_length)

    def validate_cache(self) -> None:
        _require_positive("price_cache_ttl_seconds", self.price_cache_ttl_seconds)
        _require_positive("price_cache_max_entries", self.price_cache_max_entries)

    def validate_server(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be between 1 and 65535, got {self.port}.")
        if self.log_level not in _LOG_LEVELS:
            raise ValueError(
                f"log_level {self.log_level!r} is not one of {sorted(_LOG_LEVELS)}."
            )

    def to_flask_mapping(self) -> dict[str, Any]:
        """The UPPER_CASE keys Flask and Flask-Limiter read from ``app.config``."""
        return {
            "SECRET_KEY": self.secret_key,
            "WTF_CSRF_TIME_LIMIT": None,
            "SESSION_COOKIE_HTTPONLY": True,
            "SESSION_COOKIE_SAMESITE": "Lax",
            "SESSION_COOKIE_SECURE": self.is_production,
            "RATELIMIT_DEFAULT": self.rate_limit_default,
            "RATELIMIT_STORAGE_URI": self.rate_limit_storage_uri,
            "RATELIMIT_ANALYZE": self.rate_limit_analyze,
            # The service layer reads its bounds back off the live app.
            "APP_CONFIG": self,
        }


DEFAULT_APP_CONFIG = AppConfig()
