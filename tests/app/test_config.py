"""Tests for the environment-sourced runtime configuration in :mod:`app.config`."""

from dataclasses import replace

import pytest

from app.config import DEV_SECRET_PLACEHOLDER, AppConfig


class TestDefaults:
    def test_defaults_are_usable_without_any_environment(self):
        config = AppConfig()
        assert config.secret_key == DEV_SECRET_PLACEHOLDER
        assert config.env == "development"
        assert config.is_production is False
        assert config.rate_limit_default == "120 per minute"
        assert config.rate_limit_analyze == "10 per minute"
        assert config.price_cache_ttl_seconds == 300
        assert config.min_tickers == 2
        assert config.max_tickers == 200
        assert config.port == 8080

    @pytest.mark.parametrize("value", ["production", "PRODUCTION", "Production"])
    def test_is_production_is_case_insensitive(self, value):
        assert AppConfig(secret_key="x" * 64, env=value).is_production is True


class TestFromEnv:
    def test_empty_environment_yields_defaults(self):
        assert AppConfig.from_env({}) == AppConfig()

    def test_reads_flask_names_unprefixed(self):
        config = AppConfig.from_env(
            {"FLASK_SECRET_KEY": "a" * 64, "FLASK_ENV": "production"}
        )
        assert config.secret_key == "a" * 64
        assert config.is_production is True

    def test_reads_prefixed_names(self):
        config = AppConfig.from_env(
            {
                "FMM_MAX_TICKERS": "25",
                "FMM_RATE_LIMIT_ANALYZE": "3 per minute",
                "FMM_PRICE_CACHE_TTL_SECONDS": "60",
                "FMM_LOG_LEVEL": "debug",
            }
        )
        assert config.max_tickers == 25
        assert config.rate_limit_analyze == "3 per minute"
        assert config.price_cache_ttl_seconds == 60
        assert config.log_level == "DEBUG"

    def test_blank_and_whitespace_values_fall_back_to_defaults(self):
        config = AppConfig.from_env({"FMM_MAX_TICKERS": "   ", "FLASK_ENV": ""})
        assert config.max_tickers == 200
        assert config.env == "development"

    def test_unparseable_int_names_the_variable(self):
        with pytest.raises(ValueError, match="FMM_MAX_TICKERS must be an integer"):
            AppConfig.from_env({"FMM_MAX_TICKERS": "lots"})

    def test_keyword_overrides_beat_the_environment(self):
        config = AppConfig.from_env({"FMM_MAX_TICKERS": "25"}, max_tickers=7)
        assert config.max_tickers == 7


class TestSecretKeyPolicy:
    def test_production_refuses_missing_key(self):
        with pytest.raises(RuntimeError, match="FLASK_SECRET_KEY must be set"):
            AppConfig.from_env({"FLASK_ENV": "production"})

    def test_production_refuses_placeholder_key(self):
        with pytest.raises(RuntimeError, match="FLASK_SECRET_KEY must be set"):
            AppConfig.from_env(
                {"FLASK_ENV": "production", "FLASK_SECRET_KEY": DEV_SECRET_PLACEHOLDER}
            )

    def test_production_accepts_real_key(self):
        config = AppConfig.from_env(
            {"FLASK_ENV": "production", "FLASK_SECRET_KEY": "a" * 64}
        )
        assert config.secret_key == "a" * 64

    def test_development_tolerates_missing_key(self):
        assert AppConfig.from_env({}).secret_key == DEV_SECRET_PLACEHOLDER


class TestValidation:
    def test_max_tickers_must_not_be_below_min(self):
        with pytest.raises(ValueError, match="max_tickers"):
            replace(AppConfig(), min_tickers=10, max_tickers=5)

    def test_min_tickers_must_be_at_least_two(self):
        with pytest.raises(ValueError, match="min_tickers"):
            replace(AppConfig(), min_tickers=1)

    @pytest.mark.parametrize(
        "field_name", ["price_cache_ttl_seconds", "price_cache_max_entries"]
    )
    def test_cache_sizing_must_be_positive(self, field_name):
        with pytest.raises(ValueError, match=field_name):
            replace(AppConfig(), **{field_name: 0})

    def test_ticker_input_length_must_be_positive(self):
        with pytest.raises(ValueError, match="max_ticker_input_length"):
            replace(AppConfig(), max_ticker_input_length=0)

    @pytest.mark.parametrize("port", [0, 70000])
    def test_port_must_be_a_valid_tcp_port(self, port):
        with pytest.raises(ValueError, match="port"):
            replace(AppConfig(), port=port)

    def test_unknown_log_level_rejected(self):
        with pytest.raises(ValueError, match="log_level"):
            replace(AppConfig(), log_level="CHATTY")


class TestFlaskMapping:
    def test_emits_the_keys_flask_and_flask_limiter_read(self):
        mapping = AppConfig().to_flask_mapping()
        assert mapping["SECRET_KEY"] == DEV_SECRET_PLACEHOLDER
        assert mapping["RATELIMIT_DEFAULT"] == "120 per minute"
        assert mapping["RATELIMIT_STORAGE_URI"] == "memory://"
        assert mapping["RATELIMIT_ANALYZE"] == "10 per minute"
        assert mapping["SESSION_COOKIE_HTTPONLY"] is True

    def test_secure_cookie_follows_production(self):
        assert AppConfig().to_flask_mapping()["SESSION_COOKIE_SECURE"] is False
        production = AppConfig(secret_key="a" * 64, env="production")
        assert production.to_flask_mapping()["SESSION_COOKIE_SECURE"] is True

    def test_carries_itself_for_the_service_layer(self):
        config = AppConfig()
        assert config.to_flask_mapping()["APP_CONFIG"] is config
