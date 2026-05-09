"""Tests for :mod:`portfolio.models.risk`."""

import math

import pandas as pd
import pytest

from portfolio.models.risk import get_cvar, get_var


class TestRisk:
    """Tests for Value at Risk and Conditional Value at Risk."""

    def test_var_is_finite_float(self, fake_prices):
        var = get_var(fake_prices)
        assert isinstance(var, float)
        assert math.isfinite(var)

    def test_cvar_is_at_most_var(self, fake_prices):
        """CVaR is the mean of the left tail, so it must be <= VaR."""
        var = get_var(fake_prices)
        cvar = get_cvar(fake_prices)
        assert cvar <= var

    def test_invalid_confidence_raises(self, fake_prices):
        with pytest.raises(ValueError, match="confidence must be in"):
            get_var(fake_prices, confidence=0.0)
        with pytest.raises(ValueError, match="confidence must be in"):
            get_cvar(fake_prices, confidence=1.0)

    def test_empty_prices_raises(self):
        with pytest.raises(ValueError, match="prices is empty"):
            get_var(pd.DataFrame())
