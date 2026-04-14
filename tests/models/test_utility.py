import pytest

from portfolio.models.utility import get_utility


class TestUtility:
    """Test Utility Model"""
    def test_risk_averse(self, fake_prices):
        """Test level of utility when risk-aversion is < 0"""
        
        utility = get_utility(fake_prices, risk_aversion=0)
        
        assert utility is not None
        
        
