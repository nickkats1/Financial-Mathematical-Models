import numpy as np

from portfolio.models.risk import get_var, get_cvar



class TestRisk:
    """Test Value at Risk and Conditional Value at Risk"""
    def test_var(self, fake_prices):
        """Test Value At Risk"""
        
        var = get_cvar(fake_prices)
        
        
        assert np.dtype(var) == float
        
    def test_var_cvar(self, fake_prices):
        """Test that cvar is less than var"""
        
        var = get_var(fake_prices)
        cvar = get_cvar(fake_prices)
        
        
        assert cvar < var
        

        