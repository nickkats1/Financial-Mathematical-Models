from portfolio import config





class TestConfig:
    """Test that values are in config"""
    
    def test_stock_tickers(self):
        """Test that stock tickers are in config"""
        
        assert "NVDA" in config.stock_tickers
        assert "META" in config.stock_tickers
        
    def test_etf_tickers(self):
        """test etf tickers"""
        
        assert "ZCSH" in config.etf_tickers
        
    def test_crypto_tickers(self):
        """test bond tickers"""
        
        assert "BTC-USD" in config.crypto_tickers
        
        
    def test_bond_tickers(self):
        """test if ticker symbols are in 'bond_tickers'"""
        
        assert "^IRX" in config.bond_tickers
        
    def test_sp500_ticker(self):
        """test sp500 ticker in config"""
        assert "^GSPC" == config.sp500_ticker
        
    def test_all_tickers(self):
        """test all tickers in config"""
        
        all_tickers = config.all_tickers
        
   
        
        
        assert "NVDA" in all_tickers
        assert "BTC-USD" in all_tickers
        assert "^IRX" in all_tickers
        assert "QQQ" in all_tickers
        


        
        
    def test_start_date(self):
        """test start date in config"""
        
        expected = "2022-12-01"
        
        actual = config.start_date
        
        assert expected == actual
        
    def test_end_date(self):
        """test end date in config"""
        
        expected = "2026-04-12"
        
        actual = config.end_date
        
        assert expected == actual


        