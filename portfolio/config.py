from typing import List


# --stock tickers ---
stock_tickers: List[str] = [
    "INTC", "HOLX", "NVDA",
    "NOK", "PLTR", "PLUG",
    "NIO", "CRWV", "EOSE",
    "TSLA", "AMZN", "GRAB",
    "SNAP", "MARA", "NU",
    "WULF", "SOFI", "MU",
    "BBD", "VG", "NOW",
    "ABEV", "ONDS", "AAL",
    "T", "APLD", "BMNR",
    "PATH", "RKT", "NFLX",
    "OPEN", "PBR", "SMCI",
    "CIFR", "MRVL", "F",
    "ORCL", "GGB", "AMD",
    "STLA", "IREN", "MSFT",
    "VZ", "FSLY", "GOOGL",
    "CCL", "HOOD", "BB",
    "NKE", "ET", "META",
    "MCD", "TGT", "WMT"
    ]


# --- etf_tickers ---

etf_tickers: List[str] = [
    "ZCSH", "BEG", "LABX",
    "GTAO", "CRMU", "CIFU",
    "EOSU", "MVLL", "CRMX",
    "VFPAF", "VOYX", "MRVU",
    "LUNL", "ARMG", "CSEX",
    "ODOT", "USGG", "PLU",
    "AVGX", "APLX", "CWVX",
    "WULX", "USAX", "AVGG",
    "PBRG", "AVL",
    "AVGU", "SMCL",
    "IREX", "AMUU", "IRE",
    "AMDL", "COZX",
    "QUBX", "TEMT", "QQQ", "FNGU"
]


# --- crypto tickers ---

crypto_tickers: List[str] = [
    "USDT-USD", "MPRO31258-USD", "BTC-USD",
    "ETH-USD", "USDC-USD", "SOL-USD",
    "JU-USD", "XRP-USD", "TAO22974-USD",
    "BNB-USD", "DOGE-USD", "USD136148-USD",
    "WETH-USD", "ZEC-USD", "UP39665-USD",
    "LINK-USD", "TRX-USD", "SOL16116-USD",
    "ADA-USD", "CBBTC32994-USD", "QUQ-USD",
    "SUI20947-USD", "USDT39520-USD",
    "RAVE38967-USD", "PEPE24478-USD", "HYPE32196-USD"
]


# --- Bond Tickers ---

bond_tickers: List[str] = [
    "^IRX", "^FVX", "^TNX",
    "^TYX"
]

# --- SP&500 Ticker

sp500_ticker: str = "^GSPC"




# --- All Tickers ---

all_tickers: List[str] = [
    *stock_tickers, *etf_tickers,
    *bond_tickers, *crypto_tickers,
    sp500_ticker
]



# --- date range ---

start_date: str = "2022-12-01"

end_date: str = "2026-04-30"

# --- Misc ---

var_95: float = 0.95

# CI for 99%

var_99: float = 0.99





