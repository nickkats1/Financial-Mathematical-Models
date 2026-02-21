# Financial Mathematical Models

![Python](https://img.shields.io/badge/python-3.x-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This repository implements five quantitative financial models in Python, covering core topics in modern portfolio management, risk measurement, and investor utility optimization. These models are designed to be modular, configurable, and easy to extend.

## Models

| Model | Description |
|-------|-------------|
| **Modern Portfolio Theory (MPT)** | Constructs an optimal portfolio by maximizing expected return for a given level of risk using mean-variance optimization. |
| **Single Index Model (SIM)** | Simplifies portfolio construction by relating each asset's return to a single market index (e.g., S&P 500). |
| **Value at Risk (VaR)** | Estimates the maximum potential loss of a portfolio over a given time horizon at a specified confidence level. |
| **Conditional Value at Risk (CVaR)** | Extends VaR by measuring the expected loss in the worst-case tail scenarios beyond the VaR threshold. |
| **Maximizing Utility** | Optimizes portfolio allocation based on the investor's level of risk aversion using a utility function framework. |

## Results

---

### Single Index Model (SIM) — Regression Plots

Each chart below plots an asset's returns against the S&P 500 (^GSPC), with the fitted regression line representing the asset's systematic risk (beta).

#### Stocks

| | |
|---|---|
| ![AAPL](images/sim/single_index_model_AAPL.png) | ![MSFT](images/sim/single_index_model_MSFT.png) |
| ![GOOGL](images/sim/single_index_model_GOOGL.png) | ![NVDA](images/sim/single_index_model_NVDA.png) |
| ![TSLA](images/sim/single_index_model_TSLA.png) | ![NFLX](images/sim/single_index_model_NFLX.png) |
| ![F](images/sim/single_index_model_F.png) | ![GM](images/sim/single_index_model_GM.png) |
| ![MCD](images/sim/single_index_model_MCD.png) | ![SBUX](images/sim/single_index_model_SBUX.png) |
| ![WMT](images/sim/single_index_model_WMT.png) | ![TGT](images/sim/single_index_model_TGT.png) |

#### ETFs

| | |
|---|---|
| ![SPY](images/sim/single_index_model_SPY.png) | ![QQQ](images/sim/single_index_model_QQQ.png) |
| ![VOO](images/sim/single_index_model_VOO.png) | ![VTI](images/sim/single_index_model_VTI.png) |
| ![IWM](images/sim/single_index_model_IWM.png) | ![DIA](images/sim/single_index_model_DIA.png) |
| ![EFA](images/sim/single_index_model_EFA.png) | ![ARKK](images/sim/single_index_model_ARKK.png) |
| ![XLF](images/sim/single_index_model_XLF.png) | ![XLK](images/sim/single_index_model_XLK.png) |


#### Market Index

![^GSPC](images/sim/single_index_model_%5EGSPC.png)

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/nickkats1/Portfolio.git
cd Portfolio
pip install -r requirements.txt
```

## Usage

Run the main entry point:

```bash
python main.py
```

Configuration options (e.g., tickers, date ranges, risk-aversion coefficients) can be adjusted in `config.yaml`.

You can also explore the models interactively via the Jupyter notebooks in the `notebooks/` directory:

| Notebook | Description |
|----------|-------------|
| `_mpt.ipynb` | Modern Portfolio Theory walkthrough |
| `_risk.ipynb` | VaR and CVaR analysis |
| `_single_index_model.ipynb` | Single Index Model walkthrough |
| `_utility.ipynb` | Utility maximization analysis |

## Repository Structure

```text
├── main.py
├── setup.py
├── LICENSE
├── config.yaml
├── requirements.txt
├── src
│   ├── __init__.py
│   └── models
│       ├── __init__.py
│       ├── max_utility.py
│       ├── mpt.py
│       ├── single_index_model.py
│       ├── utility.py
│       └── value_at_risk.py
├── scripts
│   ├── data_ingestion.py
│   ├── __init__.py
│   └── returns.py
├── notebooks
│   ├── _mpt.ipynb
│   ├── _risk.ipynb
│   ├── _single_index_model.ipynb
│   └── _utility.ipynb
├── tools
│   ├── config.py
│   ├── __init__.py
│   └── logger.py
├── images
│   ├── ef_assets_heatmap.png
│   └── sim
└── data
    ├── processed
    │   ├── etf_returns.csv
    │   ├── expected_returns.csv
    │   ├── returns.csv
    │   ├── sp500_returns.csv
    │   ├── stock_returns.csv
    │   └── vol.csv
    └── raw
        ├── all_prices.csv
        ├── etfs.csv
        ├── sp500_prices.csv
        └── stocks.csv
```

## License

This project is licensed under the [MIT License](LICENSE).

























