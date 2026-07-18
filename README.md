# Financial Mathematical Models

A small Python library, notebook collection, and Flask web app covering a
handful of foundational topics in quantitative finance:

- **Modern Portfolio Theory** — long-only, fully-invested max-Sharpe portfolios
- **Single Index Model** — alpha, beta, and the systematic / firm-specific risk decomposition
- **Mean-variance utility theory** — utility for risk-averse, risk-neutral, and risk-loving investors, plus the optimal cash / risky-asset allocation
- **Value at Risk and Conditional Value at Risk** — historical (non-parametric) VaR and Expected Shortfall

Everything is built on `pypfopt`, `statsmodels`, `pandas`, and `yfinance`.
The Flask app on top is hardened with CSRF protection, per-IP rate
limiting, an in-memory price cache with retry on transient yfinance
failures, and a startup-time check that refuses to boot in production
without a real `FLASK_SECRET_KEY`.

## Repository layout

```
.
├── app/                       # Flask web app (form input + results page)
│   ├── __init__.py            # create_app factory
│   ├── routes.py              # HTTP routes
│   ├── services.py            # Glue between form input and portfolio.*
│   ├── templates/             # Jinja2 templates
│   └── static/                # CSS + JavaScript (Chart.js bar chart)
├── notebooks/                 # Concept + EDA + API-demo notebooks
│   ├── _data_ingestion.ipynb
│   ├── _mpt.ipynb
│   ├── _risk.ipynb
│   ├── _single_index_model.ipynb
│   └── _utility.ipynb
├── portfolio/                 # Core library
│   ├── config.py              # Tickers, date range, default confidence levels
│   ├── data/data_ingestion.py # yfinance wrapper
│   └── models/
│       ├── mpt.py
│       ├── risk.py
│       ├── single_index_model.py
│       └── utility.py
├── tests/                     # pytest suite (120+ tests, 100% coverage, fully offline)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── wsgi.py                    # gunicorn entry point
```

## Installation

```bash
git clone https://github.com/nickkats1/Financial-Mathematical-Models.git
cd Financial-Mathematical-Models
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt     # runtime deps for the library + web app
```

Optional extras:

```bash
pip install -e '.[dev]'             # pytest, ruff, mypy — to run the test suite
pip install -e '.[notebooks]'       # jupyter, matplotlib, seaborn, scikit-learn
```

## Quickstart — the library

```python
from portfolio.data import DataIngestion
from portfolio.models import (
    SingleIndexModel,
    get_cvar,
    get_utility,
    get_var,
    max_utility,
    portfolio_metrics,
)

ingestion = DataIngestion()
prices = ingestion.get_stock_prices()

# Modern Portfolio Theory: long-only max-Sharpe portfolio
metrics = portfolio_metrics(prices, risk_free_rate=0.04)
print(metrics["sharpe_ratio"], metrics["weights"])

# Historical Value at Risk and Expected Shortfall
print(get_var(prices, confidence=0.95))
print(get_cvar(prices, confidence=0.95))

# Mean-variance utility
print(get_utility(prices, risk_aversion=3.0))
print(max_utility(prices, risk_aversion=3.0, risk_free_rate=0.04))

# Single Index Model
sim = SingleIndexModel()
sim.get_models(
    tickers=list(prices.columns),
    market_ticker="^GSPC",                 # add ^GSPC to your universe first
    returns=DataIngestion.compute_returns(prices),
)
print(sim.get_betas())
```

The notebooks in `notebooks/` are short, runnable demos of the public API for
each module — start there if you prefer reading code.

## Quickstart — the web app

The Flask app exposes a single form where the user picks:

- a list of tickers, and/or any of the preset asset classes
  (Stocks, ETFs, Treasury bonds, Crypto) — the merged universe is fed
  into the same Sharpe-ratio optimisation,
- a date range,
- a risk-free rate and risk-aversion coefficient,
- and (optionally) a market proxy ticker for the Single Index Model
  (default `^GSPC`).

On submit the app fetches prices via `yfinance` and renders the
max-Sharpe portfolio (with an interactive Chart.js bar chart of the
optimal weights), historical VaR / CVaR at the 90 %, 95 %, and 99 %
confidence levels, per-asset mean-variance utility, and the Single Index
Model decomposition (alpha, beta, R², systematic / firm-specific / total
risk per asset). Any tickers that yfinance silently drops are listed in
a banner at the top of the results page.

```bash
# Local development
flask --app wsgi:application run --debug
# → http://127.0.0.1:5000/

# Production server
gunicorn --bind 0.0.0.0:8000 --workers 2 wsgi:application
```

### Production checklist

- Set `FLASK_ENV=production`. With this set, the app refuses to start
  unless `FLASK_SECRET_KEY` is also set to a non-default value.
- Generate a secret key with:
  ```bash
  python -c 'import secrets; print(secrets.token_hex(32))'
  ```
- The default rate limit is **120 requests/minute per IP** globally and
  **10 requests/minute on `/analyze`**. `/healthz` is exempt.
- The price cache and rate-limit storage are in-memory and per-process —
  fine for a single `gunicorn` worker; for multi-worker or multi-host
  deploys, point both at Redis.

### Docker

```bash
# Single container
docker build -f docker/Dockerfile -t financial-mathematical-models .
docker run --rm -p 8000:8000 -e FLASK_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')" financial-mathematical-models

# Or via docker compose — requires FLASK_SECRET_KEY in the host env or a .env file
FLASK_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')" docker compose -f docker/docker-compose.yml up --build
```

The app then serves at <http://localhost:8000/>.

## Running the tests

```bash
pip install -e '.[dev]'
pytest                      # or: pytest --cov --cov-report=term-missing
ruff check .                # lint
mypy                        # static type check
```

The test suite is fully offline — every yfinance call is mocked — so it runs
in a few seconds, and covers 100% of the library and web-app code. Continuous
integration runs the same lint / type-check / test steps on Python 3.11–3.13.

## Mathematical background

### Modern Portfolio Theory

For a portfolio with weights $w$, expected returns $\mu$, and covariance
matrix $\Sigma$, the expected return is $w^\top \mu$ and the variance is
$w^\top \Sigma w$. The Sharpe ratio at risk-free rate $r_f$ is

$$ S(w) = \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}. $$

`portfolio.models.mpt.portfolio_metrics` solves $\arg\max_w S(w)$ subject
to $\sum_i w_i = 1$ and $w_i \ge 0$.

### Single Index Model

For asset $i$ and market proxy $m$,

$$ R_i = \alpha_i + \beta_i R_m + \epsilon_i, $$

with $\sigma_i^2 = \beta_i^2 \sigma_m^2 + \sigma_{\epsilon_i}^2$ — the
classic decomposition into systematic and firm-specific risk. Fitted via OLS
in `portfolio.models.SingleIndexModel`.

### Utility theory

Mean-variance utility for risk-aversion $A$:

$$ U = \mathbb{E}[R] - \tfrac{1}{2} A \, \mathrm{Var}[R]. $$

The optimal allocation between cash and a single risky asset is
$y^\star = (\mathbb{E}[r] - r_f) / (A \sigma^2)$ — a textbook result that
`portfolio.models.utility.max_utility` evaluates per asset.

### Value at Risk and CVaR

Both are computed historically: VaR is the $(1 - c)$ percentile of the
empirical return distribution, and CVaR is the mean of returns at or below
that threshold. CVaR is a coherent risk measure; VaR is not.

## Caveats

- **Estimation risk.** Expected returns and covariances are estimated from
  historical data. Past returns are noisy, biased estimators of future
  returns; concentrated max-Sharpe portfolios should not be taken as
  trading advice.
- **Single period.** All models in this repo are single-period. Multi-period
  / dynamic extensions are out of scope.
- **No transaction costs.** Real-world frictions (fees, taxes, slippage)
  are ignored.
- **yfinance is unofficial.** Symbols can be silently renamed, delisted,
  or rate-limited; the app retries transient failures and lists dropped
  tickers in the results banner, but cannot paper over a broken upstream.
- **Educational use only.** This is a teaching repository, not investment
  advice.

## License

MIT
