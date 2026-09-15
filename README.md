# Project Sigma — Cross-Sectional Factor Research Pipeline (V1)

A research prototype for building, estimating, and backtesting a daily cross-sectional equity factor model.

This repository is the **first version of Project Sigma**. It was built to understand the full research pipeline end-to-end: market/fundamental data ingestion, factor construction, cross-sectional regression, factor-return forecasting, portfolio construction, backtesting, and paper execution.

The most important outcome of V1 was not the backtest performance itself, but identifying how strongly the result depends on **data quality, point-in-time correctness, survivorship bias, delisting treatment, and market-regime stability**.

Because those issues are not fully resolved in the original Massive/Polygon-based dataset, historical performance from this repository is **not treated as reliable evidence of an investable strategy**. Current work is focused on rebuilding the research dataset from WRDS CRSP/Compustat before evaluating the model again.

## Research Motivation

The initial question was simple:

> Can a set of cross-sectional equity factors be combined into a systematic daily ranking and portfolio construction framework?

V1 implements that question as a complete pipeline and then uses the resulting backtests as a diagnostic tool. During development, several problems became more important than improving the model itself:

- Is the historical stock universe genuinely point-in-time?
- Are delisted securities and delisting returns handled correctly?
- Are fundamentals available only from the date they could actually have been known?
- Are unusually high backtest results caused by the model, or by the dataset and simulation assumptions?
- Do factor relationships remain stable when the market regime changes?
- How sensitive are results to turnover, slippage, execution timing, and factor specification?

These questions now define the direction of Project Sigma V2.

## Current Status

**V1 — this repository**

- Massive/Polygon-compatible market and fundamental data
- Daily cross-sectional factor exposures
- Ridge-regularized linear regression for factor-return estimation
- Rolling / EWMA factor-return forecasts
- Long-short portfolio construction
- Close-to-next-open backtesting
- Transaction-cost and slippage assumptions
- Residual, turnover, exposure, and execution diagnostics
- IBKR paper-execution tooling

**V2 — current work**

The dataset is being rebuilt using **WRDS CRSP and Compustat** with a stronger emphasis on research validity:

```text
RAW
  ↓
NORMALIZED
  ↓
POINT-IN-TIME / DERIVED
  ↓
FACTOR
  ↓
BACKTEST
```

The goal is to make historical universe membership, identifiers, delistings, corporate events, fundamental availability, and look-ahead controls explicit before evaluating strategy performance.

## Model Overview

For each trading date `t`, the cross-sectional return model is

```text
r_t = X_t f_t + ε_t
```

where

- `r_t ∈ R^N` — forward returns across stocks
- `X_t ∈ R^(N×K)` — cross-sectional factor exposure matrix
- `f_t ∈ R^K` — estimated factor returns
- `ε_t` — residual returns

The pipeline is organized as:

```text
Market + Fundamental Data
          ↓
Raw Factors
          ↓
Point-in-time availability rules
          ↓
Cross-sectional winsorization / z-score
          ↓
Exposure Matrix X[T, N, K]
          ↓
Cross-sectional Ridge Regression
          ↓
Historical Factor Returns f[T, K]
          ↓
Rolling / EWMA Forecast
          ↓
Stock Scores
          ↓
Long / Short Portfolio
          ↓
Backtest + Diagnostics
```

## Factor-Return Estimation

V1 estimates factor returns cross-sectionally for each date.

### Ridge regression

```text
f_t = argmin_f ||r_t - X_t f||² + λ||f||²
```

Equivalent normal-equation form:

```text
f_t = (X_t'X_t + λI)^(-1) X_t'r_t
```

The implementation supports both fixed ridge regularization and generalized cross-validation (GCV) for selecting `λ`. The default numerical path can solve the augmented least-squares problem with QR rather than explicitly inverting `X'X`.

This was added because factor exposures can be highly correlated and the daily cross-sectional design matrix can become ill-conditioned.

## Factor Exposure Processing

For each date and factor:

1. Apply the available universe / tradability mask.
2. Preserve natural missing values during raw factor construction.
3. Winsorize the cross-section at configurable quantiles.
4. Z-score the factor cross-sectionally.
5. Drop factors with insufficient finite coverage.
6. Remove near-duplicate factors above a configurable correlation threshold.
7. Optionally map remaining missing standardized exposures to zero, interpreted as neutral exposure.

The resulting tensor has shape:

```text
X[T, N, K]
```

with trading dates `T`, securities `N`, and factors `K`.

## Factor Set

The implemented factor library includes price/volume and fundamental signals.

### Price / volume examples

- Momentum
- Realized volatility
- Parkinson volatility
- Intraday / overnight return
- Distance from 52-week range
- Liquidity
- Amihud-style illiquidity proxy
- Return skewness / kurtosis
- Volume momentum

### Fundamental examples

**Valuation**

- Earnings yield
- Book-to-market
- Sales-to-price
- Cash-flow-to-price
- Free-cash-flow yield

**Profitability / quality**

- ROE
- ROA
- Gross margin
- Operating margin
- Net margin
- Asset turnover

**Balance sheet / growth**

- Debt-to-equity
- Current ratio
- Cash-to-assets
- Revenue growth
- Net-income growth

Not every factor is available for every security/date. Coverage is tracked explicitly during preprocessing.

## Forecasting and Portfolio Construction

Historical daily factor returns can be converted into a next-period estimate using:

- latest observation
- rolling historical mean
- exponentially weighted moving average (EWMA)

The score for stock `n` is approximately

```text
alpha[t,n] = X[t,n,:] @ f_pred[t+1,:]
```

Stocks are ranked cross-sectionally and used to form configurable long/short quantile portfolios.

The current forecasting methods are deliberately simple. A major open question is whether factor-return dynamics should be conditioned on **market regimes** rather than assuming a stable relationship through time.

## Backtesting Convention

The canonical daily workflow is:

```text
close[t] information
    ↓
after-close signal generation
    ↓
target portfolio saved
    ↓
next trading day open execution
```

Signals are not recomputed using future or intraday information after the signal date.

The backtesting code tracks items such as:

- portfolio return
- gross exposure
- turnover
- transaction-cost assumptions
- slippage assumptions
- return coverage
- long / short counts
- residual diagnostics
- factor contributions

## Data Integrity and Bias Controls

V1 contains several controls intended to reduce common backtesting errors, but they should **not be interpreted as proving the dataset is fully point-in-time or survivorship-free**.

### Fundamental availability

Financial statement rows use their filing date when available. If a filing date is unavailable, V1 can fall back to an assumed lag from the reporting-period end date.

Fundamental values are then forward-filled only after their assumed availability date.

This reduces obvious look-ahead leakage, but an assumed reporting lag is still an approximation and is one reason the dataset is being rebuilt.

### Universe / survivorship handling

The pipeline can request inactive as well as active ticker metadata and uses a date-by-security tradability mask during preprocessing and regression.

However, this does **not** guarantee a historically exact exchange-membership universe. Delisting-return handling and vendor-backed historical security membership are incomplete in V1.

### Return and execution timing

Forward returns and signal dates are separated explicitly, and the close-to-next-open workflow is tested to prevent accidental same-period execution.

## Why the Original Backtest Is Not Presented as a Result

Earlier versions of this repository produced unusually strong risk-adjusted performance in historical backtests.

Rather than treating those numbers as evidence of alpha, V1 now treats them as a **validation warning**. The result may be affected by unresolved issues including:

- incomplete historical universe reconstruction
- survivorship effects
- incomplete delisting treatment
- approximate point-in-time fundamental availability
- corporate-action assumptions
- model-selection and specification risk
- simplified execution and transaction-cost assumptions
- changing factor behavior across market regimes

For that reason, headline Sharpe or return numbers are intentionally not presented here as strategy performance.

The strategy will be evaluated again after the WRDS-based dataset and validation pipeline are complete.

## Diagnostics

The project includes diagnostics for inspecting whether the regression and backtest behave as expected.

Examples include:

- selected ridge parameter by date
- number of securities entering each regression
- factor coverage
- high-correlation factor removal
- residual heatmaps
- daily cross-sectional MSE
- large residual events
- return coverage
- turnover
- exposure
- cost-adjusted alpha thresholds

Residual diagnostics can be generated with:

```bash
python scripts/plot_residual_heatmap.py \
  --input-dir data/nasdaq_full \
  --clip-percentile 99.5 \
  --sort-tickers-by coverage \
  --top-tickers 50 \
  --top-events 100
```

## Paper Execution

V1 also contains an IBKR paper-execution layer. This exists mainly to test whether a research signal can be translated into a reproducible order workflow.

The daily execution path loads a saved after-close order plan and submits or monitors orders on the next trading session. Legacy intraday signal recomputation is disabled by default.

Paper-execution functionality should be considered an engineering prototype rather than evidence that the research model is ready for live capital.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set a Massive-compatible API key:

```bash
export MASSIVE_API_KEY="your_key"
```

## Minimal Research Run

```bash
python scripts/run_clean_pipeline.py \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --max-tickers 30 \
  --out-dir data/test_clean
```

A larger run can use the broader historical ticker set:

```bash
python scripts/run_clean_pipeline.py \
  --start 2023-01-01 \
  --end 2026-01-01 \
  --ticker-status all \
  --out-dir data/nasdaq_full
```

Run the basic factor backtest:

```bash
python scripts/run_backtest.py \
  --input-dir data/nasdaq_full \
  --method ewma \
  --lookback 120 \
  --quantile 0.10 \
  --ewma-halflife 20
```

Run the close-to-next-open simulation:

```bash
python scripts/run_close_to_next_open_backtest.py \
  --config configs/daily_factor_backtest.yaml
```

## Main Outputs

```text
X.npy                         # T x N x K exposure tensor
r.npy                         # T x N forward returns
factor_returns.npy            # T x K estimated factor returns
factor_names.csv              # factor names
tickers.csv                   # securities
dates.csv                     # trading dates
tradable_mask.npy             # T x N universe/tradability mask
factor_diagnostics_*.csv      # preprocessing diagnostics
financials_flat.parquet       # flattened filing rows
financials_ttm.parquet        # point-in-time-style TTM fields
pipeline_summary.json         # run metadata and parameters
```

## Tests

```bash
pytest -q
```

The test suite covers core components including:

- factor preprocessing
- ridge estimation and diagnostics
- dynamic factor construction
- fundamental-factor construction
- close-to-next-open timing
- backtest workflow
- report generation
- paper-execution logic

## Project Structure

```text
factor_pipeline/
├── backtest.py
├── signals.py
├── estimation.py
├── preprocess.py
├── fundamental_factors.py
├── price_volume_factors.py
├── massive_client.py
└── ...

backtests/
configs/
docs/
live/
reports/
scripts/
tests/
tools/
```

## Main Lessons from V1

Project Sigma V1 started as a modeling project, but the main lesson was that a convincing quantitative result depends at least as much on **dataset construction and validation** as on the forecasting model.

The current priority is therefore not to add a more complicated model to an uncertain dataset. It is to establish a research dataset whose historical information set can be defended, then re-test simple models before increasing complexity.

## References

- Fama, E. F., & MacBeth, J. D. (1973). *Risk, Return, and Equilibrium: Empirical Tests*. Journal of Political Economy, 81(3), 607–636.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Blitz, D., Hanauer, M. X., & Vidojevic, M. (2019). *The Idiosyncratic Momentum Anomaly*. International Review of Economics & Finance.

---

**Status:** V1 research prototype / validation stage.  
**Current direction:** WRDS CRSP + Compustat point-in-time dataset reconstruction and backtest validation.
