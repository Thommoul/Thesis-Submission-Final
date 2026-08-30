# Rossmann Store Sales — Demand Forecasting Thesis

Diploma thesis project (Thomas Mouliaras) applying classical time-series methods and
machine learning to the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
dataset, comparing their accuracy at forecasting daily store sales.

## What it does

The project loads the Rossmann drug-store sales data, cleans/merges it with store
metadata, and runs several forecasting and analysis techniques per store or across
all stores:

- **Moving Average** — rolling-window baseline forecast
- **Simple Exponential Smoothing (SES)** — smoothed forecast with error statistics
- **SARIMAX** — seasonal ARIMA forecast for a given year/horizon
- **Linear Regression** — regression-based sales model with feature preprocessing
- **XGBoost** — gradient-boosted regression per store and across all stores,
  automatically trained on a CUDA GPU if one is available, otherwise CPU
- **Backtesting** — rolling 30-day backtest evaluation of a trained model
- **Sales analytics** — closed-day detection, average sales, sales density/growth
  plots, zero-sales reporting
- **Excel reporting** — writes results/forecasts to `.xlsx` files with embedded
  charts (via `openpyxl`)

Each function saves its output (CSV/XLSX and plots) to the `outputs/` folder.

## Project layout

```
data/                  Input datasets (train.csv, test.csv, store.csv)
docs/                  Architecture, methods, and thesis review notes (+ el/ Greek translations)
outputs/               Generated forecasts, reports, and charts (gitignored)
rossmann/              Package with all pipeline logic
  config.py              Paths (data/output dirs, model path)
  data.py                load_data, preprocess_data, load_store_history
  features.py            Shared calendar feature engineering for the ML models
  forecasting.py         Moving average, SES, SARIMAX forecasts
  regression.py          OLS regression and per-store/per-month linear models
  ml.py                  XGBoost training, backtesting, all-stores model
  device.py              Detects a usable CUDA device, falls back to CPU
  analytics.py           Sales/closed-day/zero-sales reports
  plots.py               Sales density and growth plots
run.py                 CLI entry point — runs one, several, or all steps
rossmann_model_store{id}.pkl  Per-store XGBoost model, written by xgboost_train
rossmann_model.pkl     Legacy pre-trained model, used as a fallback for store 1
requirements.txt       Python dependencies
```

## Dataset

Place the Rossmann competition files in `data/`:

- `train.csv`
- `test.csv`
- `store.csv`

(Already included in this repo under `data/`.) Source:
[Kaggle — Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales).

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

No extra setup is needed for GPU training — the `xgboost` wheel bundles CUDA
support. If a working NVIDIA GPU/driver is detected at runtime, XGBoost steps
(`xgboost_train`, `xgboost_single_store`, `xgboost_all_stores`) train on the
GPU automatically; otherwise they silently fall back to CPU.

## Run

The entry point is `run.py`. By default it loads/preprocesses the data and runs
a moving-average forecast for store 1:

```bash
python run.py
```

List all available steps:

```bash
python run.py --list
```

Run one or more specific steps:

```bash
python run.py --steps sarimax ols_regression xgboost_train backtest
```

Run the entire pipeline:

```bash
python run.py --all
```

Note: `backtest` loads the model saved by `xgboost_train` **for the same
`--store` id**, so run `xgboost_train --store N` at least once for a given
store before backtesting it (`--all` handles ordering automatically, but only
for the default store). Store 1 additionally falls back to the legacy
`rossmann_model.pkl` if no per-store file has been trained yet.

Individual functions can also be imported and called directly, e.g.:

```python
from rossmann.data import load_data, preprocess_data
from rossmann import forecasting, ml

store, test, train = load_data()
store, test, train = preprocess_data(store, test, train)
forecasting.forecast_sarimax(year=2015, forecast_days=30)
ml.train_xgboost_model(store_id=1)
```

## Output

Results are written to `outputs/`, including per-store forecast CSVs, Excel
workbooks with charts (e.g. `xgboost_store_1_results.xlsx`,
`backtest_dashboard_store_1.xlsx`), and matplotlib plots.

## Documentation

`docs/` has more detail than this README, current as of the actual code
(not the thesis narrative):

- [`01_code_architecture.md`](docs/01_code_architecture.md) — module-by-module
  breakdown of `rossmann/` and `run.py`, including known quirks (e.g.
  `Date` becomes a string after `preprocess_data`, `regression.py` re-implements
  its own feature block instead of importing `features.py`)
- [`02_forecasting_methods.md`](docs/02_forecasting_methods.md) — how each
  forecasting method works
- [`03_methods_comparison.md`](docs/03_methods_comparison.md) — accuracy/tradeoff
  comparison across methods
- [`04_thesis_review_notes.md`](docs/04_thesis_review_notes.md) — where the
  code and the thesis write-up (`Διπλωματική Μούλιαρας Θωμάς.docx`) diverge

Greek translations of the same docs are in `docs/el/`.

## Author

Thomas Mouliaras
