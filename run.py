"""Entry point for the Rossmann Store Sales forecasting pipeline.

Run all analyses:               python run.py --all
Run one/more steps:             python run.py --steps moving_average sarimax xgboost moving_average_open
Run one step out of the list:   python -c "from rossmann.plots import plot_store_1_sales; plot_store_1_sales();"
List available steps:           python run.py --list
"""

import argparse

from rossmann import analytics, forecasting, ml, plots, regression
from rossmann.data import load_data, preprocess_data

STEPS = {}


def step(name):
    def register(fn):
        STEPS[name] = fn
        return fn
    return register

@step("plot_store_1_sales")
def _(**_kw):
    plots.plot_store_1_sales()

@step("sales_by_day_report")
def _(**_kw):
    analytics.print_sales_by_day_report()

@step("sales_by_day_for_full_years")
def _(**_kw):
    result = analytics.sales_by_day_for_full_years()
    print(result.to_string())

@step("moving_average_open_4")
def _(train, **_kw):
    forecasting.moving_average_open(
        train,
        window=4,
        store_id=None,
        save=True
    )

@step("moving_average_open_7")
def _(train, **_kw):
    forecasting.moving_average_open(
        train,
        window=7,
        store_id=1,
        save=True
    )

@step("moving_average_open_traction")
def _(train, **_kw):
    forecasting.moving_average_open_traction(
        train,
        window=7,
        store_id=1,
        save=True,
        test_fraction=0.2
    )

@step("ses")
def _(store_id=1, **_kw):
    forecasting.ses_forecast_store_stats(
        alpha=0.8,
        store_id=store_id
    )

@step("ses_forecast_store_stats_alpha_validation")
def _(train=None, store_id=1, **kwargs):
    return forecasting.ses_forecast_store_stats_alpha_validation(
        store_id=store_id
    )

@step("ses_alpha_validation_traction")
def _(train, **_kw):
    forecasting.ses_forecast_store_stats_alpha_validation_traction(
        store_id=1,
        test_fraction=0.2
    )
    
@step("ols_regression")
def _(**_kw):
    regression.run_rossmann_regression()


@step("xgboost_train")
def _(store_id=1, **_kw):
    ml.train_xgboost_model(store_id)


@step("backtest")
def _(store_id=1, **_kw):
    ml.backtest_last_30_days(store_id)


DEFAULT_STEPS = ["moving_average"]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all steps"
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        metavar="STEP",
        help="Run selected steps"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available steps"
    )

    parser.add_argument(
        "--store",
        type=int,
        default=1,
        help="Store ID"
    )

    parser.add_argument(
        "--window",
        type=int,
        default=7,
        help="Moving Average window"
    )

    args = parser.parse_args()

    if args.list:
        for name in STEPS:
            print(name)
        return

    if args.all:
        steps_to_run = list(STEPS)

    elif args.steps:

        unknown = [
            s for s in args.steps
            if s not in STEPS
        ]

        if unknown:
            parser.error(
                f"Unknown step(s): {', '.join(unknown)}"
            )

        steps_to_run = args.steps

    else:
        steps_to_run = DEFAULT_STEPS

    store, test, train = load_data()
    store, test, train = preprocess_data(
        store,
        test,
        train
    )

    for name in steps_to_run:

        print(f"\n>>> Running: {name}")

        STEPS[name](
            train=train,
            store_id=args.store,
            window=args.window
        )


if __name__ == "__main__":
    main()