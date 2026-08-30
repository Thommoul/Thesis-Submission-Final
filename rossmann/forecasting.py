import os

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import DATA_DIR, OUTPUT_DIR


def moving_average(train, window=7, store_id=None, save=True):
    df = train.copy()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values(["Store", "Date"])

    if store_id is not None:
        df = df[df["Store"] == store_id].reset_index(drop=True)

    sales = df["Sales"]

    raw_ma = sales.rolling(window=window).mean()
    scaled_ma = (raw_ma / 1000).round(3)

    error = sales - raw_ma
    abs_error = error.abs()
    sq_error = error ** 2
    ape = (abs_error / sales.replace(0, np.nan)) * 100

    result = pd.DataFrame(
        {
            "Store": df["Store"],
            "Date": df["Date"],
            "Sales": sales,
            f"Moving_Average,n={window}": scaled_ma,
            f"Error_{window}": error,
            f"AbsError_{window}": abs_error,
            f"SqError_{window}": sq_error,
            f"APE_{window} (%)": (ape / 100),
        }
    )

    if save:
        store_str = f"_store{store_id}" if store_id is not None else ""
        path = os.path.join(
            OUTPUT_DIR,
            f"moving_average{window}{store_str}.csv"
        )
        result.to_csv(
            path,
            index=False,
            float_format="%.3f",
            sep=";"
        )
        print(f"Saved moving_average{window}{store_str}.csv")

    return result


def ses_forecast_store_stats(alpha=0.3, store_id=None):
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), low_memory=False)

    if store_id is not None:
        output_excel = os.path.join(OUTPUT_DIR, f"ses_forecast_store{store_id}_stats.xlsx")
    else:
        output_excel = os.path.join(OUTPUT_DIR, "ses_forecast_all_stores_stats.xlsx")

    results_daily = []
    results_stats = []

    stores = [store_id] if store_id is not None else train["Store"].unique()

    for s_id in stores:
        store_data = train[(train["Store"] == s_id) & (train["Open"] == 1)]
        store_data = store_data[store_data["Sales"] > 0].copy()
        store_data["Date"] = pd.to_datetime(store_data["Date"])
        store_data = store_data.sort_values("Date")

        if store_data.empty:
            continue

        sales = store_data["Sales"].values
        ses_model = SimpleExpSmoothing(sales).fit(smoothing_level=alpha, optimized=False)
        ses_values = ses_model.fittedvalues

        error = sales - ses_values
        abs_error = np.abs(error)
        sq_error = error ** 2

        df_daily = pd.DataFrame(
            {
                "Store": s_id,
                "Date": store_data["Date"],
                "Sales": sales,
                "SES_Sales": ses_values,
                "Error": error,
                "AbsError": abs_error,
                "SqError": sq_error,
            }
        )

        mae = np.mean(abs_error)
        mse = np.mean(sq_error)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(error / sales)) * 100
        corr = np.corrcoef(sales, ses_values)[0, 1]

        df_stats = pd.DataFrame(
            {
                "Store": [s_id],
                "MAE": [mae],
                "MSE": [mse],
                "RMSE": [rmse],
                "MAPE": [mape],
                "Correlation": [corr],
            }
        )

        results_daily.append(df_daily)
        results_stats.append(df_stats)

    ses_all_daily = pd.concat(results_daily, ignore_index=True)
    ses_all_stats = pd.concat(results_stats, ignore_index=True)

    with pd.ExcelWriter(output_excel) as writer:
        ses_all_daily.to_excel(writer, sheet_name="Daily", index=False)
        ses_all_stats.to_excel(writer, sheet_name="Stats", index=False)

    print(f"Saved: {output_excel}")

    return ses_all_daily, ses_all_stats


def _fit_sarimax_forecast(ts, forecast_days):
    model = SARIMAX(
        ts,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)

    future_dates = pd.date_range(start=ts.index.max() + pd.Timedelta(days=1), periods=forecast_days)
    forecast_values = fit.forecast(forecast_days)
    return future_dates, forecast_values


def forecast_sarimax(year=2015, forecast_days=30):
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), low_memory=False)
    train["Date"] = pd.to_datetime(train["Date"], dayfirst=True)
    train = train[train["Date"].dt.year == year]

    results_daily = []
    failed_stores = []

    for s in train["Store"].unique():
        store_data = train[train["Store"] == s].sort_values("Date")
        ts = store_data.set_index("Date")["Sales"]

        if len(ts) < 30:
            continue

        try:
            future_dates, forecast_values = _fit_sarimax_forecast(ts, forecast_days)
            for d, val in zip(future_dates, forecast_values):
                results_daily.append({"Store": s, "Forecast_Date": d, "Forecast_Sales": round(val, 2)})
        except Exception as exc:
            failed_stores.append((s, exc))
            print(f"  Store {s}: SARIMAX fit failed ({exc!r}), forecast filled with NaN.")
            future_dates = pd.date_range(start=ts.index.max() + pd.Timedelta(days=1), periods=forecast_days)
            for d in future_dates:
                results_daily.append({"Store": s, "Forecast_Date": d, "Forecast_Sales": np.nan})

    forecast_daily_df = pd.DataFrame(results_daily)

    save_path = os.path.join(OUTPUT_DIR, f"forecast_daily_30days_{year}.xlsx")
    forecast_daily_df.to_excel(save_path, index=False)

    print(f"Saved forecast to {save_path}")
    if failed_stores:
        print(f"{len(failed_stores)} store(s) failed to fit and were filled with NaN: "
              f"{[s for s, _ in failed_stores]}")
    print(forecast_daily_df.head(20))

    return forecast_daily_df


def forecast_store_sales_per_year(year=2015, min_length=30, forecast_days=30):
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), low_memory=False)
    train["Date"] = pd.to_datetime(train["Date"], dayfirst=True)
    train = train[train["Date"].dt.year == year]

    if train.empty:
        print(f"No data found for year {year}.")
        return None

    output_excel = os.path.join(OUTPUT_DIR, f"store_sales_actual_plus_forecast_{year}.xlsx")

    results = []
    failed_stores = []

    for s in train["Store"].unique():
        store_data = train[train["Store"] == s].sort_values("Date")
        ts = store_data.set_index("Date")["Sales"]

        if len(ts) < min_length:
            continue

        try:
            future_dates, forecast_values = _fit_sarimax_forecast(ts, forecast_days)

            actual_df = store_data[["Date", "Sales"]].copy()
            actual_df["Forecast_Sales"] = np.nan
            actual_df.rename(columns={"Sales": f"Actual_Sales_{year}"}, inplace=True)

            forecast_df = pd.DataFrame(
                {
                    "Date": future_dates,
                    f"Actual_Sales_{year}": np.nan,
                    "Forecast_Sales": forecast_values.values,
                }
            )

            combined = pd.concat([actual_df, forecast_df], ignore_index=True)
            combined["Store"] = s
            results.append(combined)
        except Exception as exc:
            failed_stores.append((s, exc))
            print(f"  Store {s}: SARIMAX fit failed ({exc!r}), store skipped.")
            continue

    if not results:
        print("No stores produced results.")
        return None

    final_df = pd.concat(results, ignore_index=True)
    final_df.to_excel(output_excel, index=False)
    print(f"Saved: {output_excel}")
    if failed_stores:
        print(f"{len(failed_stores)} store(s) failed to fit and were skipped: "
              f"{[s for s, _ in failed_stores]}")
    print(final_df.head(30))
    return final_df


def forecast_store_sales_per_month(year=2015, month=1):
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), low_memory=False)
    train["Date"] = pd.to_datetime(train["Date"], dayfirst=True)

    filtered = train[(train["Date"].dt.year == year) & (train["Date"].dt.month == month)].copy()

    if filtered.empty:
        print(f"No data found for {year}-{month:02d}.")
        return None

    filtered = filtered[["Store", "Date", "Sales"]]
    filtered.rename(columns={"Sales": f"Actual_Sales_{year}_{month:02d}"}, inplace=True)

    output_excel = os.path.join(OUTPUT_DIR, f"actual_store_sales_{year}_{month:02d}.xlsx")
    filtered.to_excel(output_excel, index=False)

    print(f"Saved: {output_excel}")
    print(filtered.head(30))

    return filtered
