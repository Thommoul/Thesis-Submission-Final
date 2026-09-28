import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import DATA_DIR, OUTPUT_DIR
from .data import load_data
from .features import add_calendar_features, chronological_split


def run_rossmann_regression(train_path=None, store_path=None, plot=True):
    train_path = train_path or os.path.join(DATA_DIR, "train.csv")
    store_path = store_path or os.path.join(DATA_DIR, "store.csv")

    train = pd.read_csv(train_path, parse_dates=["Date"])
    store = pd.read_csv(store_path)

    df = train.merge(store, on="Store", how="left")
    df = df[df["Open"] == 1].copy()

    df["Year"] = df["Date"].dt.year

    df = add_calendar_features(df)
    df["Promo2"] = df["Promo2"].fillna(0)

    features = [
        "Month",
        "DayOfWeek",
        "Promo",
        "SchoolHoliday",
        "CompetitionDistance",
        "IsSummer",
        "IsChristmas",
        "IsWeekend",
    ]


    df_train, df_test = chronological_split(df)

    X_train = sm.add_constant(df_train[features])
    X_test = sm.add_constant(df_test[features])
    y_train = df_train["Sales"]
    y_test = df_test["Sales"]

    model = sm.OLS(y_train, X_train).fit()
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n===== REGRESSION RESULTS =====\n")
    print(model.summary())
    print("\nRMSE:", rmse)

    if plot:
        print("\nCreating seasonality graphs...")

        plt.figure()
        df.groupby("Month")["Sales"].mean().plot(kind="line")
        plt.title("Average Sales by Month")
        plt.xlabel("Month")
        plt.ylabel("Average Sales")
        plt.show()

        plt.figure()
        df.groupby("DayOfWeek")["Sales"].mean().plot(kind="bar")
        plt.title("Average Sales by Day of Week")
        plt.xlabel("Day of Week")
        plt.ylabel("Average Sales")
        plt.show()

        plt.figure()
        df.groupby("IsChristmas")["Sales"].mean().plot(kind="bar")
        plt.title("Sales Comparison: Christmas vs Normal Period")
        plt.xlabel("Christmas Period")
        plt.ylabel("Average Sales")
        plt.show()

    sample = X_test.iloc[0:1]
    print("\nExample prediction:", model.predict(sample).values)

    return model, rmse


def predict_sales_filtered(year, month=None):
    """Fit a per-month DayOfWeek/Promo -> Sales linear model and write predictions to Excel."""
    _, _, train = load_data()
    data = train.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    output_file = os.path.join(OUTPUT_DIR, f"predictions_{year}.xlsx")

    if month:
        months_to_process = [month]
    else:
        months_to_process = sorted(data[data["Date"].dt.year == year]["Date"].dt.month.unique())

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for m in months_to_process:
            month_data = data[
                (data["Date"].dt.year == year) & (data["Date"].dt.month == m)
            ][["Date", "DayOfWeek", "Promo", "Sales"]].dropna()

            if month_data.empty:
                continue

            train_data, test_data = chronological_split(month_data)

            X_train, X_test = train_data[["DayOfWeek", "Promo"]], test_data[["DayOfWeek", "Promo"]]
            y_train, y_test = train_data["Sales"], test_data["Sales"]

            model = LinearRegression()
            model.fit(X_train, y_train)
            mae = mean_absolute_error(y_test, model.predict(X_test))

            predictions = [
                {
                    "DayOfWeek": day,
                    "Promo": promo,
                    "PredictedSales": model.predict([[day, promo]])[0],
                    "MeanAbsoluteError": mae,
                }
                for day in range(1, 8)
                for promo in [0, 1]
            ]

            pd.DataFrame(predictions).to_excel(writer, sheet_name=f"Month_{m}", index=False)

    print(f"Saved: {output_file}")


def predict_sales_for_store(store_id, year=None, month=None):
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train["Date"] = pd.to_datetime(train["Date"], dayfirst=True)

    store_df = train[(train["Store"] == store_id) & (train["Open"] == 1)].copy()

    if year:
        store_df = store_df[store_df["Date"].dt.year == year]
    if month:
        if not year:
            print("A year is required to filter by month.")
            return None
        store_df = store_df[store_df["Date"].dt.month == month]

    if store_df.empty:
        print(f"No data found for Store {store_id} with the given filters.")
        return None


    train_data, test_data = chronological_split(store_df)
    X_train, X_test = train_data[["DayOfWeek", "Promo"]], test_data[["DayOfWeek", "Promo"]]
    y_train, y_test = train_data["Sales"], test_data["Sales"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    print(f"===== Store {store_id} Linear Regression =====")
    print("Mean Absolute Error (MAE):", round(mae, 2))
    print("Root Mean Squared Error (RMSE):", round(rmse, 2))

    store_df = store_df[["DayOfWeek", "Promo", "Sales"]].copy()
    store_df["PredictedSales"] = model.predict(store_df[["DayOfWeek", "Promo"]]).round(2)
    store_df.rename(columns={"Sales": "ActualSales"}, inplace=True)

    if year and month:
        output_path = os.path.join(OUTPUT_DIR, f"predictions_store{store_id}_{year}_{month:02d}.xlsx")
    elif year:
        output_path = os.path.join(OUTPUT_DIR, f"predictions_store{store_id}_{year}.xlsx")
    else:
        output_path = os.path.join(OUTPUT_DIR, f"predictions_store{store_id}.xlsx")

    store_df.to_excel(output_path, index=False)
    print(f"\nPredictions saved to Excel at: {output_path}")

    return store_df


def analyze_sales_factors():

    train_path = os.path.join(DATA_DIR, "train.csv")
    store_path = os.path.join(DATA_DIR, "store.csv")

    train = pd.read_csv(train_path)
    store = pd.read_csv(store_path)



    train["Date"] = pd.to_datetime(train["Date"])



    data = train.merge(store, on="Store", how="left")



    data = data[data["Open"] == 1].copy()


    data["DayOfWeek"] = data["Date"].dt.dayofweek + 1
    data["Month"] = data["Date"].dt.month


    data["IsChristmas"] = (
        (data["Date"].dt.month == 12) &
        (data["Date"].dt.day >= 20)
    ).astype(int)


    data = data[
        [
            "Sales",
            "Promo",
            "DayOfWeek",
            "Month",
            "IsChristmas"
        ]
    ].dropna()


    data = pd.get_dummies(
        data,
        columns=["DayOfWeek", "Month"],
        drop_first=True,
        dtype=int
    )


    y = data["Sales"]

    X = data.drop(columns=["Sales"])

    # Προσθήκη σταθερού όρου
    X = sm.add_constant(X)


    model = sm.OLS(y, X).fit()


    results = pd.DataFrame({
        "Μεταβλητή": model.params.index,
        "Συντελεστής": model.params.values,
        "p-value": model.pvalues.values
    })

    results["Σημαντικότητα"] = results["p-value"].apply(
        lambda x:
        "Στατιστικά σημαντική" if x < 0.05
        else "Μη στατιστικά σημαντική"
    )


    model_stats = pd.DataFrame({
        "Δείκτης": [
            "R²",
            "Adjusted R²",
            "F-statistic",
            "Prob (F-statistic)",
            "Αριθμός παρατηρήσεων"
        ],
        "Τιμή": [
            model.rsquared,
            model.rsquared_adj,
            model.fvalue,
            model.f_pvalue,
            int(model.nobs)
        ]
    })


    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        "sales_factors_regression.xlsx"
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        results.to_excel(
            writer,
            sheet_name="Regression Results",
            index=False
        )

        model_stats.to_excel(
            writer,
            sheet_name="Model Statistics",
            index=False
        )

    print("\n" + "=" * 70)
    print("ΑΝΑΛΥΣΗ ΕΠΙΔΡΑΣΗΣ ΠΑΡΑΓΟΝΤΩΝ ΣΤΙΣ ΠΩΛΗΣΕΙΣ")
    print("=" * 70)

    print("\nΣτατιστικά μοντέλου:")
    print(model_stats.to_string(index=False))

    print("\n\nΣυντελεστές παλινδρόμησης:")
    print(results.to_string(index=False))

    print("\n\nΑναλυτικά αποτελέσματα:")
    print(model.summary())

    print("\nΤο αρχείο αποθηκεύτηκε στο:")
    print(output_path)

    return model, results, model_stats