import os

import joblib
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from .config import DATA_DIR, MODEL_PATH, OUTPUT_DIR, model_path
from .data import load_store_history
from .device import get_xgb_device
from .features import FEATURES, add_calendar_features

def _make_xgb_regressor(**params):
    device = get_xgb_device()
    if device == "cuda":
        print("Using CUDA for XGBoost training.")
    return XGBRegressor(tree_method="hist", device=device, random_state=42, **params)


def _train_test_split_by_time(df, features, target_col="Sales", test_fraction=0.2, log_target=True):
    df = add_calendar_features(df)

    y = np.log1p(df[target_col]) if log_target else df[target_col]
    X = df[features]

    split = int(len(df) * (1 - test_fraction))
    return df, X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:], split


def run_store_full_analysis(store_id):
    df = load_store_history(store_id)

    if len(df) < 50:
        raise ValueError("Not enough data for analysis")

    df, X_train, X_test, y_train, y_test, split = _train_test_split_by_time(df, FEATURES)

    model = _make_xgb_regressor(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
    )
    model.fit(X_train, y_train)

    y_pred = np.expm1(model.predict(X_test))
    y_true = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    results = pd.DataFrame({
        "Date": df.iloc[split:]["Date"].values,
        "Actual_Sales": y_true.values,
        "Predicted_Sales": y_pred,
    })

    print("\n===== STORE FULL ANALYSIS =====")
    print("Store:", store_id)
    print("RMSE:", rmse)
    print("\nSample results:")
    print(results.head(10))

    return model, results, rmse


def train_xgboost_model(store_id):
    """Train and persist an XGBoost model for a single store; also used by backtest_last_30_days."""
    df = load_store_history(store_id)

    if len(df) < 50:
        raise ValueError("Not enough data for analysis")

    df, X_train, X_test, y_train, y_test, split = _train_test_split_by_time(df, FEATURES)

    model = _make_xgb_regressor(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
    )
    model.fit(X_train, y_train)

    joblib.dump((model, None, FEATURES), model_path(store_id))

    y_pred = np.expm1(model.predict(X_test))
    y_true = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    results = pd.DataFrame({
        "Date": df.iloc[split:]["Date"].values,
        "Actual_Sales": y_true.values,
        "Predicted_Sales": y_pred,
        "Error": y_true.values - y_pred,
    })

    full_path = os.path.join(OUTPUT_DIR, f"xgboost_store_{store_id}_results.xlsx")
    results.to_excel(full_path, index=False)

    print("\n===== XGBOOST MODEL =====")
    print("Store:", store_id)
    print("RMSE:", rmse)
    print("Saved:", full_path)

    return model, results, rmse


def xgboost_all_stores_model_detailed():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["Date"])
    store = pd.read_csv(os.path.join(DATA_DIR, "store.csv"))

    df = train.merge(store, on="Store", how="left")
    df = df[df["Open"] == 1].copy()
    df = df.sort_values(["Store", "Date"])
    df = add_calendar_features(df)
    df["Store"] = df["Store"].astype(int)

    features = ["Store"] + FEATURES
    X = df[features]
    y = np.log1p(df["Sales"])

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    test_meta = df.iloc[split:][["Store", "Date"]].copy()

    model = _make_xgb_regressor(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
    )
    model.fit(X_train, y_train)

    preds = np.expm1(model.predict(X_test))
    true = np.expm1(y_test)

    results = pd.DataFrame({
        "Store": test_meta["Store"].values,
        "Date": test_meta["Date"].values,
        "Actual_Sales": true.values,
        "Predicted_Sales": preds,
        "Error": true.values - preds,
        "Abs_Error": np.abs(true.values - preds),
    })

    store_summary = results.groupby("Store").agg(
        MAE=("Abs_Error", "mean"),
        RMSE=("Error", lambda x: np.sqrt(np.mean(x ** 2))),
        Avg_Actual=("Actual_Sales", "mean"),
        Avg_Predicted=("Predicted_Sales", "mean"),
    ).reset_index()

    best_store = store_summary.sort_values("MAE").head(5)
    worst_store = store_summary.sort_values("MAE", ascending=False).head(5)

    output_file = os.path.join(OUTPUT_DIR, "xgboost_all_stores_detailed.xlsx")
    with pd.ExcelWriter(output_file) as writer:
        results.to_excel(writer, sheet_name="Global_Predictions", index=False)
        store_summary.to_excel(writer, sheet_name="Store_Summary", index=False)
        best_store.to_excel(writer, sheet_name="Best_Stores", index=False)
        worst_store.to_excel(writer, sheet_name="Worst_Stores", index=False)

    rmse_global = np.sqrt(mean_squared_error(true, preds))

    print("\n===== XGBOOST GLOBAL DSS MODEL (DETAILED) =====")
    print("Saved:", output_file)
    print("Global RMSE:", round(rmse_global, 2))
    print("\nBest performing stores (lowest MAE):")
    print(best_store)
    print("\nWorst performing stores (highest MAE):")
    print(worst_store)

    return model, results, store_summary


def backtest_last_30_days(store_id):
    path = model_path(store_id)
    if not os.path.exists(path):
        # Fall back to the legacy single-model file, which was pre-trained for store 1.
        if store_id == 1 and os.path.exists(MODEL_PATH):
            path = MODEL_PATH
        else:
            raise FileNotFoundError(
                f"No trained model found for store {store_id} at {path}. "
                f"Run train_xgboost_model({store_id}) first."
            )

    model, _le, features = joblib.load(path)

    df = load_store_history(store_id)
    test_df = add_calendar_features(df.tail(30))

    y_true = test_df["Sales"].values
    X_test = test_df[features]
    y_pred = np.expm1(model.predict(X_test))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    results = pd.DataFrame({
        "Date": test_df["Date"].values,
        "Actual_Sales": y_true,
        "Predicted_Sales": y_pred,
        "Absolute_Error": np.abs(y_true - y_pred),
    })

    file_path = os.path.join(OUTPUT_DIR, f"backtest_dashboard_store_{store_id}.xlsx")
    results.to_excel(file_path, index=False)

    wb = load_workbook(file_path)
    ws = wb.active

    chart = LineChart()
    chart.title = "Actual vs Predicted Sales"
    chart.y_axis.title = "Sales"
    chart.x_axis.title = "Days"

    data_ref = Reference(ws, min_col=2, max_col=3, min_row=1, max_row=31)
    categories = Reference(ws, min_col=1, min_row=2, max_row=31)
    chart.add_data(data_ref, titles_from_data=True)
    chart.set_categories(categories)
    ws.add_chart(chart, "F2")

    wb.save(file_path)

    print("\n===== BACKTEST DASHBOARD READY =====")
    print("File:", file_path)
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("MAPE:", round(mape, 2), "%")

    return results
