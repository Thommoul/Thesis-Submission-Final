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
    df = load_store_history(store_id)

    if len(df) < 50:
        raise ValueError("Not enough data for analysis")

    df, X_train, X_test, y_train, y_test, split = (
        _train_test_split_by_time(df, FEATURES)
    )

    model = _make_xgb_regressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    model.fit(X_train, y_train)

    joblib.dump((model, None, FEATURES), model_path(store_id))

    y_pred = np.expm1(model.predict(X_test))
    y_true = np.expm1(np.asarray(y_test).reshape(-1))

    if len(y_true) != len(y_pred):
        raise ValueError("Actual and predicted values have different lengths")

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    if np.any(y_true == 0):
        mape = np.nan
        print("Warning: MAPE cannot be calculated because test set contains zero sales.")
    else:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    df_test = df.iloc[split:].copy()

    results = pd.DataFrame({
        "Date": df_test["Date"].values,
        "Actual_Sales": y_true,
        "Predicted_Sales": y_pred,
        "Error": y_true - y_pred,
        "Absolute_Error": np.abs(y_true - y_pred),
        "Squared_Error": (y_true - y_pred) ** 2,
    })

    metrics = pd.DataFrame([{
    "Store": store_id,
    "N": len(y_true),
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "MAPE (%)": mape}])


    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_path = os.path.join(
        OUTPUT_DIR,
        f"xgboost_store_{store_id}_results.xlsx"
    )

    with pd.ExcelWriter(full_path, engine="openpyxl") as writer:
        results.to_excel(
            writer,
            sheet_name="Predictions",
            index=False
        )

        metrics.to_excel(
            writer,
            sheet_name="Evaluation_Metrics",
            index=False
        )


    print("\n===== XGBOOST MODEL =====")
    print("Store:", store_id)
    print("Test observations (N):", len(y_true))
    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("MAPE (%):", round(mape, 4) if not np.isnan(mape) else "N/A")

    return model, results, metrics


def backtest_last_30_days(store_id):
    path = model_path(store_id)
    if not os.path.exists(path):
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
