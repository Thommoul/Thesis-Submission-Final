import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

from .config import DATA_DIR


def plot_sales_density():
    store = pd.read_csv(os.path.join(DATA_DIR, "store.csv"), low_memory=False)
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), low_memory=False)

    train_merged = train.merge(store[["Store", "Promo2"]], on="Store", how="left")

    promo_sales = train_merged[train_merged["Promo2"] == 1]["Sales"]
    non_promo_sales = train_merged[train_merged["Promo2"] == 0]["Sales"]

    combined_data = pd.DataFrame(
        {
            "Sales": pd.concat([promo_sales, non_promo_sales]),
            "Period": ["Promotion"] * len(promo_sales) + ["Non-Promotion"] * len(non_promo_sales),
        }
    )

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=combined_data, x="Sales", hue="Period", fill=True, alpha=0.5)
    plt.title("Comparison of Sales Density During Promotion and Non-Promotion Periods")
    plt.xlabel("Sales")
    plt.ylabel("Density")
    plt.xlim(0, 5000)
    plt.show()

# Για να τρέξει γράψτε: "python -c "from rossmann.plots import plot_sales_growth; plot_sales_growth()"
def plot_sales_growth():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), low_memory=False)
    train["Date"] = pd.to_datetime(train["Date"], errors="coerce")

    invalid_dates = (train["Date"] < pd.to_datetime("2013-01-01")) | (train["Date"] > pd.to_datetime("2015-07-26"))
    train.loc[invalid_dates, "Date"] = pd.NaT

    if train["Date"].isna().any():
        mean_date = train["Date"].dropna().mean()
        train["Date"] = train["Date"].fillna(mean_date)

    train = train.sort_values("Date")

    daily_sales = train.groupby("Date")["Sales"].sum()
    sales_growth_rate = daily_sales.pct_change().fillna(0)
    sales_growth_rate = sales_growth_rate.replace([np.inf, -np.inf], np.nan).dropna()

    dates = sales_growth_rate.index

    plt.figure(figsize=(12, 6))
    plt.plot(dates, sales_growth_rate, color="blue", linewidth=2, label="Sales Growth Rate")
    plt.axhline(0, color="red", linestyle="--", linewidth=1, label="Zero Growth")

    if len(dates) > 0:
        smoothed = lowess(sales_growth_rate.values, np.arange(len(sales_growth_rate)), frac=0.05)
        plt.plot(dates, smoothed[:, 1], color="orange", linewidth=2, label="Trend Line")

    plt.xlabel("Date")
    plt.ylabel("Sales Growth Rate")
    plt.title("Sales Growth Rate Over Time", fontsize=14)
    plt.suptitle("Trend line represents smoothed growth trend", fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.show()

#python -c "from rossmann.plots import plot_store_1_sales; plot_store_1_sales()"
def plot_store_1_sales():
    import pandas as pd
    import matplotlib.pyplot as plt

    train = pd.read_csv(
        r"C:\Users\30695\Desktop\Programming\python\Files\train.csv"
    )

    train["Date"] = pd.to_datetime(
        train["Date"],
        dayfirst=True
    )

    store_1 = train[train["Store"] == 1].copy()
    store_1 = store_1.sort_values("Date").head(20)

    plt.figure(figsize=(12, 5))

    plt.plot(
        store_1["Date"],
        store_1["Sales"]
    )

    plt.title("Εξέλιξη των ημερήσιων πωλήσεων του καταστήματος 1")
    plt.xlabel("Ημερομηνία")
    plt.ylabel("Ημερήσιες Πωλήσεις")

    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
