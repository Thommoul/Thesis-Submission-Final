import os

import pandas as pd

from .config import DATA_DIR


def load_data():
    store = pd.read_csv(os.path.join(DATA_DIR, "store.csv"), low_memory=False)
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), low_memory=False)
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), low_memory=False)
    return store, test, train


def preprocess_data(store, test, train):
    store = store.drop(["StoreType", "Assortment"], axis=1)
    test = test.drop("StateHoliday", axis=1)
    train = train.drop("StateHoliday", axis=1)


    for df in (train, test):
        date = pd.to_datetime(df["Date"])
        df["Day of the week"] = date.dt.day_name()
        df["Month"] = date.dt.month_name()
        df["Week/52"] = date.dt.isocalendar().week

        df["Date"] = date.dt.strftime("%d/%m/%Y")

    train_cols = list(train.columns)
    train_cols.insert(train_cols.index("Day of the week"), train_cols.pop(train_cols.index("Week/52")))
    train = train[train_cols]

    test_cols = list(test.columns)
    test_cols.insert(test_cols.index("Day of the week"), test_cols.pop(test_cols.index("Week/52")))
    test = test[test_cols]

    return store, test, train


def load_store_history(store_id):
    """Train rows for a single store, merged with store metadata, open days only."""
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["Date"])
    store = pd.read_csv(os.path.join(DATA_DIR, "store.csv"))

    df = train.merge(store, on="Store", how="left")
    df = df[df["Store"] == store_id]
    df = df[df["Open"] == 1].copy()
    return df.sort_values("Date")
