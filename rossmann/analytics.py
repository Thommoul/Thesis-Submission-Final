import os

import pandas as pd

from .config import DATA_DIR, OUTPUT_DIR


def sales_by_day_for_year(train, year, plot=True):
    train = train.copy()
    train["Date"] = pd.to_datetime(train["Date"])

    train_year = train[train["Date"].dt.year == year]

    if train_year.empty:
        print(f"No data found for year {year}.")
        return None

    sales_by_day = train_year.groupby("DayOfWeek", as_index=False)["Sales"].sum()

    print(f"\nTotal sales by day of week for {year}:")
    print(sales_by_day.assign(Sales=sales_by_day["Sales"].apply(lambda x: f"{x:,.0f}")))

    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.bar(sales_by_day["DayOfWeek"], sales_by_day["Sales"])
        plt.title(f"Total Sales by Day of Week ({year})")
        plt.xlabel("Day of Week")
        plt.ylabel("Total Sales")
        plt.xticks([1, 2, 3, 4, 5, 6, 7], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        plt.show()

    return sales_by_day


def totalsalesofyear(store_number, year):
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), low_memory=False)
    train["Date"] = pd.to_datetime(train["Date"])

    filtered = train[(train["Store"] == store_number) & (train["Date"].dt.year == year)]
    total = filtered["Sales"].sum()

    print(f"The total sales for Store {store_number} in {year} is: {total:,}")
    return total


def check_closed_days(plot=True):
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), low_memory=False)
    train["Date"] = pd.to_datetime(train["Date"])

    print("\nChecking for days the store was closed:")

    closed_days = train[train["Open"] == 0]

    if closed_days.empty:
        print("The store was open every day in the dataset.")
        return closed_days

    print(f"The store was closed on {closed_days['Date'].nunique()} unique dates.")
    print("\nSample closed dates:")
    print(closed_days[["Store", "Date", "DayOfWeek"]].drop_duplicates().head(10))

    closed_by_day = closed_days.groupby("DayOfWeek").size().reset_index(name="ClosedCount")
    closed_by_day = closed_by_day.sort_values("DayOfWeek")

    print("\nDays of week most often closed:")
    print(closed_by_day)

    if plot:
        import matplotlib.pyplot as plt

        plt.bar(closed_by_day["DayOfWeek"], closed_by_day["ClosedCount"])
        plt.title("Frequency of Store Closures by Day of Week")
        plt.xlabel("Day of Week (1=Mon, 7=Sun)")
        plt.ylabel("Number of Closed Days")
        plt.xticks(range(1, 8))
        plt.show()

    return closed_by_day


def average_sales_when_open():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), low_memory=False)
    df_open = train[train["Open"] == 1]

    avg_sales = df_open.groupby("Store")["Sales"].mean().round().reset_index()
    avg_sales["Sales"] = avg_sales["Sales"].apply(lambda x: "{:,.0f}".format(x).replace(",", "."))

    output_excel = os.path.join(OUTPUT_DIR, "average_sales_when_open.xlsx")
    avg_sales.to_excel(output_excel, index=False)

    print("File saved to:", output_excel)
    return avg_sales


def average_sales_for_store(store_id):
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), low_memory=False)
    df_open = train[(train["Store"] == store_id) & (train["Open"] == 1)]
    avg = df_open["Sales"].mean().round()

    print(f"Average sales for store {store_id} when open is: {avg}")
    return avg


def zero_sales_report(train):
    df = train.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    zero = df[df["Sales"] == 0].copy()
    total = len(zero)
    closed = (zero["Open"] == 0).sum()
    open_store = (zero["Open"] == 1).sum()

    state_holiday = (zero["StateHoliday"] != "0").sum() if "StateHoliday" in zero.columns else None
    school_holiday = (zero["SchoolHoliday"] == 1).sum() if "SchoolHoliday" in zero.columns else None

    print("\n==============================")
    print(" ZERO SALES REPORT")
    print("==============================")
    print(f"Total zero sales : {total}")
    print(f"Closed stores    : {closed}")
    print(f"Open stores      : {open_store}")
    print(f"State holidays   : {state_holiday}")
    print(f"School holidays  : {school_holiday}")

    if total > 0:
        print(f"\nPercentage closed stores : {closed / total * 100:.2f}%")
        print(f"Percentage open stores   : {open_store / total * 100:.2f}%")

    output_excel = os.path.join(OUTPUT_DIR, "zero_sales_report.xlsx")
    zero.to_excel(output_excel, index=False)
    print(f"Saved: {output_excel}")

    return zero
