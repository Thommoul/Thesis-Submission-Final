import os
import matplotlib.pyplot as plt
import pandas as pd

from .config import DATA_DIR, OUTPUT_DIR
from pathlib import Path

DAY_LABELS = {1: "Δευτέρα", 2: "Τρίτη", 3: "Τετάρτη", 4: "Πέμπτη", 5: "Παρασκευή", 6: "Σάββατο", 7: "Κυριακή"}

#Σχόλιο 4.3? ή το από κάτω;
# python -c "from rossmann.analytics import sales_by_day_for_year; sales_by_day_for_year(2014)"
def sales_by_day_for_year(year, plot=True):

    train = pd.read_csv(
        os.path.join(DATA_DIR, "train.csv"),
        low_memory=False
    )

    train["Date"] = pd.to_datetime(train["Date"])

    train_year = train[train["Date"].dt.year == year]

    if train_year.empty:
        print(f"No data found for year {year}.")
        return None

    open_days = train_year[train_year["Open"] == 1]

    if open_days.empty:
        print(f"No open store-days found for year {year}.")
        return None

    days = list(DAY_LABELS.keys())

    summary = (
        open_days.groupby("DayOfWeek")["Sales"]
        .agg(
            N="count",
            Mean_Sales="mean",
            Median_Sales="median"
        )
        .reindex(days)
    )

    summary["N"] = summary["N"].fillna(0).astype(int)

    summary["Total_Sales"] = (
        train_year
        .groupby("DayOfWeek")["Sales"]
        .sum()
        .reindex(days)
        .fillna(0)
    )

    summary = summary.reset_index()

    summary.insert(
        1,
        "Day",
        summary["DayOfWeek"].map(DAY_LABELS)
    )

    print(
        f"\nΣυνολικές πωλήσεις ανά Ημέρα Εβδομάδας {year} "
        f"(Open = 1 only):"
    )

    print(
        "Mean/Median = Sales per open store-day; "
        "N = number of open store-day observations.\n"
        "Total_Sales is supplementary and depends on N."
    )

    print(
        summary.assign(
            N=summary["N"].map("{:,}".format),
            Mean_Sales=summary["Mean_Sales"].map(
                lambda x: "-" if pd.isna(x) else f"{x:,.0f}"
            ),
            Median_Sales=summary["Median_Sales"].map(
                lambda x: "-" if pd.isna(x) else f"{x:,.0f}"
            ),
            Total_Sales=summary["Total_Sales"].map(
                "{:,.0f}".format
            )
        ).to_string(index=False)
    )

    if plot:

        plt.figure(figsize=(8, 5))

        plt.bar(
            summary["DayOfWeek"],
            summary["Mean_Sales"].fillna(0),
            label="Mean"
        )

        plt.scatter(
            summary["DayOfWeek"],
            summary["Median_Sales"],
            color="black",
            zorder=3,
            label="Median"
        )

        for _, row in summary.iterrows():
            plt.annotate(
                f"N={row['N']:,}",
                (
                    row["DayOfWeek"],
                    row["Mean_Sales"]
                    if pd.notna(row["Mean_Sales"])
                    else 0
                ),
                ha="center",
                va="bottom",
                fontsize=8
            )

        plt.title(
            f"Μέση πώληση όπου το κατάστημα ήταν ανοιχτό ({year})"
        )
        plt.xlabel("Ημέρα της εβδομάδας")
        plt.ylabel("Πωλήσεις ανά ανοιχτό κατάστημα-ημέρα")
        plt.xticks(
            days,
            [DAY_LABELS[d] for d in days]
        )
        plt.legend()
        plt.tight_layout()
        plt.show()


        plt.figure(figsize=(8, 5))

        plt.bar(
            summary["DayOfWeek"],
            summary["Total_Sales"],
            color="grey"
        )

        plt.title(
            f"Συνολικές πωλήσεις ανά ημέρα της εβδομάδας ({year})"
        )
        plt.xlabel("Ημέρα της εβδομάδας")
        plt.ylabel("Συνολικές πωλήσεις")
        plt.xticks(
            days,
            [DAY_LABELS[d] for d in days]
        )

        plt.tight_layout()
        plt.show()

    return summary

# python -c "from rossmann.analytics import sales_by_day_for_year; sales_by_day_for_year(2014)"

def sales_by_day_for_full_years(csv_path=None, details=False):
    """Υπολογίζει τις πωλήσεις ανά ημέρα και συγκρίνει τα πλήρη έτη."""
    if csv_path is None:
        csv_path = Path(DATA_DIR) / "train.csv"

    train = pd.read_csv(
        csv_path,
        usecols=["Date", "DayOfWeek", "Open", "Sales"],
        parse_dates=["Date"],
    )
    train["Year"] = train["Date"].dt.year

    coverage = train.groupby("Year").agg(
        First_date=("Date", "min"),
        Last_date=("Date", "max"),
        Days=("Date", "nunique"),
        Total_sales=("Sales", "sum"),
    )

    first_day = pd.to_datetime(coverage.index.astype(str) + "-01-01")
    coverage["Expected_days"] = 365 + first_day.is_leap_year.astype(int)
    coverage["Full_year"] = coverage["Days"].eq(coverage["Expected_days"])

    open_days = train.loc[train["Open"].eq(1)]

    by_day = open_days.groupby(["Year", "DayOfWeek"])["Sales"].agg(
        N="count",
        Mean="mean",
        Median="median",
        Total="sum",
    )

    day_names = {
        1: "Δευτέρα",
        2: "Τρίτη",
        3: "Τετάρτη",
        4: "Πέμπτη",
        5: "Παρασκευή",
        6: "Σάββατο",
        7: "Κυριακή",
    }
    by_day = by_day.rename(index=day_names, level="DayOfWeek")

    full_years = coverage.index[coverage["Full_year"]]
    comparison = by_day.loc[full_years, ["N", "Mean", "Median"]]
    comparison = comparison.unstack("Year").swaplevel(axis=1)

    columns = pd.MultiIndex.from_product(
        [full_years, ["N", "Mean", "Median"]]
    )
    comparison = comparison.reindex(
        index=list(day_names.values()),
        columns=columns,
    )

    if details:
        return coverage, by_day, comparison

    return comparison

#python -c "from rossmann.analytics import print_sales_by_day_report; print_sales_by_day_report()"
def print_sales_by_day_report():
    """Εμφανίζει μαζί την κάλυψη, τα ελλιπή έτη και την τελική σύγκριση."""
    coverage, by_day, comparison = sales_by_day_for_full_years(details=True)

    print("\n1. Χρονική κάλυψη και συνολικές πωλήσεις ανά έτος")
    print(coverage.to_string())

    print("\n2. Πωλήσεις ανά ημέρα εβδομάδας για τα έτη με ελλιπή κάλυψη")
    for year in coverage.index[~coverage["Full_year"]]:
        first_date = coverage.loc[year, "First_date"].strftime("%d/%m/%Y")
        last_date = coverage.loc[year, "Last_date"].strftime("%d/%m/%Y")
        print(f"\n{year} ({first_date} έως {last_date})")
        print(by_day.loc[year].round(2).to_string())

    print("\n3. Σύγκριση πωλήσεων ανά ημέρα εβδομάδας για τα πλήρη έτη")
    print(comparison.round(2).to_string())

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
