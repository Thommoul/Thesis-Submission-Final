"""Shared feature engineering for the regression / XGBoost models."""

FEATURES = [
    "Promo",
    "SchoolHoliday",
    "CompetitionDistance",
    "Month",
    "DayOfWeek",
    "WeekOfYear",
    "IsWeekend",
    "IsSummer",
    "IsChristmas",
]


def add_calendar_features(df):
    df = df.copy()

    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["IsSummer"] = df["Month"].isin([6, 7, 8]).astype(int)
    df["IsChristmas"] = (df["Month"] == 12).astype(int)

    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].median()
    )

    return df


def chronological_split(df, date_col="Date", test_fraction=0.2):

    df = df.sort_values(date_col).reset_index(drop=True)
    split = int(len(df) * (1 - test_fraction))
    return df.iloc[:split], df.iloc[split:]
