import pandas as pd

def create_features(df):
    df = df.copy()

    # Ensure sorted
    df = df.sort_values(["meter_id", "timestamp"])

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Lag features (grouped by meter)
    df["lag_1"] = df.groupby("meter_id")["consumption"].shift(1)
    df["lag_4"] = df.groupby("meter_id")["consumption"].shift(4)
    df["lag_96"] = df.groupby("meter_id")["consumption"].shift(96)

    # Rolling features
    df["rolling_mean_4"] = (
        df.groupby("meter_id")["consumption"]
        .rolling(window=4)
        .mean()
        .reset_index(0, drop=True)
    )

    df["rolling_mean_96"] = (
        df.groupby("meter_id")["consumption"]
        .rolling(window=96)
        .mean()
        .reset_index(0, drop=True)
    )

    return df