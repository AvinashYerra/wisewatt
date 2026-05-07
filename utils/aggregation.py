import pandas as pd

def aggregate_zone_level(df):
    zone_df = (
        df.groupby(["zone_id", "timestamp"])
        .agg({
            "consumption": "sum",
            "hour": "first",
            "day_of_week": "first",
            "is_weekend": "first"
        })
        .reset_index()
    )

    zone_df = zone_df.sort_values(["zone_id", "timestamp"])

    zone_df["lag_1"] = zone_df.groupby("zone_id")["consumption"].shift(1)
    zone_df["lag_4"] = zone_df.groupby("zone_id")["consumption"].shift(4)
    zone_df["lag_96"] = zone_df.groupby("zone_id")["consumption"].shift(96)

    return zone_df