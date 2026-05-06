import pandas as pd
import numpy as np

def inject_anomalies(df, anomaly_fraction=0.05):
    df = df.copy()
    df["anomaly_type"] = "normal"

    meter_ids = df["meter_id"].unique()
    num_anomalous_meters = int(len(meter_ids) * anomaly_fraction)

    anomalous_meters = np.random.choice(
        meter_ids,
        num_anomalous_meters,
        replace=False
    )

    for meter in anomalous_meters:
        meter_idx = df["meter_id"] == meter
        meter_data = df[meter_idx]

        anomaly_choice = np.random.choice(
            ["drop", "flat", "spike"]
        )

        start_idx = np.random.randint(0, len(meter_data) - 40)
        end_idx = start_idx + np.random.randint(20, 50)

        indices = meter_data.index[start_idx:end_idx]

        if anomaly_choice == "drop":
            df.loc[indices, "consumption"] *= np.random.uniform(0.1, 0.5)
            df.loc[indices, "anomaly_type"] = "sudden_drop"

        elif anomaly_choice == "flat":
            flat_value = meter_data.iloc[start_idx]["consumption"]
            df.loc[indices, "consumption"] = flat_value
            df.loc[indices, "anomaly_type"] = "flatline"

        elif anomaly_choice == "spike":
            df.loc[indices, "consumption"] *= np.random.uniform(2, 4)
            df.loc[indices, "anomaly_type"] = "spike"

    return df