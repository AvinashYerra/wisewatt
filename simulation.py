import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.anomaly_injection import inject_anomalies
from utils.features import create_features
from utils.aggregation import aggregate_zone_level
from utils.forecasting import train_forecast_model
from utils.risk import compute_zone_risk


def generate_smart_meter_data(
    num_zones=10,
    meters_per_zone=50,
    days=7,
    freq="15min"
):
    timestamps = pd.date_range(
        start="2025-01-01",
        periods=days * 24 * 4,
        freq=freq
    )

    data = []

    for z in range(num_zones):
        zone_id = f"Z{z+1}"

        for m in range(meters_per_zone):
            meter_id = f"{zone_id}_M{m+1}"

            base_load = np.random.uniform(0.5, 2.0)

            for ts in timestamps:
                hour = ts.hour

                # Daily pattern
                if 6 <= hour <= 10:
                    load = base_load * np.random.uniform(1.5, 2.0)
                elif 18 <= hour <= 22:
                    load = base_load * np.random.uniform(1.8, 2.5)
                else:
                    load = base_load * np.random.uniform(0.5, 1.2)

                # Weekend effect
                if ts.dayofweek >= 5:
                    load *= np.random.uniform(0.8, 1.0)

                # Noise
                load += np.random.normal(0, 0.05)

                data.append([meter_id, zone_id, ts, max(load, 0)])

    df = pd.DataFrame(
        data,
        columns=["meter_id", "zone_id", "timestamp", "consumption"]
    )

    return df



df = generate_smart_meter_data()
df = inject_anomalies(df)
df = create_features(df)
zone_df = aggregate_zone_level(df)
model, forecast_df = train_forecast_model(zone_df)
risk_df = compute_zone_risk(forecast_df, df)

df.to_parquet("data/smart_meter.parquet", index=False)
zone_df.to_parquet("data/zone_level.parquet", index=False)
forecast_df.to_parquet("data/forecast_output.parquet", index=False)
risk_df.to_parquet("data/risk_output.parquet", index=False)

print(risk_df)