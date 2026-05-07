import pandas as pd

def compute_zone_risk(forecast_df, original_df):
    """
    Computes zone-level risk using:
    - Demand deviation
    - Peak demand
    - Anomaly density
    """

    df = forecast_df.dropna(subset=["prediction"]).copy()

    hist_avg = (
        original_df
        .groupby(["zone_id", "timestamp"])["consumption"]
        .sum()
        .groupby("zone_id")
        .mean()
        .to_dict()
    )

    anomaly_count = (
        original_df[original_df["anomaly_type"] != "normal"]
        .groupby("zone_id")
        .size()
        .to_dict()
    )
    results = []

    for zone in df["zone_id"].unique():
        zone_data = df[df["zone_id"] == zone]

        forecast_mean = zone_data["prediction"].mean()
        hist_mean = hist_avg.get(zone, 1)

        demand_ratio = forecast_mean / hist_mean
        peak_ratio = zone_data["prediction"].max() / hist_mean

        anomalies = anomaly_count.get(zone, 0)

        results.append({
            "zone_id": zone,
            "forecast_mean": forecast_mean,
            "historical_mean": hist_mean,
            "demand_ratio": demand_ratio,
            "peak_ratio": peak_ratio,
            "anomaly_count": anomalies
        })

    risk_df = pd.DataFrame(results)

    risk_df["demand_score"] = (
        (risk_df["demand_ratio"] - risk_df["demand_ratio"].min()) /
        (risk_df["demand_ratio"].max() - risk_df["demand_ratio"].min() + 1e-6)
    )

    risk_df["peak_score"] = (
        (risk_df["peak_ratio"] - risk_df["peak_ratio"].min()) /
        (risk_df["peak_ratio"].max() - risk_df["peak_ratio"].min() + 1e-6)
    )

    risk_df["anomaly_score"] = (
        risk_df["anomaly_count"] /
        (risk_df["anomaly_count"].max() + 1e-6)
    )

    risk_df["risk_score"] = (
        0.5 * risk_df["demand_score"] +
        0.3 * risk_df["peak_score"] +
        0.2 * risk_df["anomaly_score"]
    )

    def classify(score):
        if score > 0.7:
            return "HIGH"
        elif score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    risk_df["risk_level"] = risk_df["risk_score"].apply(classify)

    return risk_df