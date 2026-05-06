import pandas as pd
import numpy as np

def assign_zone_coordinates(risk_df):
    """
    Assign fake lat/lon around Bangalore
    """

    base_lat = 12.9716
    base_lon = 77.5946

    np.random.seed(42)

    coords = []

    for zone in risk_df["zone_id"]:
        lat = base_lat + np.random.uniform(-0.05, 0.05)
        lon = base_lon + np.random.uniform(-0.05, 0.05)

        coords.append((lat, lon))

    risk_df["lat"] = [c[0] for c in coords]
    risk_df["lon"] = [c[1] for c in coords]

    return risk_df