import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

def train_forecast_model(zone_df):
    df = zone_df.dropna().copy()

    features = [
        "hour", "day_of_week", "is_weekend",
        "lag_1", "lag_4", "lag_96"
    ]

    target = "consumption"
    df = df.sort_values("timestamp")
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, preds))

    df["prediction"] = None
    df.loc[test_df.index, "prediction"] = preds

    try:
        import shap
        import matplotlib.pyplot as plt

        sample_size = min(200, len(X_test))
        X_sample = X_test.sample(sample_size, random_state=42).copy()
        X_sample.rename(columns={
            "lag_96": "Same Time Yesterday Usage",
            "lag_1": "Last Interval Usage",
            "lag_4": "Usage 1 Hour Ago",
            "hour": "Time of Day",
            "day_of_week": "Weekday",
            "is_weekend": "Weekend"
        }, inplace=True)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.xlim(0, 10)
        plt.xlabel("Impact on Prediction (Higher = More Important)")
        plt.title("Key Drivers of Electricity Consumption Forecast")
        plt.tight_layout()
        plt.savefig("data/shap_summary.png", bbox_inches = "tight")
        plt.close()

        print("SHAP summary saved!")

    except Exception as e:
        print("SHAP failed:", e)

    return model, df