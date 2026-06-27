import pandas as pd
import numpy as np
# Meta's open-source forecasting tool,
from prophet import Prophet
# Imports an unsupervised machine learning algorithm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_percentage_error
# Suppresses warning messages
import warnings
warnings.filterwarnings("ignore")

def run_forecasting_pipeline(df, retailer, sku, forecast_weeks=12):
    subset = df[(df["retailer"] == retailer) & (df["sku"] == sku)].copy()
    subset = subset.sort_values("week").reset_index(drop=True)

    prophet_df = subset[["week", "actual_demand"]].rename(
        columns={"week": "ds", "actual_demand": "y"}
    )

    # add promo regressor
    prophet_df["is_promo"] = subset["is_promo"].astype(int).values

    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        weekly_seasonality=False,
        yearly_seasonality=True,
        interval_width=0.90,
    )
    model.add_regressor("is_promo")
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=forecast_weeks, freq="W")
    future["is_promo"] = 0  # assume no promo in forecast horizon

    forecast = model.predict(future)

    # split into history + future
    history_forecast = forecast[forecast["ds"] <= prophet_df["ds"].max()].copy()
    future_forecast  = forecast[forecast["ds"] >  prophet_df["ds"].max()].copy()

    # MAPE on history
    merged = prophet_df.merge(history_forecast[["ds","yhat"]], on="ds")
    mape = mean_absolute_percentage_error(merged["y"], merged["yhat"]) * 100

    return {
        "history":        prophet_df,
        "history_fc":     history_forecast,
        "future_fc":      future_forecast,
        "mape":           round(mape, 2),
        "model":          model,
    }


def detect_anomalies(df):
    results = []
    for (retailer, sku), group in df.groupby(["retailer", "sku"]):
        group = group.sort_values("week").copy()
        X = group[["actual_demand", "planned_production", "forecast_error_pct"]].values

        iso = IsolationForest(contamination=0.05, random_state=42)
        group["anomaly_score"] = iso.fit_predict(X)
        group["is_detected_anomaly"] = group["anomaly_score"] == -1

        results.append(group)

    return pd.concat(results, ignore_index=True)


def compute_sustainability_impact(df):
    total_waste    = df["waste_units"].sum()
    baseline_mape  = df["forecast_error_pct"].abs().mean()

    # if MAPE improves by 10%, waste reduces proportionally
    improved_waste = total_waste * (1 - 0.10)
    waste_saved    = total_waste - improved_waste

    # Irving's goal: divert 90% of waste from landfill
    # avg tissue roll ~ 0.12 kg
    kg_saved       = waste_saved * 0.12
    trees_equiv    = kg_saved / 60  # ~60kg of tissue per tree equivalent

    return {
        "total_waste_units":    int(total_waste),
        "waste_saved_10pct":    int(waste_saved),
        "kg_saved":             round(kg_saved, 1),
        "trees_equivalent":     round(trees_equiv, 0),
        "baseline_error_pct":   round(baseline_mape, 2),
    }


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/DemandSense-Irving/data/irving_demand_data.csv", parse_dates=["week"])

    print("Running anomaly detection...")
    df_with_anomalies = detect_anomalies(df)
    df_with_anomalies.to_csv("/home/claude/DemandSense-Irving/data/irving_demand_with_anomalies.csv", index=False)
    print(f"Detected anomalies: {df_with_anomalies['is_detected_anomaly'].sum()}")

    print("\nRunning Prophet forecast for Loblaws - Bath Tissue 12pk...")
    result = run_forecasting_pipeline(df, retailer="Loblaws", sku="Bath Tissue 12pk")
    print(f"MAPE: {result['mape']}%")
    print(f"Forecast horizon: {result['future_fc']['ds'].min().date()} → {result['future_fc']['ds'].max().date()}")

    print("\nSustainability impact:")
    impact = compute_sustainability_impact(df)
    for k, v in impact.items():
        print(f"  {k}: {v}")