import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

RETAILERS = {
    "Loblaws":       {"base": 4200, "sensitivity": 1.3, "promo_freq": 0.18},
    "Walmart CA":    {"base": 5800, "sensitivity": 1.5, "promo_freq": 0.22},
    "Costco CA":     {"base": 7100, "sensitivity": 0.9, "promo_freq": 0.10},
    "Metro":         {"base": 2100, "sensitivity": 1.2, "promo_freq": 0.15},
    "Sobeys":        {"base": 2600, "sensitivity": 1.1, "promo_freq": 0.16},
}

SKUS = {
    "Bath Tissue 12pk":   {"multiplier": 1.0, "waste_rate": 0.03},
    "Bath Tissue 24pk":   {"multiplier": 1.6, "waste_rate": 0.025},
    "Paper Towel 6pk":    {"multiplier": 0.8, "waste_rate": 0.035},
    "Facial Tissue 3pk":  {"multiplier": 0.5, "waste_rate": 0.04},
}

def holiday_bump(date):
    m, d = date.month, date.day
    if m == 12 and d >= 15: return 1.35
    if m == 11 and 20 <= d <= 30: return 1.28
    if m == 3 and 10 <= d <= 20: return 1.45  # flu/covid wave proxy
    if m == 9 and 1 <= d <= 14:  return 1.18  # back to school
    if m == 4 and 1 <= d <= 14:  return 1.12  # spring clean
    return 1.0

def seasonal_factor(date):
    # tissue demand peaks in winter (flu), dips in summer
    day_of_year = date.timetuple().tm_yday
    return 1 + 0.18 * np.cos(2 * np.pi * (day_of_year - 15) / 365)

def generate_dataset(start="2022-01-01", end="2024-12-31"):
    dates = pd.date_range(start, end, freq="W-MON")
    rows = []

    for date in dates:
        hbump = holiday_bump(date)
        sfactor = seasonal_factor(date)

        for retailer, rconf in RETAILERS.items():
            for sku, sconf in SKUS.items():
                base = rconf["base"] * sconf["multiplier"]

                # promo spike
                is_promo = np.random.random() < rconf["promo_freq"]
                promo_mult = np.random.uniform(1.25, 1.65) if is_promo else 1.0

                # noise
                noise = np.random.normal(1.0, 0.07)

                demand = base * sfactor * hbump * promo_mult * noise
                demand = max(0, round(demand))

                # anomaly injection (supply shocks, demand surges — ~2% of rows)
                is_anomaly = np.random.random() < 0.02
                anomaly_type = None
                if is_anomaly:
                    shock = np.random.choice(["surge", "drop"])
                    if shock == "surge":
                        demand = round(demand * np.random.uniform(1.8, 2.6))
                        anomaly_type = "demand_surge"
                    else:
                        demand = round(demand * np.random.uniform(0.1, 0.35))
                        anomaly_type = "supply_shock"

                # production planned vs actual
                forecast_error = np.random.normal(0, 0.09)
                planned_production = round(demand * (1 + forecast_error))
                waste_units = round(abs(planned_production - demand) * sconf["waste_rate"] * np.random.uniform(0.8, 1.2))

                rows.append({
                    "week":               date,
                    "retailer":           retailer,
                    "sku":                sku,
                    "actual_demand":      demand,
                    "planned_production": planned_production,
                    "forecast_error_pct": round(forecast_error * 100, 2),
                    "is_promo":           is_promo,
                    "is_anomaly":         is_anomaly,
                    "anomaly_type":       anomaly_type,
                    "waste_units":        waste_units,
                    "waste_rate_pct":     sconf["waste_rate"] * 100,
                })

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "irving_demand_data.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Generated {len(df):,} rows across {df['retailer'].nunique()} retailers and {df['sku'].nunique()} SKUs")
    print(f"Date range: {df['week'].min().date()} → {df['week'].max().date()}")
    print(f"Anomalies injected: {df['is_anomaly'].sum()}")
    return df

if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())