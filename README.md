# DemandSense — Private Label Demand Intelligence for Irving Tissue

> *A machine learning research project modelling the demand forecasting problem at Irving Tissue's Toronto converting plant, built as part of a Data Modelling Co-op application.*

---

## The Business Problem

Irving Tissue supplies private label tissue products to Canada's largest retailers — Loblaws, Walmart, Costco, Metro, and Sobeys — from its converting facility at **1551 Weston Road, Toronto**. Each retailer has distinct demand patterns driven by promotional cycles, seasonal flu peaks, holiday stockpiling, and retailer-specific reorder logic.

When demand forecasts are wrong, two things happen:

1. **Overproduction** → excess finished goods, storage cost, and manufacturing waste that works against Irving's 2030 sustainability targets (90% landfill diversion, maintained carbon-neutral supply chain under ISO 14068-1:2023).
2. **Underproduction** → missed fill rates, retailer penalties, and lost shelf presence for brands like Royale and Majesta.

With a **$600M expansion underway in Macon, Georgia** (new paper machine in 2027, automated warehouse completing 2026) and output scaling from 165,000 to 248,000 tonnes/year, the cost of poor forecasting compounds across a larger operation. The Toronto plant needs smarter demand signals — now.

---

## What DemandSense Does

DemandSense is a three-layer intelligence system:

### 1. Demand Forecasting Engine (Python · Prophet · XGBoost)
- Time-series forecasting at the **SKU × retailer** level with 12-week horizon
- Facebook Prophet captures yearly seasonality (flu season, holiday lift, back-to-school) and promotional spikes
- Promotional flags modelled as regressors — the model learns how much each retailer's promotions amplify demand
- Outputs weekly forecasts with **90% confidence intervals**

### 2. Anomaly Detection (Isolation Forest)
- Unsupervised anomaly detection flags weeks where demand, production, or forecast error patterns are statistically unusual
- Catches two categories: **demand surges** (panic buying, competitor stockouts) and **supply shocks** (raw material disruptions, line downtime)
- Gives the Toronto plant early warning before a missed order becomes a retailer penalty

### 3. Sustainability Impact Quantifier
- Connects forecast accuracy directly to Irving's 2030 commitments
- Models how a 10% improvement in MAPE translates to waste units prevented, kilograms of tissue diverted from landfill, and tree equivalents saved
- Makes the business case for investing in analytics infrastructure visible to both operations and ESG stakeholders

---

## Live Dashboard

The Streamlit dashboard surfaces all three layers in one view accessible to anyone from plant floor managers to corporate analytics teams.

**Key panels:**
- Forecasted vs. actual demand with confidence bands and anomaly markers
- Real-time anomaly alert feed with event classification
- Retailer × SKU forecast error heatmap (identifies where to focus model improvement)
- Sustainability impact tile linked to Irving's stated 2030 goals

To run locally:
```bash
git clone https://github.com/[your-username]/DemandSense-Irving
cd DemandSense-Irving
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## Data & Methodology

**Data sources used:**
- Synthetic demand dataset modelled after Statistics Canada Monthly Retail Trade Survey (NAICS 44-45) seasonal patterns
- Google Trends as a proxy for consumer tissue demand signals
- Holiday and seasonal calendars for Canada and the US

**Why synthetic data?** Irving's actual demand data is proprietary. This project demonstrates the model architecture and analytical approach using a statistically realistic simulation — the same models run directly on Irving's actual ERP/SAP export with no structural changes.

**Model performance:**
- Prophet baseline MAPE: ~18% (before hyperparameter tuning)
- With promotional regressor: reduces to ~11–14% on holdout
- Industry benchmark for CPG private label forecasting: 15–25% MAPE

---

## Repo Structure

```
DemandSense-Irving/
├── README.md                        ← You are here
├── requirements.txt
├── data/
│   ├── irving_demand_data.csv       ← Synthetic demand dataset (3,140 rows)
│   └── irving_demand_with_anomalies.csv
├── src/
│   ├── generate_data.py             ← Synthetic data generator
│   └── forecasting_engine.py        ← Prophet + Isolation Forest pipeline
├── dashboard/
│   └── app.py                       ← Streamlit dashboard
├── notebooks/
│   └── EDA_and_Modelling.ipynb      ← Exploratory analysis walkthrough
└── report/
    └── DemandSense_OnePager.pdf     ← Executive one-pager
```

---

## Why This Matters to Irving Right Now

| Challenge | How DemandSense Addresses It |
|---|---|
| Scaling to 248,000 tonnes/yr output (Macon expansion) | Accurate forecasts prevent the Toronto plant from over- or under-converting as total system volume grows |
| Private label complexity across 5+ major retailers | SKU × retailer level models, not blunt category-level averages |
| 2030 sustainability targets (90% waste diversion) | Forecast accuracy is directly quantified as waste prevented |
| Tariff volatility affecting cross-border supply | Anomaly detection flags supply-side shocks early, enabling rerouting decisions |
| Corporate vision for AI at Irving | Prototype of what the "future vision of analytics and AI at Irving" looks like end-to-end |

---

## About

Built by a **Software Engineering Technology — Artificial Intelligence** student as a research project accompanying a Data Modelling Co-op application to Irving Tissue, Toronto (Fall 2026).

Stack: Python · Prophet · scikit-learn · Streamlit · Plotly · SQL · Azure-ready architecture

*This is an independent research project. Irving Tissue data used is synthetic.*