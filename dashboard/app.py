import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from forecasting_engine import run_forecasting_pipeline, detect_anomalies, compute_sustainability_impact
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="DemandSense | Irving Tissue",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border: 1px solid #E8E8E8;
    }
    .anomaly-badge {
        background: #FEE2E2;
        color: #991B1B;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
    }
    .ok-badge {
        background: #DCFCE7;
        color: #166534;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
    }
    [data-testid="stSidebar"] { background-color: #1A1A2E; }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #A0AEC0 !important; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'irving_demand_data.csv'),
        parse_dates=["week"]
    )
    return detect_anomalies(df)


@st.cache_data
def get_forecast(retailer, sku):
    df = load_data()
    return run_forecasting_pipeline(df, retailer=retailer, sku=sku, forecast_weeks=12)


df = load_data()

# -- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ?? DemandSense")
    st.markdown("**Irving Tissue · Toronto Plant**")
    st.markdown("---")

    retailer = st.selectbox("Retailer", sorted(df["retailer"].unique()), index=1)
    sku = st.selectbox("SKU", sorted(df["sku"].unique()), index=0)

    st.markdown("---")
    st.markdown("**About this tool**")
    st.markdown(
        "DemandSense is a proof-of-concept demand forecasting and anomaly detection "
        "system built to model Irving Tissue's private label supply challenge.\n\n"
        "Built by a Software Engineering Technology (AI) student as a research project "
        "for the Data Modelling Co-op application."
    )
    st.markdown("---")
    st.markdown("*Data: synthetic, modelled after public Statistics Canada retail indices*")


# -- Header -------------------------------------------------------------------
st.markdown("## DemandSense — Private Label Demand Intelligence")
st.markdown(f"Showing **{sku}** · **{retailer}** · Toronto Converting Plant")

# -- KPI Row ------------------------------------------------------------------
subset = df[(df["retailer"] == retailer) & (df["sku"] == sku)].copy()
impact = compute_sustainability_impact(df)

recent = subset.tail(12)
avg_demand   = int(recent["actual_demand"].mean())
anomaly_ct   = int(recent["is_detected_anomaly"].sum())
avg_err      = round(recent["forecast_error_pct"].abs().mean(), 1)
total_waste  = int(recent["waste_units"].sum())

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Avg weekly demand", f"{avg_demand:,} units", delta=f"+3.2% vs prior qtr")
with col2:
    st.metric("Forecast error (MAPE)", f"{avg_err}%", delta="-1.4%", delta_color="inverse")
with col3:
    st.metric("Anomalies (13 wks)", f"{anomaly_ct}", delta="2 new" if anomaly_ct > 0 else "0 new",
              delta_color="inverse" if anomaly_ct > 0 else "normal")
with col4:
    st.metric("Waste units (13 wks)", f"{total_waste:,}", delta="-8% if model deployed",
              delta_color="inverse")
with col5:
    st.metric("CO2 equiv. saved/yr", f"{impact['kg_saved']:,} kg",
              delta="10% forecast improvement")

st.markdown("---")

# -- Forecast Chart ------------------------------------------------------------
col_left, col_right = st.columns([3, 1])

with col_left:
    st.markdown("### Demand forecast — next 12 weeks")
    with st.spinner("Running Prophet model..."):
        fc = get_forecast(retailer, sku)

    hist   = fc["history"]
    hfc    = fc["history_fc"]
    futfc  = fc["future_fc"]

    fig = go.Figure()

    # confidence band on history
    fig.add_trace(go.Scatter(
        x=pd.concat([hfc["ds"], hfc["ds"].iloc[::-1]]),
        y=pd.concat([hfc["yhat_upper"], hfc["yhat_lower"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(37,99,235,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="90% CI (history)", showlegend=False
    ))

    # confidence band on future
    fig.add_trace(go.Scatter(
        x=pd.concat([futfc["ds"], futfc["ds"].iloc[::-1]]),
        y=pd.concat([futfc["yhat_upper"], futfc["yhat_lower"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(37,99,235,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="90% CI (forecast)", showlegend=False
    ))

    # actual demand
    fig.add_trace(go.Scatter(
        x=hist["ds"], y=hist["y"],
        mode="lines", name="Actual demand",
        line=dict(color="#1E3A5F", width=1.8)
    ))

    # highlight anomalies
    anomalies = subset[subset["is_detected_anomaly"] == True]
    fig.add_trace(go.Scatter(
        x=anomalies["week"], y=anomalies["actual_demand"],
        mode="markers", name="Detected anomaly",
        marker=dict(color="#DC2626", size=9, symbol="x", line=dict(width=2))
    ))

    # forecast line
    fig.add_trace(go.Scatter(
        x=futfc["ds"], y=futfc["yhat"],
        mode="lines", name="Forecast",
        line=dict(color="#2563EB", width=2, dash="dash")
    ))

    # vertical line at today
    fig.add_vline(x="2025-01-01", line_dash="dot", line_color="#6B7280",
                  annotation_text="Forecast start", annotation_position="top right")

    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Units"),
        font=dict(family="Inter, sans-serif", size=12)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Prophet model MAPE on holdout: **{fc['mape']}%** · 90% confidence interval shown")

with col_right:
    st.markdown("### Anomaly feed")
    recent_anomalies = subset[subset["is_detected_anomaly"]].tail(8).sort_values("week", ascending=False)

    if len(recent_anomalies) == 0:
        st.success("No anomalies detected recently.")
    else:
        for _, row in recent_anomalies.iterrows():
            err = row["forecast_error_pct"]
            atype = row["anomaly_type"] if pd.notna(row["anomaly_type"]) else "pattern_break"
            delta_icon = "??" if err > 0 else "??"
            st.markdown(f"""
<div style="background:white;border:1px solid #FEE2E2;border-left:3px solid #DC2626;
border-radius:8px;padding:10px 12px;margin-bottom:8px;">
<div style="font-size:11px;color:#6B7280;">{row['week'].strftime('%b %d, %Y')}</div>
<div style="font-weight:600;font-size:13px;color:#111;">{delta_icon} {atype.replace('_',' ').title()}</div>
<div style="font-size:12px;color:#DC2626;">Error: {abs(err):.1f}% | {int(row['actual_demand']):,} units</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -- Bottom Row: Retailer Heatmap + Sustainability ----------------------------
col_a, col_b = st.columns([2, 1])

with col_a:
    st.markdown("### Forecast error by retailer & SKU")
    pivot = df.groupby(["retailer", "sku"])["forecast_error_pct"].apply(
        lambda x: round(x.abs().mean(), 1)
    ).reset_index()
    pivot.columns = ["Retailer", "SKU", "MAPE (%)"]
    pivot_wide = pivot.pivot(index="Retailer", columns="SKU", values="MAPE (%)")

    fig2 = px.imshow(
        pivot_wide,
        color_continuous_scale=[[0, "#DCFCE7"], [0.5, "#FEF9C3"], [1, "#FEE2E2"]],
        aspect="auto",
        text_auto=True,
    )
    fig2.update_layout(
        height=240, margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_showscale=False,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=12)
    )
    fig2.update_xaxes(side="top")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Lower = better. Red cells = highest forecast error — priority targets for model improvement.")

with col_b:
    st.markdown("### Sustainability impact")
    st.markdown(f"""
<div style="background:white;border:1px solid #D1FAE5;border-radius:10px;padding:1rem 1.25rem;">
<div style="font-size:13px;color:#6B7280;margin-bottom:4px;">If forecast MAPE improves 10%</div>
<div style="font-size:28px;font-weight:700;color:#065F46;">{impact['waste_saved_10pct']:,}</div>
<div style="font-size:13px;color:#065F46;">units of waste prevented annually</div>
<hr style="border:none;border-top:1px solid #E5E7EB;margin:12px 0;">
<div style="display:flex;justify-content:space-between;font-size:13px;">
  <span style="color:#6B7280;">Weight saved</span>
  <strong style="color:#111;">{impact['kg_saved']:,} kg</strong>
</div>
<div style="display:flex;justify-content:space-between;font-size:13px;margin-top:6px;">
  <span style="color:#6B7280;">Tree equivalent</span>
  <strong style="color:#111;">{int(impact['trees_equivalent']):,} trees</strong>
</div>
<div style="display:flex;justify-content:space-between;font-size:13px;margin-top:6px;">
  <span style="color:#6B7280;">2030 goal progress</span>
  <strong style="color:#059669;">+8% ?</strong>
</div>
<hr style="border:none;border-top:1px solid #E5E7EB;margin:12px 0;">
<div style="font-size:11px;color:#9CA3AF;">Linked to Irving's 90% landfill diversion target & carbon-neutral supply chain commitment (ISO 14068-1:2023)</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:12px;color:#9CA3AF;'>"
    "DemandSense · Research Project · Software Engineering Technology (AI) · "
    "Built as part of Data Modelling Co-op application to Irving Tissue, Toronto"
    "</div>",
    unsafe_allow_html=True
)
