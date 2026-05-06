import streamlit as st
import pandas as pd
import pydeck as pdk

from utils.geo import assign_zone_coordinates

st.set_page_config(layout="wide")

st.markdown("""
<style>
h1, h2, h3 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center;'>
⚡ WiseWatt — Smart Grid Intelligence
</h1>
""", unsafe_allow_html=True)

forecast_df = pd.read_parquet("data/forecast_output.parquet")
risk_df = pd.read_parquet("data/risk_output.parquet")

col1, col2, col3 = st.columns(3)

high = (risk_df["risk_level"] == "HIGH").sum()
medium = (risk_df["risk_level"] == "MEDIUM").sum()
low = (risk_df["risk_level"] == "LOW").sum()

st.markdown("""
<style>
.kpi-card {
    background-color: #111;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.kpi-title {
    font-size: 16px;
    color: #aaa;
}
.kpi-value {
    font-size: 32px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

col1.markdown(f"""
<div class="kpi-card">
    <div class="kpi-title">High Risk Zones</div>
    <div class="kpi-value" style="color:red;">{high}</div>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="kpi-card">
    <div class="kpi-title">Medium Risk Zones</div>
    <div class="kpi-value" style="color:orange;">{medium}</div>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="kpi-card">
    <div class="kpi-title">Low Risk Zones</div>
    <div class="kpi-value" style="color:lightgreen;">{low}</div>
</div>
""", unsafe_allow_html=True)

risk_df = assign_zone_coordinates(risk_df)

def get_color(level):
    if level == "HIGH":
        return [255, 0, 0]
    elif level == "MEDIUM":
        return [255, 165, 0]
    else:
        return [0, 200, 0]

risk_df["color"] = risk_df["risk_level"].apply(get_color)

# Map
layer = pdk.Layer(
    "ScatterplotLayer",
    data=risk_df,
    get_position='[lon, lat]',
    get_color='color',
    get_radius=500,
    pickable=True
)

view_state = pdk.ViewState(
    latitude=12.9716,
    longitude=77.5946,
    zoom=11
)
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Zone level Risk Indication")
st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={
        "text": """
        Zone: {zone_id}
        Risk: {risk_level}
        Risk Score: {risk_score}
        Demand Ratio: {demand_ratio}
        Anomalies: {anomaly_count}
        """
    }
))
st.markdown('</div>', unsafe_allow_html=True)



st.markdown("""
<style>
.section-box {
    background-color: #0e1117;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Priority Alerts - Zone Level")
alerts = risk_df.sort_values("risk_score", ascending=False).head(5)
display_df = alerts[[
    "zone_id",
    "risk_level",
    "risk_score",
    "demand_ratio",
    "anomaly_count" 
]].rename(
    columns ={
        "zone_id" : "Zone",
        "risk_level" : "Risk Level",
        "risk_score" : "Risk Score",
        "demand_ratio" : "Demand Ratio",
        "anomaly_count" : "Anomaly Count"
    }
)
st.dataframe(display_df)
st.markdown('</div>', unsafe_allow_html=True)

st.subheader("Risk Score - Zone Level")
st.bar_chart(risk_df.set_index("zone_id")["risk_score"])

st.subheader("Forecast vs Actual - Zone wise")
zone_selected = st.selectbox("Select Zone", risk_df["zone_id"].unique())
zone_data = forecast_df[forecast_df["zone_id"] == zone_selected]
st.line_chart(
    zone_data.set_index("timestamp")[["consumption", "prediction"]]
)

st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Model Explainability using SHAP")
st.image("data/shap_summary.png", use_container_width=True)
st.markdown("""
<div style='text-align:center; font-size:14px; color:gray;'>
Higher values indicate stronger influence on prediction
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) 