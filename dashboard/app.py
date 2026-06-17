import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys, json

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

st.set_page_config(
    page_title="Karachi AQI",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── helpers ──────────────────────────────────────────────────
def aqi_from_pm25(pm25):
    # Updated EPA 2024 breakpoints (effective May 6, 2024)
    if pd.isna(pm25): return 0
    v = float(pm25)
    for lo, hi, alo, ahi in [
        (0.0,   9.0,   0,  50),
        (9.1,  35.4,  51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5,225.4, 201, 300),
        (225.5, 500,  301, 500),
    ]:
        if lo <= v <= hi:
            return round(((ahi - alo) / (hi - lo)) * (v - lo) + alo)
    return 500

def aqi_color(a):
    if a<=50:  return "#00e676"
    if a<=100: return "#ffea00"
    if a<=150: return "#ff9100"
    if a<=200: return "#ff1744"
    if a<=300: return "#d500f9"
    return "#b71c1c"

def aqi_label(a):
    if a<=50:  return "Good"
    if a<=100: return "Moderate"
    if a<=150: return "Unhealthy (Sensitive)"
    if a<=200: return "Unhealthy"
    if a<=300: return "Very Unhealthy"
    return "Hazardous"

def aqi_advice(a):
    if a<=50:  return "Air quality is excellent. Safe for all activities."
    if a<=100: return "Acceptable. Sensitive individuals take caution outdoors."
    if a<=150: return "Sensitive groups limit prolonged outdoor exposure."
    if a<=200: return "Everyone reduce outdoor activity. Wear a mask."
    if a<=300: return "Health alert — avoid all outdoor activity."
    return "Emergency — stay indoors, seal windows immediately."

# ── load data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    feat = pd.read_csv(
        os.path.join(ROOT,"data","featured_data.csv"),
        parse_dates=["timestamp"]
    )
    fc = pd.read_csv(
        os.path.join(ROOT,"data","forecast_72hr.csv"),
        parse_dates=["timestamp"]
    )
    for c in ["pm2_5","pm10","no2","ozone","co","so2",
              "temperature","humidity","wind_speed","pressure"]:
        if c in feat.columns:
            feat[c] = pd.to_numeric(feat[c], errors="coerce")
    feat["aqi"] = feat["pm2_5"].apply(aqi_from_pm25)
    fc["aqi_predicted"]   = pd.to_numeric(fc["aqi_predicted"],   errors="coerce")
    fc["pm2_5_predicted"] = pd.to_numeric(fc["pm2_5_predicted"], errors="coerce")
    return feat, fc

feat_df, fc_df = load_data()
import os as _os
_waqi_path = _os.path.join(ROOT, "data", "waqi_live.csv")

if _os.path.exists(_waqi_path):
    waqi_live  = pd.read_csv(_waqi_path, parse_dates=["timestamp"]).iloc[-1]
    cur_aqi    = int(waqi_live["aqi_live"])
    pm25_now   = round(float(waqi_live["pm25_live"]), 1)
    temp       = round(float(waqi_live.get("temperature", 0) or 0), 1)
    humidity   = int(float(waqi_live.get("humidity", 0) or 0))
    wind_speed = round(float(waqi_live.get("wind_speed", 0) or 0), 1)
    pressure   = int(float(waqi_live.get("pressure", 0) or 0))
    data_source = "🏛️ US Embassy Karachi Ground Station"
else:
    row        = feat_df.iloc[-1]
    cur_aqi    = int(aqi_from_pm25(float(row["pm2_5"])))
    pm25_now   = round(float(row["pm2_5"] or 0), 1)
    temp       = round(float(row.get("temperature", 0) or 0), 1)
    humidity   = int(float(row.get("humidity", 0) or 0))
    wind_speed = round(float(row.get("wind_speed", 0) or 0), 1)
    pressure   = int(float(row.get("pressure", 0) or 0))
    data_source = "Open-Meteo Model"

pm10_now = round(float(feat_df.iloc[-1]["pm10"] or 0), 1)
color    = aqi_color(cur_aqi)
label    = aqi_label(cur_aqi)
advice   = aqi_advice(cur_aqi)
color   = aqi_color(cur_aqi)
label   = aqi_label(cur_aqi)
advice  = aqi_advice(cur_aqi)

temp       = round(float(row.get("temperature", 0) or 0), 1)
humidity   = int(float(row.get("humidity", 0) or 0))
wind_speed = round(float(row.get("wind_speed", 0) or 0), 1)
pressure   = int(float(row.get("pressure", 0) or 0))
pm25_now   = round(float(row["pm2_5"] or 0), 1)
pm10_now   = round(float(row["pm10"]  or 0), 1)

# ── HIDE streamlit chrome ────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden}
.block-container {padding-top: 1rem !important}
</style>
""", unsafe_allow_html=True)

# ── NAVIGATION ───────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

col_brand, col_nav, col_live = st.columns([2, 3, 2])
with col_brand:
    st.markdown(f"### 🌫️ Karachi AQI")
with col_nav:
    tabs = st.radio(
        "", ["Dashboard", "Forecast", "Analysis"],
        horizontal=True,
        label_visibility="collapsed",
        key="nav"
    )
with col_live:
    st.markdown(
        f"<div style='text-align:right;padding-top:8px;font-size:12px;color:#64748b'>● Live · Open-Meteo</div>",
        unsafe_allow_html=True
    )

st.divider()
page = tabs

# ═══════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════
if page == "Dashboard":

    # current AQI hero
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        st.markdown(f"""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;
    padding:24px;text-align:center">
      <div style="font-size:64px;font-weight:800;color:{color};
      font-family:monospace;line-height:1">{cur_aqi}</div>
      <div style="font-size:12px;font-weight:600;color:{color};
      letter-spacing:.1em;text-transform:uppercase;margin-top:6px">{label}</div>
      <div style="font-size:10px;color:#475569;margin-top:4px">US EPA AQI</div>
</div>""", unsafe_allow_html=True)
    st.caption(f"Source: {data_source}")

    with c2:
        st.markdown(f"""
<div style="background:#0f172a;border:1px solid #1e293b;border-left:3px solid {color};
border-radius:12px;padding:20px;height:100%">
  <div style="font-size:11px;font-weight:600;color:{color};
  letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px">Health Advisory</div>
  <div style="font-size:13px;color:#94a3b8;line-height:1.6">{advice}</div>
</div>""", unsafe_allow_html=True)

    with c3:
        w1,w2,w3,w4 = st.columns(4)
        w1.metric("🌡️ Temp",    f"{temp}°C")
        w2.metric("💧 Humidity", f"{humidity}%")
        w3.metric("💨 Wind",     f"{wind_speed} km/h")
        w4.metric("⬇️ Pressure", f"{pressure} hPa")
        w5,w6,_,_ = st.columns(4)
        w5.metric("🌫️ PM2.5",   f"{pm25_now} µg/m³")
        w6.metric("🟤 PM10",    f"{pm10_now} µg/m³")

    st.divider()

    # 72hr forecast chart
    st.subheader("📅 72-Hour AQI Forecast")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=fc_df["timestamp"],
        y=fc_df["aqi_predicted"],
        mode="lines",
        name="AQI Forecast",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba(255,234,0,0.07)",
        hovertemplate="<b>%{x|%b %d %H:%M}</b><br>AQI: %{y}<extra></extra>"
    ))
    fig1.add_trace(go.Scatter(
        x=fc_df["timestamp"],
        y=fc_df["pm2_5_predicted"],
        mode="lines",
        name="PM2.5 µg/m³",
        line=dict(color="#60a5fa", width=1.5, dash="dot"),
        yaxis="y2",
        hovertemplate="PM2.5: %{y:.1f} µg/m³<extra></extra>"
    ))
    # AQI category zones
    for lo,hi,c2_,nm in [
        (0,50,"rgba(0,230,118,0.06)","Good"),
        (50,100,"rgba(255,234,0,0.06)","Moderate"),
        (100,150,"rgba(255,145,0,0.06)","Unhealthy-S"),
        (150,200,"rgba(255,23,68,0.06)","Unhealthy")
    ]:
        fig1.add_hrect(y0=lo, y1=hi, fillcolor=c2_, line_width=0, layer="below")

    fig1.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", size=11),
        margin=dict(l=0,r=0,t=8,b=0),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True, title="AQI"),
        yaxis2=dict(overlaying="y", side="right", title="PM2.5 µg/m³",
                    showgrid=False, tickfont=dict(color="#60a5fa")),
    )
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # 7-day history + 24hr table
    col_h, col_t = st.columns([3, 2])

    with col_h:
        st.subheader("📈 7-Day Historical AQI")
        h7 = feat_df[
            feat_df["timestamp"] >= feat_df["timestamp"].max() - pd.Timedelta(days=7)
        ].copy()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=h7["timestamp"],
            y=h7["aqi"],
            mode="lines",
            name="AQI",
            line=dict(color="#60a5fa", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.08)",
            hovertemplate="<b>%{x|%b %d %H:%M}</b><br>AQI: %{y}<extra></extra>"
        ))
        fig2.add_hline(
            y=100, line_dash="dash",
            line_color="rgba(255,145,0,0.4)",
            annotation_text="Moderate threshold",
            annotation_font_color="rgba(255,145,0,0.7)",
            annotation_font_size=10
        )
        fig2.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=11),
            margin=dict(l=0,r=0,t=8,b=0),
            showlegend=False,
            hovermode="x unified",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="AQI", rangemode="tozero"),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with col_t:
        st.subheader("🕐 Next 24 Hours")
        next24 = fc_df.head(24).copy()
        next24["Time"] = next24["timestamp"].dt.strftime("%a %H:%M")
        next24["AQI"]  = next24["aqi_predicted"].astype(int)
        next24["PM2.5"]= next24["pm2_5_predicted"].round(1)
        next24["Category"] = next24["aqi_category"]
        st.dataframe(
            next24[["Time","AQI","PM2.5","Category"]],
            use_container_width=True,
            hide_index=True,
            height=280
        )

    st.divider()

    # Pollutants + SHAP
    col_p, col_s = st.columns(2)

    with col_p:
        st.subheader("🧪 Pollutant Levels")
        poll_labels = ["PM2.5","PM10","NO2","SO2","Ozone"]
        poll_vals   = [
            round(float(row["pm2_5"] or 0),1),
            round(float(row["pm10"]  or 0),1),
            round(float(row["no2"]   or 0),1),
            round(float(row["so2"]   or 0),1),
            round(float(row["ozone"] or 0),1),
        ]
        who_limits = [15, 45, 25, 40, 100]
        poll_colors = ["#FF6B35","#FFA500","#4CAF50","#F44336","#2196F3"]

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=poll_labels, y=poll_vals,
            marker_color=poll_colors,
            marker_line_width=0,
            name="Current",
            text=[f"{v}" for v in poll_vals],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
        ))
        # WHO limit dots
        fig3.add_trace(go.Scatter(
            x=poll_labels, y=who_limits,
            mode="markers",
            marker=dict(symbol="line-ew", size=20,
                       line=dict(color="rgba(255,255,255,0.5)", width=2)),
            name="WHO limit",
        ))
        fig3.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=11),
            margin=dict(l=0,r=0,t=8,b=0),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="µg/m³", rangemode="tozero"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            barmode="group",
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with col_s:
        st.subheader("🔍 What Drives the Forecast (SHAP)")
        shap_labels = ["3hr Rolling Avg","PM10","PM Ratio","Wind Dispersion",
                       "PM2.5 (3hr lag)","AQI Rolling Avg","PM2.5 (1hr lag)",
                       "Wind Speed","Rush Hour","AQI (3hr lag)"]
        shap_vals   = [7.09,1.96,1.19,0.77,0.65,0.47,0.35,0.33,0.32,0.18]
        shap_colors = [color if v>1 else "#60a5fa" for v in shap_vals]

        fig4 = go.Figure(go.Bar(
            x=shap_vals[::-1],
            y=shap_labels[::-1],
            orientation="h",
            marker_color=shap_colors[::-1],
            marker_line_width=0,
            text=[f"{v:.2f}" for v in shap_vals[::-1]],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=10),
        ))
        fig4.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=11),
            margin=dict(l=0,r=60,t=8,b=0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Mean |SHAP value|"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════
# FORECAST
# ═══════════════════════════════════════════════════
elif page == "Forecast":
    st.subheader("📅 72-Hour Detailed Forecast")

    # accuracy metrics
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Model", "XGBoost")
    m2.metric("24hr MAE", "7.5 µg/m³")
    m3.metric("48hr MAE", "8.4 µg/m³")
    m4.metric("72hr MAE", "8.8 µg/m³")

    st.caption("MAE increases with forecast horizon — this is expected and shows the model is not overfit.")
    st.divider()

    # big forecast chart
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=fc_df["timestamp"],
        y=fc_df["aqi_predicted"],
        mode="lines",
        name="AQI",
        line=dict(color=color, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(255,234,0,0.07)",
        hovertemplate="<b>%{x|%b %d %H:%M}</b><br>AQI: %{y}<extra></extra>"
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_df["timestamp"],
        y=fc_df["pm2_5_predicted"],
        mode="lines",
        name="PM2.5 µg/m³",
        line=dict(color="#60a5fa", width=1.5, dash="dot"),
        yaxis="y2",
        hovertemplate="PM2.5: %{y:.1f} µg/m³<extra></extra>"
    ))
    for lo,hi,c2_,nm in [
        (0,50,"rgba(0,230,118,0.06)","Good"),
        (50,100,"rgba(255,234,0,0.06)","Moderate"),
        (100,150,"rgba(255,145,0,0.06)","Unhealthy-S"),
        (150,200,"rgba(255,23,68,0.06)","Unhealthy")
    ]:
        fig_fc.add_hrect(y0=lo, y1=hi, fillcolor=c2_, line_width=0, layer="below")

    fig_fc.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", size=11),
        margin=dict(l=0,r=0,t=8,b=0),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="AQI", rangemode="tozero"),
        yaxis2=dict(overlaying="y", side="right", title="PM2.5 µg/m³",
                    showgrid=False, tickfont=dict(color="#60a5fa")),
    )
    st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": False})

    st.divider()
    st.subheader("📋 Complete 72-Hour Breakdown")

    full_fc = fc_df.copy()
    full_fc["Time"]     = full_fc["timestamp"].dt.strftime("%b %d, %Y  %H:%M")
    full_fc["AQI"]      = full_fc["aqi_predicted"].astype(int)
    full_fc["PM2.5"]    = full_fc["pm2_5_predicted"].round(1)
    full_fc["Category"] = full_fc["aqi_category"]
    full_fc["Horizon"]  = full_fc["horizon_hour"].astype(int).astype(str) + "h ahead"

    st.dataframe(
        full_fc[["Time","AQI","PM2.5","Category","Horizon"]],
        use_container_width=True,
        hide_index=True,
        height=420
    )

# ═══════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════
elif page == "Analysis":
    st.subheader("🔬 Model Analysis & Explainability")

    # model metrics
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Algorithm", "XGBoost")
    m2.metric("R² Score",  "0.978")
    m3.metric("MAE (1-step)", "0.779 µg/m³")
    m4.metric("Training Data", "8,712 rows")

    st.divider()

    # model comparison table
    st.subheader("📊 Model Comparison")
    comparison = pd.DataFrame({
        "Model":      ["Ridge Regression", "Random Forest", "XGBoost ✓"],
        "MAE (µg/m³)":["0.853",            "0.937",         "0.779"],
        "RMSE":       ["1.332",            "1.622",         "1.378"],
        "R² Score":   ["0.980",            "0.970",         "0.978"],
        "Notes":      ["Linear baseline",  "Ensemble trees","Best overall — selected"]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.divider()

    # SHAP full chart
    st.subheader("🔍 SHAP Feature Importance — Full Breakdown")
    st.caption("Higher score = feature has more impact on the model's PM2.5 prediction")

    shap_labels = ["3hr Rolling Avg","PM10","PM Ratio","Wind Dispersion",
                   "PM2.5 (3hr lag)","AQI Rolling Avg","PM2.5 (1hr lag)",
                   "Wind Speed","Rush Hour","AQI (3hr lag)"]
    shap_vals   = [7.09,1.96,1.19,0.77,0.65,0.47,0.35,0.33,0.32,0.18]
    shap_colors = [color if v>1 else "#60a5fa" for v in shap_vals]

    fig_sh = go.Figure(go.Bar(
        x=shap_vals[::-1],
        y=shap_labels[::-1],
        orientation="h",
        marker_color=shap_colors[::-1],
        marker_line_width=0,
        text=[f"{v:.2f}" for v in shap_vals[::-1]],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig_sh.update_layout(
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", size=12),
        margin=dict(l=0,r=80,t=8,b=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Mean |SHAP value|"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    st.plotly_chart(fig_sh, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # feature explanations
    st.subheader("💡 Why These Features Matter")
    e1,e2,e3 = st.columns(3)
    with e1:
        st.info("**3hr Rolling Avg — Score 7.09**\n\nPM2.5 has strong momentum. If it's been high for 3 hours, it stays high. This is the single strongest predictor.")
    with e2:
        st.info("**PM10 — Score 1.96**\n\nCoarse and fine particles share the same sources — traffic, dust, industry. High PM10 reliably predicts high PM2.5.")
    with e3:
        st.info("**Wind Dispersion — Score 0.77**\n\nStrong wind physically disperses pollution. The model learned real atmospheric physics from the data.")

    st.divider()

    # feature engineering summary
    st.subheader("⚙️ Feature Engineering Summary")
    feat_table = pd.DataFrame({
        "Feature Type":   ["Lag Features","Rolling Averages","Time Features","Weather Features","Interaction Features"],
        "Count":          ["6","8","7","5","3"],
        "Examples":       [
            "pm2_5_lag1, lag3, lag6, lag12, lag24, lag48",
            "pm2_5_roll3, roll6, roll12, roll24 + AQI versions",
            "hour, day_of_week, month, is_weekend, is_rush_hour, hour_sin, hour_cos",
            "temperature, humidity, wind_speed, wind_dir, pressure",
            "pm_ratio, wind_dispersion, heat_humidity"
        ],
        "Purpose": [
            "Capture temporal autocorrelation",
            "Smooth short & long term trends",
            "Capture daily & weekly patterns",
            "Atmospheric dispersion context",
            "Non-linear pollutant relationships"
        ]
    })
    st.dataframe(feat_table, use_container_width=True, hide_index=True)