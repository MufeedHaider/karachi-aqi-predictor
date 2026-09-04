"""
Karachi AQI dashboard.

Every number on these pages is read from what the pipeline writes:
models/all_results.json, models/horizon_results.json, models/shap_importance.csv,
data/forecast_72hr.csv and data/recent_history.csv. Nothing is hardcoded.

That is a change from an earlier version, which carried literal copies of the
metrics and SHAP scores in its source. The retraining job ran every day and the
dashboard kept showing the numbers from the day they were typed in.
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from aqi import advice, aqi_from_pm25, color, short_category  # noqa: E402
from freshness import describe_age, is_stale  # noqa: E402

st.set_page_config(
    page_title="Karachi AQI",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")

PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", size=11),
    margin=dict(l=0, r=0, t=8, b=0),
)
GRID = "rgba(255,255,255,0.05)"


@st.cache_data(ttl=900)
def load_csv(name, folder=DATA):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["timestamp"] if "timestamp" in
                       open(path).readline() else None)


@st.cache_data(ttl=900)
def load_json(name):
    path = os.path.join(MODELS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=900)
def load_shap():
    path = os.path.join(MODELS, "shap_importance.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


fc_df = load_csv("forecast_72hr.csv")
hist_df = load_csv("recent_history.csv")
results = load_json("all_results.json")
horizons = load_json("horizon_results.json")
shap_df = load_shap()

if fc_df is None or hist_df is None:
    st.error(
        "No pipeline output found. Run the pipeline first:\n\n"
        "```\npython src/fetch_data.py\npython src/ground_truth.py --history 400\n"
        "python src/feature_engineering.py\npython src/train_model.py\n"
        "python src/forecast_model.py\npython src/explain_model.py\n```"
    )
    st.stop()

latest = hist_df.iloc[-1]
reading_time = pd.to_datetime(latest["timestamp"])
measured = float(latest["ground_pm25"])
cams_now = float(latest["cams_pm25"]) if "cams_pm25" in hist_df.columns else None

# If the monitor network has gone quiet, say so rather than presenting an old
# number as current. This is the guard that the dead WAQI station taught us to
# build.
stale = is_stale(reading_time, fc_df["timestamp"].min() - pd.Timedelta(hours=1),
                 max_lag_hours=12)

cur_aqi = aqi_from_pm25(measured)
aqi_color = color(cur_aqi)


def fmt(value, spec, suffix=""):
    if value is None or pd.isna(value):
        return "—"
    return f"{format(float(value), spec)}{suffix}"


st.markdown(
    "<style>#MainMenu, footer, header {visibility: hidden}"
    ".block-container {padding-top: 1rem !important}</style>",
    unsafe_allow_html=True,
)

brand, nav, clock = st.columns([2, 3, 2])
with brand:
    st.markdown("### 🌫️ Karachi AQI")
with nav:
    page = st.radio(
        "Navigation", ["Dashboard", "Forecast", "Analysis"],
        horizontal=True, label_visibility="collapsed", key="nav",
    )
with clock:
    st.markdown(
        f"<div style='text-align:right;padding-top:8px;font-size:12px;color:#64748b'>"
        f"Measured {reading_time:%b %d, %H:%M}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        st.markdown(
            f"""
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;
padding:24px;text-align:center">
  <div style="font-size:64px;font-weight:800;color:{aqi_color};
  font-family:monospace;line-height:1">{cur_aqi}</div>
  <div style="font-size:12px;font-weight:600;color:{aqi_color};
  letter-spacing:.1em;text-transform:uppercase;margin-top:6px">{short_category(cur_aqi)}</div>
  <div style="font-size:10px;color:#475569;margin-top:4px">US EPA AQI (2024 scale)</div>
</div>""",
            unsafe_allow_html=True,
        )
        st.caption(
            f"From {int(latest['n_stations'])} ground monitors (median)"
            if "n_stations" in hist_df.columns else "From ground monitors"
        )
        if stale:
            st.caption(
                f"⚠️ Monitor network last reported "
                f"{describe_age(reading_time, fc_df['timestamp'].min())}."
            )

    with c2:
        st.markdown(
            f"""
<div style="background:#0f172a;border:1px solid #1e293b;border-left:3px solid {aqi_color};
border-radius:12px;padding:20px;height:100%">
  <div style="font-size:11px;font-weight:600;color:{aqi_color};
  letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px">Health advisory</div>
  <div style="font-size:13px;color:#94a3b8;line-height:1.6">{advice(cur_aqi)}</div>
</div>""",
            unsafe_allow_html=True,
        )

    with c3:
        a, b, c, d = st.columns(4)
        a.metric("🌡️ Temp", fmt(latest.get("temperature"), ".1f", "°C"))
        b.metric("💧 Humidity", fmt(latest.get("humidity"), ".0f", "%"))
        c.metric("💨 Wind", fmt(latest.get("wind_speed"), ".1f", " km/h"))
        d.metric("⬇️ Pressure", fmt(latest.get("pressure"), ".0f", " hPa"))
        e, f, g, _ = st.columns(4)
        e.metric("🌫️ PM2.5 measured", fmt(measured, ".1f"))
        f.metric("🛰️ PM2.5 per CAMS", fmt(cams_now, ".1f"))
        if cams_now:
            g.metric("Gap", fmt(measured - cams_now, "+.1f"),
                     help="How far the CAMS simulation sits from the monitors")
        st.caption(
            "Measured values come from Karachi's ground monitor network. "
            "CAMS is the Copernicus atmospheric model, shown for comparison."
        )

    st.divider()
    st.subheader("📅 72-hour forecast")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc_df["timestamp"], y=fc_df["aqi_predicted"], mode="lines", name="AQI",
        line=dict(color=aqi_color, width=2), fill="tozeroy",
        fillcolor="rgba(148,163,184,0.07)",
        hovertemplate="<b>%{x|%b %d %H:%M}</b><br>AQI %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["timestamp"], y=fc_df["pm2_5_predicted"], mode="lines",
        name="PM2.5 forecast", line=dict(color="#60a5fa", width=1.5, dash="dot"),
        yaxis="y2", hovertemplate="PM2.5 %{y:.1f} µg/m³<extra></extra>",
    ))
    if "cams_pm2_5" in fc_df.columns:
        fig.add_trace(go.Scatter(
            x=fc_df["timestamp"], y=fc_df["cams_pm2_5"], mode="lines",
            name="CAMS forecast", line=dict(color="#f87171", width=1.2, dash="dash"),
            yaxis="y2", hovertemplate="CAMS %{y:.1f} µg/m³<extra></extra>",
        ))
    for lo, hi, band in [(0, 50, "rgba(0,230,118,0.06)"), (50, 100, "rgba(255,234,0,0.06)"),
                         (100, 150, "rgba(255,145,0,0.06)"), (150, 200, "rgba(255,23,68,0.06)")]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=band, line_width=0, layer="below")
    fig.update_layout(
        height=330, hovermode="x unified",
        xaxis=dict(gridcolor=GRID),
        yaxis=dict(gridcolor=GRID, title="AQI"),
        yaxis2=dict(overlaying="y", side="right", title="PM2.5 µg/m³",
                    showgrid=False, tickfont=dict(color="#60a5fa")),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12),
        **PLOT,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "The red dashed line is what CAMS alone predicts. The gap between it and "
        "the blue line is the correction this model applies."
    )

    if horizons and horizons.get("degraded_inputs"):
        st.warning(
            f"Produced without live {', '.join(horizons['degraded_inputs'])} "
            "forecast — last observed values were carried forward. Accuracy will "
            "be below the backtested figures."
        )

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.subheader("📈 Last 30 days — measured vs CAMS")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hist_df["timestamp"], y=hist_df["ground_pm25"], mode="lines",
            name="Measured", line=dict(color="#60a5fa", width=1.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        ))
        if "cams_pm25" in hist_df.columns:
            fig2.add_trace(go.Scatter(
                x=hist_df["timestamp"], y=hist_df["cams_pm25"], mode="lines",
                name="CAMS", line=dict(color="#f87171", width=1.2, dash="dash"),
            ))
        fig2.update_layout(
            height=290, hovermode="x unified",
            xaxis=dict(gridcolor=GRID),
            yaxis=dict(gridcolor=GRID, title="PM2.5 µg/m³", rangemode="tozero"),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.15),
            **PLOT,
        )
        st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

    with right:
        st.subheader("🕐 Next 24 hours")
        nxt = fc_df.head(24).copy()
        nxt["Time"] = nxt["timestamp"].dt.strftime("%a %H:%M")
        nxt["AQI"] = nxt["aqi_predicted"].astype(int)
        nxt["PM2.5"] = nxt["pm2_5_predicted"].round(1)
        st.dataframe(
            nxt[["Time", "AQI", "PM2.5", "aqi_category"]].rename(
                columns={"aqi_category": "Category"}),
            width="stretch", hide_index=True, height=290,
        )

    if shap_df is not None:
        st.divider()
        st.subheader("🔍 What drives the 24-hour forecast")
        top = shap_df.head(10).iloc[::-1]
        fig3 = go.Figure(go.Bar(
            x=top["mean_shap"], y=top["label"], orientation="h",
            marker_color="#60a5fa", marker_line_width=0,
            text=[f"{v:.2f}" for v in top["mean_shap"]],
            textposition="outside", textfont=dict(color="#94a3b8", size=10),
        ))
        fig3.update_layout(
            height=320, showlegend=False, margin=dict(l=0, r=70, t=8, b=0),
            xaxis=dict(gridcolor=GRID, title="Mean |SHAP| (µg/m³)"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=11),
        )
        st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})

# ═════════════════════════════════════════════════════════════════════════════
elif page == "Forecast":
    st.subheader("📅 72-hour detailed forecast")

    if horizons:
        h = horizons["horizons"]
        cols = st.columns(4)
        cols[0].metric("Mean skill vs CAMS",
                       f"{horizons['mean_skill_vs_cams']:.0%}",
                       help="Error reduction against the Copernicus model's own forecast")
        for col, key in zip(cols[1:], ["24", "48", "72"]):
            if key in h:
                col.metric(f"{key}h error", f"{h[key]['MAE']:.2f} µg/m³",
                           f"{h[key]['skill_vs_cams']:.0%} vs CAMS")
        st.caption(
            "Errors are from a rolling-origin backtest — the model is retrained at "
            "five points in time and scored only on the period after each, so every "
            "season is tested on data the model never saw."
        )

    st.divider()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc_df["timestamp"], y=fc_df["pm2_5_predicted"], mode="lines",
        name="This model", line=dict(color="#60a5fa", width=2.5),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
    ))
    if "cams_pm2_5" in fc_df.columns:
        fig.add_trace(go.Scatter(
            x=fc_df["timestamp"], y=fc_df["cams_pm2_5"], mode="lines",
            name="CAMS forecast", line=dict(color="#f87171", width=1.5, dash="dash"),
        ))
    fig.update_layout(
        height=380, hovermode="x unified",
        xaxis=dict(gridcolor=GRID),
        yaxis=dict(gridcolor=GRID, title="PM2.5 µg/m³", rangemode="tozero"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1),
        **PLOT,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.divider()
    st.subheader("📋 Hour by hour")
    table = fc_df.copy()
    table["Time"] = table["timestamp"].dt.strftime("%b %d, %H:%M")
    table["AQI"] = table["aqi_predicted"].astype(int)
    table["PM2.5"] = table["pm2_5_predicted"].round(1)
    table["CAMS"] = table["cams_pm2_5"].round(1) if "cams_pm2_5" in table else None
    table["Lead"] = table["horizon_hour"].astype(int).astype(str) + "h"
    cols = ["Time", "AQI", "PM2.5", "CAMS", "aqi_category", "Lead"]
    cols = [c for c in cols if c in table.columns and table[c].notna().any()]
    st.dataframe(
        table[cols].rename(columns={"aqi_category": "Category"}),
        width="stretch", hide_index=True, height=420,
    )

# ═════════════════════════════════════════════════════════════════════════════
elif page == "Analysis":
    st.subheader("🔬 Model analysis")

    if results:
        c = st.columns(4)
        c[0].metric("Selected model", results["best_model"])
        c[1].metric("Features", results["n_features"])
        c[2].metric("Training rows", f"{results['n_rows']:,}")
        c[3].metric("Selection horizon", f"{results['selection_horizon_hours']}h")
        st.caption(
            f"Target: {results['target']}. "
            f"Period {results['period_start'][:10]} to {results['period_end'][:10]}. "
            f"Evaluation: {results['evaluation']}."
        )

        st.divider()
        st.subheader("📊 Models and baselines")
        st.caption(
            "Three baselines, all scored on identical folds. CAMS is the one that "
            "matters: it is a real operational forecast from the Copernicus "
            "atmospheric model, not a strawman. Beating it is the test of whether "
            "this project adds anything to the physics."
        )
        for horizon_key, table in results["by_horizon"].items():
            rows = [
                {
                    "Model": name,
                    "MAE (µg/m³)": f"{m['MAE']:.2f}",
                    "RMSE": f"{m['RMSE']:.2f}",
                    "R²": f"{m['R2']:.3f}",
                    "vs persistence": f"{m['skill_vs_persistence']:.1%}",
                    "vs CAMS": f"{m.get('skill_vs_cams', 0):.1%}",
                }
                for name, m in table.items()
            ]
            st.markdown(f"**{horizon_key.replace('hr', '-hour')} ahead**")
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if horizons:
        st.divider()
        st.subheader("📉 Error growth with lead time")
        h = horizons["horizons"]
        keys = sorted(h, key=lambda k: int(k))
        fig4 = go.Figure()
        for label, key, colr, dash in [
            ("This model", "MAE", "#60a5fa", None),
            ("CAMS forecast", "cams_MAE", "#f87171", "dash"),
        ]:
            fig4.add_trace(go.Scatter(
                x=[int(k) for k in keys], y=[h[k].get(key) for k in keys],
                mode="lines+markers", name=label,
                line=dict(color=colr, width=2, dash=dash),
            ))
        fig4.update_layout(
            height=330,
            xaxis=dict(gridcolor=GRID, title="Lead time (hours)"),
            yaxis=dict(gridcolor=GRID, title="MAE (µg/m³)", rangemode="tozero"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            **PLOT,
        )
        st.plotly_chart(fig4, width="stretch", config={"displayModeBar": False})

        folds = horizons.get("folds", {})
        if "24" in folds:
            st.subheader("Per-season detail (24-hour horizon)")
            st.caption(
                "One row per backtest fold. Reported in full, including the fold "
                "where the model does not beat persistence."
            )
            st.dataframe(
                pd.DataFrame([
                    {
                        "Test window": f["window"],
                        "Observed mean": f["mean_observed"],
                        "MAE": f["MAE"],
                        "vs persistence": f"{f['skill_vs_persistence']:.1%}",
                        "vs CAMS": f"{f.get('skill_vs_cams', 0):.1%}",
                    }
                    for f in folds["24"]
                ]),
                width="stretch", hide_index=True,
            )

    if shap_df is not None:
        st.divider()
        st.subheader("🔍 SHAP feature importance")
        st.caption(
            f"Computed on the {int(shap_df['horizon_hours'].iloc[0])}-hour model, "
            "in µg/m³ of predicted change."
        )
        top = shap_df.head(15).iloc[::-1]
        fig5 = go.Figure(go.Bar(
            x=top["mean_shap"], y=top["label"], orientation="h",
            marker_color="#60a5fa", marker_line_width=0,
            text=[f"{v:.2f}" for v in top["mean_shap"]],
            textposition="outside", textfont=dict(color="#94a3b8", size=10),
        ))
        fig5.update_layout(
            height=470, showlegend=False, margin=dict(l=0, r=90, t=8, b=0),
            xaxis=dict(gridcolor=GRID, title="Mean |SHAP value|"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=12),
        )
        st.plotly_chart(fig5, width="stretch", config={"displayModeBar": False})
        st.caption(
            "The top drivers are the current gap between measurement and CAMS, and "
            "the CAMS forecast itself — the model uses the physics and corrects its "
            "bias at the same time."
        )
