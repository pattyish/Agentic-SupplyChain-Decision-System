"""
streamlit_app.py
----------------
Interactive supply chain monitoring dashboard powered by Streamlit.

Tabs:
  1. Overview        — KPI cards, delay rate trend, transport breakdown
  2. Risk Map        — Risk score distribution, port congestion heatmap
  3. Anomalies       — Anomaly detection results table + scatter plot
  4. Port Congestion — Daily congestion time-series per port
  5. Predict         — Interactive single-shipment delay prediction

Run:
  streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# ── Project path ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import SupplyChainLoader
from src.anomaly.anomaly_detection import AnomalyDetector, congestion_alerts

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title   = "Supply Chain AI Monitor",
    page_icon    = "🚢",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# =============================================================================
# CSS / Styling
# =============================================================================

st.markdown("""
<style>
    .kpi-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        border-left: 5px solid;
        margin-bottom: 8px;
    }
    .kpi-value { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .kpi-label { font-size: 0.88rem; color: #aaa; margin: 0; }
    .alert-critical { color: #ff4b4b; font-weight: bold; }
    .alert-high      { color: #ff8c00; font-weight: bold; }
    .alert-elevated  { color: #ffd700; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data Loading (cached)
# =============================================================================

@st.cache_data(ttl=300, show_spinner="Loading data …")
def load_data() -> dict[str, pd.DataFrame]:
    loader = SupplyChainLoader(data_dir=ROOT / "data")
    return {
        "shipments":  loader.shipments(),
        "congestion": loader.port_congestion(),
        "weather":    loader.weather(),
        "suppliers":  loader.suppliers(),
        "disruptions":loader.disruptions(),
    }


@st.cache_data(ttl=300, show_spinner="Running anomaly detection …")
def get_anomaly_results(shipments_hash: int) -> pd.DataFrame:
    """Run anomaly detection (cached by data hash)."""
    loader    = SupplyChainLoader(data_dir=ROOT / "data")
    shipments = loader.shipments()
    det       = AnomalyDetector()
    det.fit(shipments)
    return det.predict(shipments)


# =============================================================================
# Sidebar — Filters
# =============================================================================

def render_sidebar(data: dict) -> dict:
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/cargo-ship.png", width=72)
    st.sidebar.title("Supply Chain AI Monitor")
    st.sidebar.markdown("---")

    shp = data["shipments"]
    min_date = pd.to_datetime(shp["ship_date"]).min().date()
    max_date = pd.to_datetime(shp["ship_date"]).max().date()

    st.sidebar.subheader("Filters")
    date_range = st.sidebar.date_input(
        "Ship Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    modes = st.sidebar.multiselect(
        "Transport Mode",
        options=["truck", "ship", "air"],
        default=["truck", "ship", "air"],
    )

    ports = data["congestion"]["port_id"].unique().tolist()
    sel_ports = st.sidebar.multiselect(
        "Port IDs",
        options=sorted(ports),
        default=sorted(ports)[:5],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Data: Synthetic (2023–2025) | Model: XGBoost + LSTM")

    return {"date_range": date_range, "modes": modes, "ports": sel_ports}


def apply_filters(shp: pd.DataFrame, filters: dict) -> pd.DataFrame:
    df = shp.copy()
    df["ship_date"] = pd.to_datetime(df["ship_date"])
    if len(filters["date_range"]) == 2:
        df = df[
            (df["ship_date"].dt.date >= filters["date_range"][0]) &
            (df["ship_date"].dt.date <= filters["date_range"][1])
        ]
    if filters["modes"]:
        df = df[df["transport_mode"].isin(filters["modes"])]
    return df


# =============================================================================
# KPI Cards
# =============================================================================

def render_kpis(df: pd.DataFrame) -> None:
    delayed_df = df[df["delayed"] == 1]
    cols = st.columns(5)

    metrics = [
        ("Total Shipments",    f"{len(df):,}",            "#4fc3f7", ""),
        ("Delayed",            f"{df['delayed'].sum():,}", "#ff4b4b", ""),
        ("Delay Rate",         f"{df['delayed'].mean()*100:.1f}%",  "#ff8c00", ""),
        ("Avg Delay (hrs)",    f"{delayed_df['delay_hours'].mean():.1f}",  "#ab47bc", ""),
        ("Avg Risk Score",     f"{df['supplier_risk'].mean():.3f}",  "#66bb6a", ""),
    ]

    for col, (label, value, color, _) in zip(cols, metrics):
        col.markdown(
            f"""<div class="kpi-card" style="border-color:{color}">
                <p class="kpi-value" style="color:{color}">{value}</p>
                <p class="kpi-label">{label}</p>
            </div>""",
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 1 — Overview
# =============================================================================

def tab_overview(df: pd.DataFrame) -> None:
    st.header("Supply Chain Overview")
    render_kpis(df)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Weekly delay rate trend
        df_trend = df.copy()
        df_trend["week"] = pd.to_datetime(df_trend["ship_date"]).dt.to_period("W").apply(lambda x: x.start_time)
        weekly = (
            df_trend.groupby("week")
            .agg(delay_rate=("delayed", "mean"), shipments=("shipment_id", "count"))
            .reset_index()
        )
        fig = px.line(
            weekly, x="week", y="delay_rate",
            labels={"week": "Week", "delay_rate": "Delay Rate"},
            title="Weekly Shipment Delay Rate",
        )
        fig.update_yaxes(tickformat=".0%")
        fig.add_hrect(y0=0.30, y1=weekly["delay_rate"].max() * 1.05,
                      fillcolor="red", opacity=0.07, line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Delay rate by transport mode
        mode_dr = (
            df.groupby("transport_mode")["delayed"]
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={"delayed": "delay_rate_pct"})
        )
        fig2 = px.bar(
            mode_dr, x="transport_mode", y="delay_rate_pct",
            color="transport_mode",
            labels={"delay_rate_pct": "Delay Rate (%)", "transport_mode": "Mode"},
            title="Delay Rate by Transport Mode",
            text="delay_rate_pct",
        )
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # Delay hours distribution
        delayed_df = df[df["delayed"] == 1]
        fig3 = px.histogram(
            delayed_df, x="delay_hours", nbins=60,
            log_y=True,
            title="Delay Hours Distribution (log scale)",
            labels={"delay_hours": "Delay Hours"},
            color_discrete_sequence=["#ff4b4b"],
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Weather severity vs delay rate
        ws_dr = (
            df.groupby("weather_severity")["delayed"]
            .mean()
            .mul(100)
            .reset_index()
        )
        fig4 = px.bar(
            ws_dr, x="weather_severity", y="delayed",
            labels={"weather_severity": "Weather Severity (0–3)", "delayed": "Delay Rate (%)"},
            title="Delay Rate by Weather Severity",
            color="weather_severity",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig4, use_container_width=True)


# =============================================================================
# TAB 2 — Risk Map
# =============================================================================

def tab_risk_map(df: pd.DataFrame) -> None:
    st.header("Risk Distribution Map")

    # Risk score computation
    df = df.copy()
    df["risk_score"] = (
        0.30 * (df["weather_severity"] / 3.0)
        + 0.25 * df["port_congestion"]
        + 0.30 * df["supplier_risk"]
        + 0.15 * (df["traffic_level"] / 5.0)
    ).clip(0, 1)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x="risk_score", color="transport_mode", nbins=50,
            barmode="overlay", opacity=0.75,
            title="Risk Score Distribution by Transport Mode",
            labels={"risk_score": "Risk Score (0–1)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk score vs delay hours scatter
        delayed_df = df[df["delayed"] == 1].sample(min(1500, len(df[df["delayed"]==1])))
        fig2 = px.scatter(
            delayed_df, x="risk_score", y="delay_hours",
            color="transport_mode", opacity=0.6, size_max=6,
            title="Risk Score vs Delay Hours (Delayed Shipments)",
            labels={"risk_score": "Risk Score", "delay_hours": "Delay Hours"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Port congestion heatmap
    st.subheader("Port Congestion Heatmap (Last 60 Days)")
    loader  = SupplyChainLoader(data_dir=ROOT / "data")
    cng     = loader.port_congestion()
    cng_dt  = cng.copy()
    cng_dt["date"] = pd.to_datetime(cng_dt["date"])
    cng_last = cng_dt[cng_dt["date"] >= cng_dt["date"].max() - pd.Timedelta(days=60)].copy()
    cng_last["date_str"] = cng_last["date"].dt.strftime("%Y-%m-%d")

    pivot = cng_last.pivot_table(
        index="location", columns="date_str", values="congestion_level", aggfunc="mean"
    )
    fig3 = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn_r",
        zmin=0, zmax=1,
        title="Daily Port Congestion — Last 60 Days",
        aspect="auto",
        labels={"color": "Congestion"},
    )
    fig3.update_xaxes(showticklabels=False)
    st.plotly_chart(fig3, use_container_width=True)


# =============================================================================
# TAB 3 — Anomaly Detection
# =============================================================================

def tab_anomalies(shp: pd.DataFrame) -> None:
    st.header("Anomaly Detection")

    with st.spinner("Running anomaly detection …"):
        det = AnomalyDetector()
        det.fit(shp)
        scored = det.predict(shp, threshold=0.55)

    anomalies = scored[scored["is_anomaly"] == 1].copy()

    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Anomalies", f"{len(anomalies):,}")
    col2.metric("Critical", int((anomalies["anomaly_level"] == "critical").sum()))
    col3.metric("High", int((anomalies["anomaly_level"] == "high").sum()))

    # Scatter: anomaly score vs delay hours
    fig = px.scatter(
        scored.sample(min(3000, len(scored))),
        x="anomaly_score", y="delay_hours",
        color="anomaly_level",
        color_discrete_map={
            "critical": "#ff4b4b", "high": "#ff8c00",
            "elevated": "#ffd700", "normal": "#66bb6a",
        },
        opacity=0.6,
        title="Anomaly Score vs Delay Hours",
        labels={"anomaly_score": "Ensemble Anomaly Score", "delay_hours": "Delay Hours"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Alert table
    st.subheader("Top Anomalous Shipments")
    disp_cols = ["shipment_id", "ship_date", "transport_mode", "delayed",
                 "delay_hours", "port_congestion", "supplier_risk",
                 "anomaly_score", "anomaly_level"]
    disp_cols = [c for c in disp_cols if c in anomalies.columns]
    top_anomalies = anomalies.sort_values("anomaly_score", ascending=False).head(30)

    def highlight_level(row):
        color_map = {"critical": "#ff4b4b33", "high": "#ff8c0033",
                     "elevated": "#ffd70033", "normal": ""}
        return [f"background-color: {color_map.get(row.get('anomaly_level', ''), '')}"
                for _ in row]

    st.dataframe(
        top_anomalies[disp_cols].reset_index(drop=True),
        use_container_width=True,
        height=420,
    )


# =============================================================================
# TAB 4 — Port Congestion
# =============================================================================

def tab_congestion(filters: dict) -> None:
    st.header("Port Congestion Analysis")

    loader = SupplyChainLoader(data_dir=ROOT / "data")
    cng    = loader.port_congestion()
    cng["date"] = pd.to_datetime(cng["date"])

    # Filter to selected ports
    if filters["ports"]:
        cng_filt = cng[cng["port_id"].isin(filters["ports"])]
    else:
        cng_filt = cng

    # Average daily congestion per port
    cng_daily = (
        cng_filt.groupby(["date", "location"])
        .agg(congestion_level=("congestion_level", "mean"),
             queue_time=("queue_time_hours", "mean"))
        .reset_index()
    )

    fig = px.line(
        cng_daily, x="date", y="congestion_level",
        color="location",
        title="Port Congestion Level Over Time",
        labels={"congestion_level": "Congestion (0–1)", "date": "Date"},
    )
    fig.add_hrect(y0=0.80, y1=1.0, fillcolor="red", opacity=0.12, line_width=0,
                  annotation_text="Critical", annotation_position="top right")
    fig.add_hrect(y0=0.65, y1=0.80, fillcolor="orange", opacity=0.08, line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    # Current alerts
    alerts = congestion_alerts(cng, _load_config())
    if len(alerts) > 0:
        st.subheader("Active Port Alerts")
        for _, row in alerts.head(10).iterrows():
            lvl = row.get("alert_level", "elevated")
            cls = f"alert-{lvl}"
            st.markdown(
                f"<span class='{cls}'>⚠ Port {row['port_id']} ({row.get('location', '')})"
                f" — Congestion: {row['congestion_level']:.2f} [{lvl.upper()}]</span>",
                unsafe_allow_html=True,
            )
    else:
        st.success("No ports currently in elevated alert state.")


# =============================================================================
# TAB 5 — Interactive Prediction
# =============================================================================

def _load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def _compute_risk(weather, congestion, supplier_risk, traffic) -> float:
    return min(1.0, (
        0.30 * (weather / 3.0)
        + 0.25 * congestion
        + 0.30 * supplier_risk
        + 0.15 * (traffic / 5.0)
    ))


def tab_predict() -> None:
    st.header("Shipment Delay Predictor")
    st.caption("Enter shipment attributes to predict delay probability and risk level.")

    col1, col2 = st.columns(2)
    with col1:
        weather_sev  = st.slider("Weather Severity (0 = clear, 3 = severe)",  0, 3, 1)
        traffic_lvl  = st.slider("Traffic Level (1 = free, 5 = congested)",   1, 5, 2)
        distance_km  = st.number_input("Distance (km)", min_value=50.0, max_value=20000.0, value=3500.0, step=100.0)
    with col2:
        supplier_risk = st.slider("Supplier Risk (0 = reliable, 1 = high risk)", 0.0, 1.0, 0.25, 0.01)
        port_cong    = st.slider("Port Congestion (0 = clear, 1 = fully blocked)", 0.0, 1.0, 0.40, 0.01)
        mode         = st.selectbox("Transport Mode", ["truck", "ship", "air"])

    risk_score = _compute_risk(weather_sev, port_cong, supplier_risk, traffic_lvl)

    # Logistic approximation for display (model may not be trained yet)
    air_pen  = -1.5 if mode == "air" else (0.3 if mode == "ship" else 0.0)
    logit    = (
        -3.0
        + 2.5 * (weather_sev / 3.0)
        + 2.0 * port_cong
        + 3.0 * supplier_risk
        + 1.5 * (traffic_lvl / 5.0)
        + 0.4 * (distance_km / 8000.0)
        + air_pen
    )
    prob = float(1.0 / (1.0 + np.exp(-logit)))

    # Try loading trained classifier
    model_dir = ROOT / _load_config()["data"]["models_dir"]
    model_used = "Statistical Model (Logistic Approx.)"
    try:
        import xgboost as xgb
        feat_path = model_dir / "xgb_feature_cols.json"
        clf_path  = model_dir / "xgb_classifier.json"
        if clf_path.exists() and feat_path.exists():
            with open(feat_path) as f:
                feat_cols = json.load(f)
            clf = xgb.XGBClassifier()
            clf.load_model(clf_path)

            base = {c: 0.0 for c in feat_cols}
            base.update({
                "weather_severity": weather_sev,
                "traffic_level":    traffic_lvl,
                "supplier_risk":    supplier_risk,
                "port_congestion":  port_cong,
                "distance_km":      distance_km,
                "mode_ship":        int(mode == "ship"),
                "mode_truck":       int(mode == "truck"),
                "risk_score":       risk_score,
                "feat_distance_log": float(np.log1p(distance_km)),
                "cng_roll7_mean":   port_cong,
                "cng_roll14_mean":  port_cong,
                "cng_roll30_mean":  port_cong,
                "cng_lag1": port_cong,
                "cng_lag7": port_cong,
                "cng_lag14": port_cong,
            })
            X = np.array([[base.get(c, 0.0) for c in feat_cols]])
            prob = float(clf.predict_proba(X)[0, 1])
            model_used = "XGBoost Classifier"
    except Exception:
        pass

    # Display
    delayed = prob >= 0.50
    level   = ("critical" if risk_score >= 0.75 else "high" if risk_score >= 0.55
               else "medium" if risk_score >= 0.35 else "low")
    color   = {"critical": "red", "high": "orange", "medium": "yellow", "low": "green"}[level]

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Delay Probability",  f"{prob:.1%}")
    c2.metric("Risk Score",         f"{risk_score:.3f}")
    c3.metric("Risk Level",         level.upper())
    c4.metric("Predicted Status",   "DELAYED" if delayed else "ON TIME")

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": f"Delay Probability ({model_used})"},
        delta={"reference": 30, "increasing": {"color": "red"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "darkblue"},
            "steps": [
                {"range": [0, 30],  "color": "#c8e6c9"},
                {"range": [30, 55], "color": "#fff9c4"},
                {"range": [55, 75], "color": "#ffe0b2"},
                {"range": [75, 100],"color": "#ffcdd2"},
            ],
            "threshold": {"line": {"color": "red", "width": 3}, "value": 50},
        },
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    if delayed:
        st.warning(
            f"⚠️ **Delay Risk [{level.upper()}]**: Estimated delay likely. "
            "Consider alerting stakeholders and evaluating alternative routes or suppliers."
        )
    else:
        st.success("✅ Shipment predicted to arrive **on time** under current conditions.")


# =============================================================================
# App Entry Point
# =============================================================================

def main() -> None:
    try:
        data    = load_data()
        filters = render_sidebar(data)
        shp_flt = apply_filters(data["shipments"], filters)
    except FileNotFoundError:
        st.error(
            "Data files not found. Please generate data first:\n\n"
            "```\npython data/generate_data.py\n```"
        )
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🗺️ Risk Map", "🚨 Anomalies",
        "⚓ Port Congestion", "🔮 Predict",
    ])

    with tab1: tab_overview(shp_flt)
    with tab2: tab_risk_map(shp_flt)
    with tab3: tab_anomalies(shp_flt)
    with tab4: tab_congestion(filters)
    with tab5: tab_predict()


if __name__ == "__main__":
    main()
