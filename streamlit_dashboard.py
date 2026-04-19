"""
=============================================================
 ML Model Monitoring
 File: streamlit_dashboard.py
 Purpose: Real-time monitoring dashboard using Streamlit
          (alternative to Grafana for local demo without Docker)
=============================================================
"""
import time
import requests
import random
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from collections import deque

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="ML Monitoring Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

# ── Styles ────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;600;700&display=swap');
  html,body,[class*="css"]{font-family:'Inter',sans-serif;}
  .stApp{background:#0a0d14;}
  h1,h2,h3{font-family:'JetBrains Mono',monospace!important;color:#60a5fa!important;}

  .metric-card{
    background:linear-gradient(135deg,#111827,#1a1d27);
    border:1px solid #1e2a3a;border-radius:12px;
    padding:20px 24px;margin:6px 0;
  }
  .metric-label{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:1.5px;}
  .metric-value{font-size:32px;font-weight:700;font-family:'JetBrains Mono',monospace;margin-top:4px;}
  .green{color:#34d399;} .red{color:#f87171;} .blue{color:#60a5fa;} .yellow{color:#fbbf24;}

  .status-ok  {color:#34d399;font-weight:600;}
  .status-err {color:#f87171;font-weight:600;}
  .alert-box{background:#7f1d1d22;border:1px solid #f87171;border-radius:8px;padding:12px 16px;margin:8px 0;}
  .info-box {background:#1e3a5f22;border:1px solid #60a5fa;border-radius:8px;padding:12px 16px;margin:8px 0;}

  div[data-testid="stSidebar"]{background:#070a10;border-right:1px solid #1e2332;}
  .stButton>button{background:#6366f1;color:white;border:none;border-radius:8px;padding:8px 24px;font-weight:600;}
  .stButton>button:hover{background:#818cf8;}
</style>
""", unsafe_allow_html=True)

# ── Session State for Live Data ───────────────────────────────
if "latency_history"   not in st.session_state:
    st.session_state.latency_history   = deque(maxlen=60)
if "request_history"   not in st.session_state:
    st.session_state.request_history   = deque(maxlen=60)
if "drift_history"     not in st.session_state:
    st.session_state.drift_history     = deque(maxlen=60)
if "timestamps"        not in st.session_state:
    st.session_state.timestamps        = deque(maxlen=60)
if "class_counts"      not in st.session_state:
    st.session_state.class_counts      = {"setosa": 0, "versicolor": 0, "virginica": 0}
if "error_count"       not in st.session_state:
    st.session_state.error_count       = 0
if "total_requests"    not in st.session_state:
    st.session_state.total_requests    = 0


# ── Helper Functions ──────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200, r.json()
    except:
        return False, {}

def get_stats():
    try:
        r = requests.get(f"{API_URL}/stats", timeout=3)
        return r.json()
    except:
        return None

def make_prediction(sl, sw, pl, pw):
    try:
        r = requests.post(f"{API_URL}/predict",
                          json={"sepal_length":sl,"sepal_width":sw,
                                "petal_length":pl,"petal_width":pw},
                          timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def simulate_batch(n=10):
    try:
        r = requests.get(f"{API_URL}/simulate?n={n}", timeout=30)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def parse_prometheus_metrics(text):
    """Parse Prometheus text format into a dict."""
    metrics = {}
    for line in text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(" ")
        if len(parts) >= 2:
            try:
                metrics[parts[0]] = float(parts[1])
            except:
                pass
    return metrics

def get_prometheus_metrics():
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=3)
        return parse_prometheus_metrics(r.text), r.text
    except:
        return {}, ""


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### **ML Monitoring Dashboard**")
    st.markdown("---")

    # API Status
    api_ok, health_data = check_api()
    if api_ok:
        st.markdown('<p class="status-ok">● API Online</p>', unsafe_allow_html=True)
        st.caption(f"Total requests: {health_data.get('total_requests', 0)}")
    else:
        st.markdown('<p class="status-err">● API Offline — Start app.py</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Controls
    st.markdown("### Controls")
    auto_refresh = st.toggle("Auto Refresh (5s)", value=False)
    n_simulate   = st.slider("Simulate N predictions", 5, 100, 20)

    if st.button(" Simulate Traffic"):
        with st.spinner(f"Simulating {n_simulate} predictions..."):
            result = simulate_batch(n_simulate)
            if "error" not in result:
                st.success(f" {result['message']}")
            else:
                st.error(f" {result['error']}")

    st.markdown("---")
    st.markdown("###  Quick Links")
    st.markdown(f"[ Swagger UI]({API_URL}/docs)")
    st.markdown(f"[ Raw Metrics]({API_URL}/metrics)")
    st.markdown(f" Health]({API_URL}/health)")
    st.markdown("---")
    st.caption("model-monitoring")


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(135deg,#111827 0%,#0a0d14 100%);
            border-left:4px solid #6366f1;border-radius:0 12px 12px 0;
            padding:20px 28px;margin-bottom:28px;">
  <h2 style="margin:0;font-family:'JetBrains Mono',monospace;color:#60a5fa;">
    ML Model Monitoring — Real-Time Dashboard
  </h2>
  <p style="margin:4px 0 0;color:#6b7280;">
    Prometheus Metrics · Iris Classifier · 
  </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    " Live Metrics", " Prometheus Raw", " Live Prediction", " Architecture"
])


# ══════════════════════════════════════════════════════════════
# TAB 1: LIVE METRICS
# ══════════════════════════════════════════════════════════════
with tab1:
    # Fetch current stats
    stats = get_stats()
    prom_metrics, prom_raw = get_prometheus_metrics()

    if stats is None:
        st.error(" Cannot connect to FastAPI server. Make sure `python app.py` is running!")
        st.code("python app.py", language="bash")
        st.stop()

    # Update rolling history
    now = datetime.now().strftime("%H:%M:%S")
    st.session_state.timestamps.append(now)
    st.session_state.latency_history.append(stats.get("avg_latency_ms", 0))
    st.session_state.total_requests = stats.get("total_requests", 0)

    # Simulate drift for visualization
    last_drift = list(st.session_state.drift_history)
    new_drift = max(0, min(1, (last_drift[-1] if last_drift else 0.1)
                           + random.uniform(-0.02, 0.04)))
    st.session_state.drift_history.append(new_drift)

    # ── KPI Row ───────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    total_req  = stats.get("total_requests", 0)
    total_err  = stats.get("total_errors", 0)
    error_rate = stats.get("error_rate", 0) * 100
    avg_lat    = stats.get("avg_latency_ms", 0)

    with c1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Total Requests</div>
          <div class="metric-value blue">{total_req:,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        cls = "green" if error_rate < 5 else "red"
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Error Rate</div>
          <div class="metric-value {cls}">{error_rate:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        lat_cls = "green" if avg_lat < 50 else "yellow" if avg_lat < 100 else "red"
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Avg Latency</div>
          <div class="metric-value {lat_cls}">{avg_lat:.1f}ms</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        drift = new_drift
        d_cls = "green" if drift < 0.3 else "yellow" if drift < 0.6 else "red"
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Drift Score</div>
          <div class="metric-value {d_cls}">{drift:.3f}</div>
        </div>""", unsafe_allow_html=True)

    # ── Alerts ────────────────────────────────────────────────
    st.markdown("###  Alerts")
    alert_triggered = False
    if error_rate > 5:
        st.markdown(f'<div class="alert-box">🔴 <b>HIGH ERROR RATE:</b> {error_rate:.1f}% exceeds threshold (5%)</div>', unsafe_allow_html=True)
        alert_triggered = True
    if avg_lat > 100:
        st.markdown(f'<div class="alert-box">🟡 <b>HIGH LATENCY:</b> {avg_lat:.1f}ms exceeds threshold (100ms)</div>', unsafe_allow_html=True)
        alert_triggered = True
    if new_drift > 0.6:
        st.markdown(f'<div class="alert-box">🔴 <b>MODEL DRIFT DETECTED:</b> Score {new_drift:.3f} exceeds threshold (0.6)</div>', unsafe_allow_html=True)
        alert_triggered = True
    if not alert_triggered:
        st.markdown('<div class="info-box"> <b>All systems normal.</b> No alerts triggered.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    timestamps = list(st.session_state.timestamps)

    with col1:
        latencies = list(st.session_state.latency_history)
        if latencies:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps, y=latencies,
                mode='lines+markers', name='Avg Latency (ms)',
                line=dict(color='#60a5fa', width=2),
                fill='tozeroy', fillcolor='rgba(96,165,250,0.1)'
            ))
            fig.add_hline(y=100, line_dash="dash", line_color="#f87171",
                          annotation_text="Alert threshold (100ms)")
            fig.update_layout(
                title=' Inference Latency Over Time',
                xaxis_title='Time', yaxis_title='Latency (ms)',
                template='plotly_dark', paper_bgcolor='#111827',
                plot_bgcolor='#111827', height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        drifts = list(st.session_state.drift_history)
        if drifts:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=timestamps[-len(drifts):], y=drifts,
                mode='lines+markers', name='Drift Score',
                line=dict(color='#fbbf24', width=2),
                fill='tozeroy', fillcolor='rgba(251,191,36,0.1)'
            ))
            fig2.add_hline(y=0.3, line_dash="dash", line_color="#fbbf24",
                           annotation_text="Warning (0.3)")
            fig2.add_hline(y=0.6, line_dash="dash", line_color="#f87171",
                           annotation_text="Critical (0.6)")
            fig2.update_layout(
                title=' Model Drift Score Over Time',
                xaxis_title='Time', yaxis_title='Drift Score',
                template='plotly_dark', paper_bgcolor='#111827',
                plot_bgcolor='#111827', height=300, yaxis_range=[0, 1]
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Recent predictions table
    st.markdown("### Last 10 Predictions")
    last_preds = stats.get("last_10_predictions", [])
    if last_preds:
        df = pd.DataFrame(last_preds)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No predictions yet. Click 'Simulate Traffic' in the sidebar!")

    if auto_refresh:
        time.sleep(5)
        st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 2: PROMETHEUS RAW
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Raw Prometheus Metrics (/metrics endpoint)")
    st.markdown("This is what Prometheus scrapes from `/metrics` every 15 seconds.")

    _, raw_text = get_prometheus_metrics()

    if raw_text:
        # Parse and display nicely
        st.markdown("#### Key ML Metrics")
        prom_dict, _ = get_prometheus_metrics()

        ml_metrics = {k: v for k, v in prom_dict.items()
                      if k.startswith("ml_") and not k.endswith(("_total","_count","_sum"))}

        if ml_metrics:
            cols = st.columns(3)
            for i, (k, v) in enumerate(ml_metrics.items()):
                with cols[i % 3]:
                    st.metric(k.replace("ml_","").replace("_"," ").title(), round(v, 4))

        st.markdown("#### Full Raw Output")
        st.code(raw_text, language="text")
    else:
        st.error("Cannot fetch metrics. Ensure API is running.")

    st.markdown("---")
    st.markdown("""
    #### How Prometheus Works
    1. Your FastAPI app exposes metrics at `/metrics`
    2. Prometheus server **scrapes** this URL every 15 seconds
    3. Metrics are stored in Prometheus time-series database
    4. Grafana **queries** Prometheus using PromQL
    5. You set **alerts** when metric values cross thresholds
    """)


# ══════════════════════════════════════════════════════════════
# TAB 3: LIVE PREDICTION
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader(" Live Prediction — Iris Classifier")
    st.markdown("Each prediction is tracked in Prometheus metrics.")

    col1, col2 = st.columns(2)
    with col1:
        sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
        sw = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.5, 0.1)
    with col2:
        pl = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
        pw = st.slider("Petal Width (cm)",  0.1, 2.5, 0.2, 0.1)

    if st.button("Predict & Track Metrics"):
        with st.spinner("Predicting..."):
            result = make_prediction(sl, sw, pl, pw)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            c1, c2, c3 = st.columns(3)
            flower_emoji = {"setosa": "🌸", "versicolor": "🌺", "virginica": "🌼"}
            emoji = flower_emoji.get(result["prediction"], "🌿")

            with c1:
                st.metric("Prediction", f"{emoji} {result['prediction'].upper()}")
            with c2:
                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            with c3:
                st.metric("Inference Time", f"{result['inference_time_ms']:.2f} ms")

            st.markdown("#### Probability Breakdown")
            probs = result["probabilities"]
            fig = px.bar(
                x=list(probs.keys()), y=list(probs.values()),
                color=list(probs.keys()),
                color_discrete_map={"setosa":"#f87171","versicolor":"#60a5fa","virginica":"#34d399"},
                template="plotly_dark", title="Prediction Probabilities"
            )
            fig.update_layout(paper_bgcolor='#111827', plot_bgcolor='#111827',
                               showlegend=False, yaxis_tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
            st.info(f" This prediction was logged to Prometheus at `/metrics`")


# ══════════════════════════════════════════════════════════════
# TAB 4: ARCHITECTURE
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader(" System Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────┐
    │                    CLIENT / USER                        │
    │         (Postman / Browser / Streamlit UI)              │
    └─────────────────┬───────────────────────────────────────┘
                      │ HTTP POST /predict
                      ▼
    ┌─────────────────────────────────────────────────────────┐
    │              FastAPI Application (app.py)               │
    │  ┌─────────────────┐  ┌──────────────────────────────┐  │
    │  │  ML Model       │  │  Prometheus Metrics          │  │
    │  │  (Random Forest │  │  - ml_predictions_total      │  │
    │  │   Iris Classifier│  │  - ml_inference_duration     │  │
    │  │  Port: 8000)    │  │  - ml_model_confidence       │  │
    │  └─────────────────┘  │  - ml_model_drift_score      │  │
    │                       │  - ml_prediction_errors      │  │
    │                       └──────────────────────────────┘  │
    │                              │ /metrics endpoint        │
    └──────────────────────────────┼──────────────────────────┘
                                   │ Scrape every 15s
                                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │           Prometheus Server (Port: 9090)                │
    │         Time-series database of all metrics            │
    │         prometheus.yml → scrape config                  │
    └──────────────────────────┬──────────────────────────────┘
                               │ PromQL queries
                               ▼
    ┌─────────────────────────────────────────────────────────┐
    │            Grafana Dashboard (Port: 3000)               │
    │         Visualize metrics + Set alerts                  │
    │         Pre-built ML Monitoring Dashboard               │
    └─────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("###  Prometheus Metric Types Used")
    metric_types = {
        "Counter": "ml_predictions_total, ml_prediction_errors_total — only goes up",
        "Histogram": "ml_inference_duration_seconds — tracks distribution & percentiles",
        "Gauge": "ml_model_confidence, ml_model_drift_score — can go up or down",
        "Summary": "ml_request_processing_seconds — pre-computed quantiles"
    }
    for k, v in metric_types.items():
        st.markdown(f"**{k}**: {v}")
