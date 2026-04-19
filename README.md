# ML Model Monitoring with Prometheus & Grafana
**Prometheus Metrics · Grafana Dashboards · Drift Alerts**

---

## Project Overview

This practical demonstrates a complete **ML Model Monitoring** pipeline:

- **FastAPI** serves an Iris classifier at port `8000` and exposes Prometheus metrics at `/metrics`
- **Prometheus** scrapes metrics every 10s and stores them in a time-series database (port `9090`)
- **Grafana** visualizes metrics with a pre-built dashboard and fires alerts (port `3000`)
- **Streamlit** provides an instant local monitoring UI without Docker (port `8501`)

### Metrics Tracked
| Metric | Type | Description |
|---|---|---|
| `ml_predictions_total` | Counter | Total prediction requests |
| `ml_inference_duration_seconds` | Histogram | Inference latency distribution |
| `ml_prediction_errors_total` | Counter | Total errors by type |
| `ml_model_confidence` | Gauge | Latest confidence score |
| `ml_model_drift_score` | Gauge | Simulated data drift (0–1) |
| `ml_active_requests` | Gauge | Concurrent requests in progress |
| `ml_class_predictions_total` | Counter | Per-class prediction counts |

---

##  Project Structure

```
practical10_prometheus_grafana/
├── app.py                          # FastAPI ML server + Prometheus metrics
├── generate_model.py               # Train & save Iris classifier
├── streamlit_dashboard.py          # Local monitoring dashboard (no Docker)
├── prometheus.yml                  # Prometheus scrape config
├── alert_rules.yml                 # Alert rules (drift, latency, errors)
├── docker-compose.yml              # Runs Prometheus + Grafana together
├── requirements.txt
├── environment.yml
├── README.md
├── models/                         # Auto-created after training
│   ├── iris_model.pkl
│   ├── iris_scaler.pkl
│   └── class_names.json
├── grafana/
│   └── provisioning/
│       ├── datasources/prometheus.yml
│       └── dashboards/
│           ├── dashboards.yml
│           └── ml_monitoring.json   # Pre-built Grafana dashboard
└── notebooks/
    └── Practical10_Monitoring.ipynb
```

---

##  Setup Instructions

### Step 1: Create Conda Environment
```bash
conda env create -f environment.yml
conda activate model-monitoring
```
**OR manually:**
```bash
conda create -n model-monitoring python=3.10
conda activate model-monitoring
pip install -r requirements.txt
```

### Step 2: Train the ML Model
```bash
python generate_model.py
```

---

##  Execution Steps

### OPTION A — Local Demo (No Docker Required)  Recommend

Open **3 terminal windows**, all with conda env activated:

**Terminal 1 — Start FastAPI Server:**
```bash
python app.py
# OR: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Start Streamlit Dashboard:**
```bash
streamlit run streamlit_dashboard.py
```

**Terminal 3 — Generate Traffic:**
```bash
# Open browser to: http://localhost:8501
# Click "Simulate Traffic" in sidebar
# OR hit the /simulate endpoint:
curl http://localhost:8000/simulate?n=30
```

**Access Points:**
| URL | Description |
|---|---|
| http://localhost:8000 | FastAPI landing page |
| http://localhost:8000/docs | Swagger UI (test predictions) |
| http://localhost:8000/metrics | Raw Prometheus metrics |
| http://localhost:8000/simulate?n=20 | Generate 20 fake predictions |
| http://localhost:8501 | Streamlit monitoring dashboard |

---

### OPTION B — Full Stack with Docker (Prometheus + Grafana)

**Prerequisites:** Docker Desktop installed and running

```bash
# Step 1: Start Prometheus + Grafana
docker-compose up -d

# Step 2: Start FastAPI (in conda terminal)
python app.py

# Step 3: Generate traffic to see metrics
curl http://localhost:8000/simulate?n=50
```

**Access Points:**
| URL | Credentials |
|---|---|
| http://localhost:9090 | Prometheus UI |
| http://localhost:3000 | Grafana (admin / admin123) |
| http://localhost:8000/metrics | Metrics scrape endpoint |

**In Grafana:**
1. Login → admin / admin123
2. Go to Dashboards → Browse → "ML Monitoring"
3. The dashboard loads automatically!

**Stop services:**
```bash
docker-compose down
```

---

## Expected Output

### Terminal (generate_model.py)
```
Training Iris Classifier for Monitoring Demo
Model Accuracy: 0.9667
Models saved -> models/iris_model.pkl
```

### Terminal (app.py)
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
Model loaded: models/iris_model.pkl
INFO:     GET /metrics      [Prometheus scraping]
INFO:     POST /predict     200 OK
```

### /metrics Endpoint (sample)
```
# HELP ml_predictions_total Total number of prediction requests
# TYPE ml_predictions_total counter
ml_predictions_total{predicted_class="setosa",status="success"} 12.0
ml_predictions_total{predicted_class="versicolor",status="success"} 9.0

# HELP ml_inference_duration_seconds Time taken for model inference
# TYPE ml_inference_duration_seconds histogram
ml_inference_duration_seconds_bucket{le="0.001"} 3.0
ml_inference_duration_seconds_bucket{le="0.005"} 18.0
ml_inference_duration_seconds_sum 0.087
ml_inference_duration_seconds_count 21.0

# HELP ml_model_drift_score Simulated model drift score
# TYPE ml_model_drift_score gauge
ml_model_drift_score 0.23456
```

### Grafana Dashboard Panels
| Panel | Shows |
|---|---|
| Total Predictions | Running count (stat) |
| Error Rate % | With red/yellow thresholds |
| Avg Latency | With alert threshold line |
| Drift Score | Gauge with warning zones |
| Latency Over Time | Time-series with P95 |
| Class Distribution | Donut chart |

---

## Alert Rules (alert_rules.yml)

| Alert | Condition | Severity |
|---|---|---|
| HighPredictionErrorRate | Error rate > 5% for 1min | Critical |
| HighInferenceLatency | P95 latency > 500ms for 2min | Warning |
| ModelDriftDetected | Drift score > 0.7 for 30s | Critical |
| LowModelConfidence | Confidence < 60% for 1min | Warning |
| NoPredictions | No requests in 5min | Warning |

---

## PromQL Query Examples

```promql
# Total predictions
sum(ml_predictions_total)

# Request rate per second
rate(ml_predictions_total[1m])

# Error rate percentage
sum(rate(ml_prediction_errors_total[5m])) /
sum(rate(ml_predictions_total[5m])) * 100

# Average inference latency (ms)
rate(ml_inference_duration_seconds_sum[5m]) /
rate(ml_inference_duration_seconds_count[5m]) * 1000

# 95th percentile latency
histogram_quantile(0.95,
  rate(ml_inference_duration_seconds_bucket[5m])
) * 1000

# Model drift score
ml_model_drift_score

# Model confidence
ml_model_confidence
```

---

## Common Errors & Fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError: prometheus_client` | `pip install prometheus-client` |
| `Port 8000 already in use` | `uvicorn app:app --port 8001` |
| `Port 3000 already in use` | Edit docker-compose.yml → `"3001:3000"` |
| Docker can't reach FastAPI on Windows | In prometheus.yml change `localhost:8000` → `host.docker.internal:8000` |
| Grafana shows "No data" | Ensure FastAPI is running and `/metrics` is accessible |
| `model not found` error | Run `python generate_model.py` first |
| Prometheus shows target DOWN | Check FastAPI is running, check firewall settings |

---

## Key Concepts for Students

| Concept | Explanation |
|---|---|
| **Counter** | Only goes up (requests, errors). Reset on restart. |
| **Histogram** | Tracks value distribution in buckets (latency). Enables percentiles. |
| **Gauge** | Current snapshot value (drift score, active connections). |
| **Summary** | Pre-computed quantiles client-side. |
| **PromQL** | Prometheus Query Language for filtering and aggregating metrics. |
| **Scrape** | Prometheus pulls from `/metrics` every N seconds. |
| **Model Drift** | When real-world data distribution shifts from training data. |

---