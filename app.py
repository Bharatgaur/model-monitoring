"""
=============================================================
 ML Model Monitoring with Prometheus & Grafana
 File: app.py
 Purpose: FastAPI server that:
   1. Serves ML model predictions
   2. Exposes Prometheus metrics at /metrics
   3. Tracks: inference time, request count, errors,
              prediction distribution, model drift score
=============================================================
"""

import time
import random
import json
import os
import joblib
import numpy as np
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional

# Prometheus client
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry,
    REGISTRY
)
from starlette.responses import Response

# ── App Setup ─────────────────────────────────────────────────
app = FastAPI(
    title="ML Model Monitoring API",
    description="Iris Classifier with Prometheus Metrics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Load Model ────────────────────────────────────────────────
MODEL_PATH  = "models/iris_model.pkl"
SCALER_PATH = "models/iris_scaler.pkl"
NAMES_PATH  = "models/class_names.json"

model, scaler, class_names = None, None, []

def load_model():
    global model, scaler, class_names
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found! Run: python generate_model.py first"
        )
    model       = joblib.load(MODEL_PATH)
    scaler      = joblib.load(SCALER_PATH)
    class_names = json.load(open(NAMES_PATH))
    print(f"Model loaded: {MODEL_PATH}")

load_model()

# ── Prometheus Metrics ────────────────────────────────────────
# 1. Total prediction requests (by status: success / error)
PREDICTION_COUNTER = Counter(
    "ml_predictions_total",
    "Total number of prediction requests",
    ["status", "predicted_class"]
)

# 2. Inference latency histogram (measures response time distribution)
INFERENCE_LATENCY = Histogram(
    "ml_inference_duration_seconds",
    "Time taken for model inference",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# 3. Error counter
ERROR_COUNTER = Counter(
    "ml_prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"]
)

# 4. Model confidence score (gauge = current value)
MODEL_CONFIDENCE = Gauge(
    "ml_model_confidence",
    "Latest prediction confidence score"
)

# 5. Drift score (simulated) — would be data distribution shift in production
DRIFT_SCORE = Gauge(
    "ml_model_drift_score",
    "Simulated model drift score (0=no drift, 1=full drift)"
)

# 6. Active requests gauge
ACTIVE_REQUESTS = Gauge(
    "ml_active_requests",
    "Number of requests currently being processed"
)

# 7. Request rate summary
REQUEST_SUMMARY = Summary(
    "ml_request_processing_seconds",
    "Summary of request processing time"
)

# 8. Per-class prediction distribution
CLASS_DISTRIBUTION = Counter(
    "ml_class_predictions_total",
    "Predictions per class",
    ["class_name"]
)

# Track stats in memory for dashboard
stats = {
    "total_requests": 0,
    "total_errors": 0,
    "avg_latency_ms": 0.0,
    "last_predictions": [],
    "start_time": datetime.now().isoformat()
}

# ── Pydantic Schemas ──────────────────────────────────────────
class PredictRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1, description="Sepal length in cm")
    sepal_width:  float = Field(..., example=3.5, description="Sepal width in cm")
    petal_length: float = Field(..., example=1.4, description="Petal length in cm")
    petal_width:  float = Field(..., example=0.2, description="Petal width in cm")

class PredictResponse(BaseModel):
    prediction:       str
    predicted_class:  int
    confidence:       float
    probabilities:    dict
    inference_time_ms: float
    timestamp:        str

class BatchPredictRequest(BaseModel):
    samples: List[PredictRequest]


# ── Routes ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple HTML landing page."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>ML Monitoring - Practical 10</title>
      <style>
        body{font-family:'Segoe UI',sans-serif;background:#0f1117;color:#e0e0e0;margin:0;padding:40px;}
        h1{color:#818cf8;} h2{color:#60a5fa;}
        .card{background:#1a1d27;border:1px solid #2d3142;border-radius:10px;padding:20px;margin:16px 0;}
        a{color:#818cf8;text-decoration:none;} a:hover{text-decoration:underline;}
        .tag{background:#6366f120;color:#818cf8;border:1px solid #6366f150;padding:2px 10px;border-radius:20px;font-size:13px;margin:3px;}
        code{background:#12151e;padding:2px 8px;border-radius:4px;color:#4ade80;}
      </style>
    </head>
    <body>
      <h1>AI Model Monitoring API</h1>
      <p>RTAI-242P · Practical 10 · Prometheus + Grafana</p>

      <div class="card">
        <h2> Endpoints</h2>
        <p><a href="/docs"> /docs</a> — Swagger UI (interactive API)</p>
        <p><a href="/metrics"> /metrics</a> — Prometheus metrics scrape endpoint</p>
        <p><a href="/health"> /health</a> — Health check</p>
        <p><a href="/stats"> /stats</a> — Live request statistics</p>
        <p><a href="/simulate"> /simulate</a> — Auto-generate predictions for demo</p>
      </div>

      <div class="card">
        <h2> Tracked Metrics</h2>
        <span class="tag">ml_predictions_total</span>
        <span class="tag">ml_inference_duration_seconds</span>
        <span class="tag">ml_prediction_errors_total</span>
        <span class="tag">ml_model_confidence</span>
        <span class="tag">ml_model_drift_score</span>
        <span class="tag">ml_active_requests</span>
        <span class="tag">ml_class_predictions_total</span>
      </div>

      <div class="card">
        <h2> Quick Test</h2>
        <p>POST to <code>/predict</code> with:</p>
        <pre style="background:#12151e;padding:16px;border-radius:8px;">
{
  "sepal_length": 5.1,
  "sepal_width":  3.5,
  "petal_length": 1.4,
  "petal_width":  0.2
}</pre>
      </div>
    </body>
    </html>
    """
    return html


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "uptime_since": stats["start_time"],
        "total_requests": stats["total_requests"]
    }


@app.get("/metrics")
async def metrics():
    """
    Prometheus scrape endpoint.
    Prometheus server calls this URL every 15 seconds to collect metrics.
    """
    # Update drift score with a slow random walk (simulates real drift detection)
    current_drift = DRIFT_SCORE._value.get()
    new_drift = max(0.0, min(1.0, current_drift + random.uniform(-0.02, 0.03)))
    DRIFT_SCORE.set(new_drift)

    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Main prediction endpoint.
    All metrics are recorded here:
    - Latency via INFERENCE_LATENCY histogram
    - Count via PREDICTION_COUNTER
    - Confidence via MODEL_CONFIDENCE gauge
    """
    ACTIVE_REQUESTS.inc()
    stats["total_requests"] += 1

    start_time = time.time()

    try:
        # Prepare features
        features = np.array([[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Simulate occasional extra latency (realistic variance)
        if random.random() < 0.1:
            time.sleep(random.uniform(0.05, 0.15))

        # Inference
        prediction    = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence    = float(probabilities.max())
        class_name    = class_names[prediction]

        # Calculate latency
        latency_s  = time.time() - start_time
        latency_ms = latency_s * 1000

        # ── Record Prometheus Metrics ──────────────────────
        INFERENCE_LATENCY.observe(latency_s)
        PREDICTION_COUNTER.labels(
            status="success",
            predicted_class=class_name
        ).inc()
        CLASS_DISTRIBUTION.labels(class_name=class_name).inc()
        MODEL_CONFIDENCE.set(confidence)
        REQUEST_SUMMARY.observe(latency_s)
        # ───────────────────────────────────────────────────

        # Update local stats
        stats["avg_latency_ms"] = (
            stats["avg_latency_ms"] * 0.9 + latency_ms * 0.1
        )

        result = {
            "prediction": class_name,
            "predicted_class": int(prediction),
            "confidence": round(confidence, 4),
            "probabilities": {
                class_names[i]: round(float(p), 4)
                for i, p in enumerate(probabilities)
            },
            "inference_time_ms": round(latency_ms, 3),
            "timestamp": datetime.now().isoformat()
        }

        # Keep last 10 predictions for /stats
        stats["last_predictions"].append({
            "class": class_name,
            "confidence": round(confidence, 3),
            "latency_ms": round(latency_ms, 3)
        })
        stats["last_predictions"] = stats["last_predictions"][-10:]

        return result

    except Exception as e:
        ERROR_COUNTER.labels(error_type=type(e).__name__).inc()
        PREDICTION_COUNTER.labels(status="error", predicted_class="none").inc()
        stats["total_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictRequest):
    """Batch prediction endpoint — predicts for multiple samples."""
    results = []
    for sample in request.samples:
        result = await predict(sample)
        results.append(result)
    return {"predictions": results, "count": len(results)}


@app.get("/stats")
async def get_stats():
    """Live statistics dashboard data."""
    return {
        "total_requests": stats["total_requests"],
        "total_errors": stats["total_errors"],
        "error_rate": (
            stats["total_errors"] / max(stats["total_requests"], 1)
        ),
        "avg_latency_ms": round(stats["avg_latency_ms"], 3),
        "uptime_since": stats["start_time"],
        "last_10_predictions": stats["last_predictions"]
    }


@app.get("/simulate")
async def simulate_traffic(n: int = 20):
    """
    Simulate n prediction requests with random Iris data.
    Useful for generating metrics to see in Grafana.
    Call this endpoint to populate charts!
    """
    iris_samples = [
        # setosa
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2},
        # versicolor
        {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5},
        {"sepal_length": 5.7, "sepal_width": 2.8, "petal_length": 4.1, "petal_width": 1.3},
        # virginica
        {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},
        {"sepal_length": 7.2, "sepal_width": 3.6, "petal_length": 6.1, "petal_width": 2.5},
    ]

    results = []
    for i in range(n):
        sample_data = random.choice(iris_samples).copy()
        # Add small noise
        for k in sample_data:
            sample_data[k] += random.uniform(-0.3, 0.3)
        req = PredictRequest(**sample_data)
        res = await predict(req)
        results.append(res)
        time.sleep(0.05)  # small delay to spread metrics

    return {
        "message": f"Simulated {n} predictions successfully!",
        "summary": {
            "total": n,
            "classes": {
                cn: sum(1 for r in results if r["prediction"] == cn)
                for cn in class_names
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
