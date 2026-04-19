"""
=============================================================
 ML Model Monitoring with Prometheus & Grafana
 File: generate_model.py
 Purpose: Train and save a simple ML model (Iris classifier)
          that the FastAPI app will serve and monitor
=============================================================
"""
import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

os.makedirs("models", exist_ok=True)

print("="*55)
print("  Training Iris Classifier for Monitoring Demo")
print("="*55)

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_sc, y_train)

# Evaluate
y_pred = model.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)

print(f"\n  Model       : Random Forest (100 trees)")
print(f"  Dataset     : Iris (150 samples, 4 features, 3 classes)")
print(f"  Accuracy    : {acc:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=iris.target_names)}")

# Save
joblib.dump(model,  "models/iris_model.pkl")
joblib.dump(scaler, "models/iris_scaler.pkl")

# Save class names for API
import json
with open("models/class_names.json", "w") as f:
    json.dump(iris.target_names.tolist(), f)

print("  Models saved -> models/iris_model.pkl")
print("  Scaler saved -> models/iris_scaler.pkl")
print("\n Model training complete!")

if __name__ == "__main__":
    pass
