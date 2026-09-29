from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

EXPECTED_FIELDS = [
    "Age", "TenureMonths", "MonthlyCharges",
    "Contract", "InternetService", "TechSupport", "PaymentMethod"
]

@app.get("/")
def home():
    return jsonify({
        "service": "InternSpark Customer Churn Prediction API",
        "status": "running",
        "endpoint": "/predict",
        "method": "POST"
    })

@app.get("/health")
def health():
    return jsonify({"status": "healthy"})

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Send a JSON request body."}), 400

    missing = [f for f in EXPECTED_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": "Missing fields", "fields": missing}), 400

    try:
        X = pd.DataFrame([{f: payload[f] for f in EXPECTED_FIELDS}])
        probability = float(model.predict_proba(X)[0, 1])
        prediction = int(probability >= 0.50)

        return jsonify({
            "churn_prediction": prediction,
            "label": "High Risk" if prediction == 1 else "Low Risk",
            "churn_probability": round(probability, 4)
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
