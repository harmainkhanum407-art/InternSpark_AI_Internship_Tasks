"""
InternSpark Task 1: ML Classification Project
Customer Churn Classification

Run:
    python src/train_model.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "customer_churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

# Gender is retained for Task 4 fairness analysis but is intentionally
# excluded from the predictive feature set for the main model.
target = "Churn"
sensitive = "Gender"
features = [
    "Age", "TenureMonths", "MonthlyCharges",
    "Contract", "InternetService", "TechSupport", "PaymentMethod"
]

X = df[features]
y = df[target]

numeric = ["Age", "TenureMonths", "MonthlyCharges"]
categorical = ["Contract", "InternetService", "TechSupport", "PaymentMethod"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced", min_samples_leaf=2
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

results = []
best_name, best_pipe, best_auc = None, None, -1

for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba)
    }
    results.append(metrics)

    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        n_jobs=-1
    )
    metrics["cv_accuracy_mean"] = float(np.mean(cv_scores["test_accuracy"]))
    metrics["cv_f1_mean"] = float(np.mean(cv_scores["test_f1"]))
    metrics["cv_roc_auc_mean"] = float(np.mean(cv_scores["test_roc_auc"]))

    if metrics["roc_auc"] > best_auc:
        best_auc = metrics["roc_auc"]
        best_name, best_pipe = name, pipe

results_df = pd.DataFrame(results)
results_df.to_csv(REPORT_DIR/"model_comparison.csv", index=False)

best_pipe.fit(X_train, y_train)
best_pred = best_pipe.predict(X_test)
best_proba = best_pipe.predict_proba(X_test)[:, 1]

joblib.dump(best_pipe, MODEL_DIR/"churn_model.joblib")
joblib.dump(
    {"features": features, "numeric": numeric, "categorical": categorical},
    MODEL_DIR/"feature_config.joblib"
)

summary = {
    "best_model": best_name,
    "test_metrics": {
        "accuracy": float(accuracy_score(y_test, best_pred)),
        "precision": float(precision_score(y_test, best_pred, zero_division=0)),
        "recall": float(recall_score(y_test, best_pred, zero_division=0)),
        "f1": float(f1_score(y_test, best_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, best_proba))
    },
    "test_size": int(len(y_test)),
    "random_state": 42
}
(REPORT_DIR/"task1_results.json").write_text(json.dumps(summary, indent=2))

# Confusion matrix
cm = confusion_matrix(y_test, best_pred)
fig, ax = plt.subplots(figsize=(5,4))
ax.imshow(cm)
ax.set_title(f"Confusion Matrix - {best_name}")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0,1], ["No Churn","Churn"])
ax.set_yticks([0,1], ["No Churn","Churn"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i,j], ha="center", va="center")
fig.tight_layout()
fig.savefig(REPORT_DIR/"confusion_matrix.png", dpi=160)
plt.close(fig)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, best_proba)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(fpr, tpr, label=f"AUC={summary['test_metrics']['roc_auc']:.3f}")
ax.plot([0,1], [0,1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
fig.tight_layout()
fig.savefig(REPORT_DIR/"roc_curve.png", dpi=160)
plt.close(fig)

print("Training complete.")
print(json.dumps(summary, indent=2))
