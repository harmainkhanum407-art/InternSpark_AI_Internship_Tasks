"""
InternSpark Task 4: Responsible AI & Model Interpretation

Run after Task 1 training:
    python src/responsible_ai.py

Produces SHAP plots, permutation importance, group fairness metrics,
and a short JSON summary.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT/"data/customer_churn.csv")
model = joblib.load(ROOT/"models/churn_model.joblib")
cfg = joblib.load(ROOT/"models/feature_config.joblib")
features = cfg["features"]

X = df[features]
y = df["Churn"]
gender = df["Gender"]

pred = model.predict(X)
proba = model.predict_proba(X)[:,1]

# Global permutation importance on the original feature columns.
perm = permutation_importance(
    model, X, y, n_repeats=10, random_state=42, scoring="roc_auc"
)
importance = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values("importance_mean", ascending=False)
importance.to_csv(ROOT/"reports/permutation_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(7,4))
top = importance.head(8).sort_values("importance_mean")
ax.barh(top["feature"], top["importance_mean"])
ax.set_xlabel("Mean decrease in ROC-AUC")
ax.set_title("Global Permutation Importance")
fig.tight_layout()
fig.savefig(ROOT/"reports/permutation_importance.png", dpi=160)
plt.close(fig)

# SHAP explanation. TreeExplainer is used for the Random Forest when it is
# selected; otherwise Kernel/Linear style explanation is used.
shap_status = "not_run"
try:
    import shap
    # Transform data through the pipeline's preprocessing stage.
    pre = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    Xt = pre.transform(X)
    feature_names = list(pre.get_feature_names_out())

    if hasattr(Xt, "toarray"):
        Xt_dense = Xt.toarray()
    else:
        Xt_dense = Xt

    # Keep SHAP computation lightweight for an internship demo.
    sample = Xt_dense[:120]

    if estimator.__class__.__name__ == "RandomForestClassifier":
        explainer = shap.TreeExplainer(estimator)
        sv = explainer.shap_values(sample)
        values = sv[1] if isinstance(sv, list) else sv
    else:
        explainer = shap.LinearExplainer(estimator, sample)
        sv = explainer.shap_values(sample)
        values = sv[1] if isinstance(sv, list) else sv

    shap.summary_plot(values, sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(ROOT/"reports/shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    shap_status = "completed"
except Exception as exc:
    (ROOT/"reports/shap_note.txt").write_text(
        "SHAP could not be generated in this environment. "
        "Install the versions in requirements.txt and rerun. Error: "
        + repr(exc)
    )

# Group fairness metrics by gender.
fairness_rows = []
for group in sorted(gender.unique()):
    mask = gender == group
    yp = pred[mask]
    yt = y[mask]
    fairness_rows.append({
        "group": group,
        "count": int(mask.sum()),
        "positive_prediction_rate": float(np.mean(yp)),
        "accuracy": float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0))
    })
fairness = pd.DataFrame(fairness_rows)
fairness.to_csv(ROOT/"reports/gender_fairness_metrics.csv", index=False)

summary = {
    "shap_status": shap_status,
    "fairness_groups": fairness_rows,
    "interpretation_note": (
        "Observed group differences are descriptive diagnostics, not proof of "
        "discrimination. Investigate data quality, sample size, feature choices, "
        "thresholds and business context before making deployment decisions."
    ),
    "mitigation_recommendations": [
        "Avoid unnecessary sensitive attributes in model features.",
        "Monitor subgroup precision, recall and positive prediction rate.",
        "Review threshold choices and error costs by group.",
        "Document data provenance, limitations and intended use.",
        "Re-evaluate fairness after every major data or model update."
    ]
}
(ROOT/"reports/task4_responsible_ai.json").write_text(json.dumps(summary, indent=2))

print("Responsible AI analysis complete.")
print(json.dumps(summary, indent=2))
