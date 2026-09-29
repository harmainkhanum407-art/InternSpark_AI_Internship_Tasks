# InternSpark AI Internship — 3 Task Submission

This repository completes **3 of the 4 required main tasks** using one connected Customer Churn Prediction project:

1. **Task 1 — ML Classification Project**
2. **Task 3 — Model Deployment: API & Containerization**
3. **Task 4 — Responsible AI & Model Interpretation**

Task 2 (Deep Learning — Image/Text) is intentionally not included because the internship page requires any 3 of the 4 main tasks.

## Project idea

**Customer Churn Classification and Responsible Deployment**

The project uses a small reproducible customer dataset. The classification task compares Logistic Regression and Random Forest, reports Accuracy, Precision, Recall, F1 and ROC-AUC, and uses 5-fold stratified cross-validation.

The deployment task exposes the selected model through a Flask REST API and provides a Dockerfile.

The Responsible AI task uses global feature importance, SHAP where supported, and subgroup diagnostics by Gender. Gender is not used as a predictive feature in the main model, but it is retained for fairness analysis.

## Folder structure

- `data/customer_churn.csv` — small reproducible dataset
- `src/train_model.py` — Task 1 training/evaluation
- `src/responsible_ai.py` — Task 4 interpretation/fairness analysis
- `models/` — generated trained model files
- `notebooks/` — Jupyter notebooks
- `api/app.py` — Task 3 Flask API
- `Dockerfile` — Task 3 container definition
- `reports/` — metrics, plots and JSON summaries
- `task_1_ml_classification/` — Task 1 submission document
- `task_3_model_deployment/` — Task 3 submission document
- `task_4_responsible_ai/` — Task 4 submission document

## Setup in VS Code / PowerShell

### 1. Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train Task 1

```powershell
python src/train_model.py
```

This creates:
- `models/churn_model.joblib`
- `models/feature_config.joblib`
- `reports/model_comparison.csv`
- `reports/task1_results.json`
- `reports/confusion_matrix.png`
- `reports/roc_curve.png`

### 4. Run Task 4

```powershell
python src/responsible_ai.py
```

This creates:
- `reports/permutation_importance.csv`
- `reports/permutation_importance.png`
- `reports/gender_fairness_metrics.csv`
- `reports/task4_responsible_ai.json`
- `reports/shap_summary.png` when SHAP runs successfully

### 5. Run Task 3 API locally

First make sure Task 1 has been run.

```powershell
python api/app.py
```

Open:

```text
http://127.0.0.1:5000/
```

Health check:

```text
http://127.0.0.1:5000/health
```

### 6. Test the prediction API with PowerShell

```powershell
$body = Get-Content .\api\sample_request.json -Raw
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body $body
```

Expected response shape:

```json
{
  "churn_prediction": 0,
  "label": "Low Risk",
  "churn_probability": 0.1234
}
```

The exact probability depends on the trained model and should not be copied into the report unless it is produced by your own run.

## Docker

Build:

```powershell
docker build -t internspark-churn-api .
```

Run:

```powershell
docker run --rm -p 5000:5000 internspark-churn-api
```

Then test:

```text
http://127.0.0.1:5000/health
```

## Jupyter

Install/use Jupyter:

```powershell
pip install jupyter
jupyter notebook
```

Open:
- `notebooks/task1_customer_churn.ipynb`
- `notebooks/task4_responsible_ai.ipynb`

## What to submit

The internship instructions shown in the screenshots require a **separate DOC/DOCX/PDF document for each task**, containing code links, notebooks, screenshots and explanation. They also require a README with environment setup and exact commands.

Submit these three documents separately:
- `task_1_ml_classification/Task_1_ML_Classification_Report.docx`
- `task_3_model_deployment/Task_3_Model_Deployment_Report.docx`
- `task_4_responsible_ai/Task_4_Responsible_AI_Report.docx`

Also share the GitHub repository link containing the code and notebooks.

For large model/dataset files, the internship page says Google Drive links can be used if needed. This project dataset is intentionally small.

## Important

Do not upload `.venv/`, `__pycache__/`, or other local environment folders to GitHub. A `.gitignore` is included.
