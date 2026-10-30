# MaintenIQ

**Predictive Maintenance Intelligence | Streamlit + Random Forest**

MaintenIQ helps explore machine-failure patterns and assess operating conditions. It combines a trained Random Forest classifier with an interactive Streamlit dashboard to display failure-risk scores and maintenance decision-support information.

> **Demo:** [MaintenIQ](localhost)

## Features

The application contains four pages:

| Page | What it does |
| --- | --- |
| **Operations Overview** | Displays historical observations, recorded failures, failure rate, processed readings and summary charts. |
| **Live Monitoring** | Replays held-out AI4I observations through the prediction pipeline. Supports **Start/Stop/Resume Monitoring**, **Process 25 Readings**, and **Reset Feed**. Monitoring updates at approximately three-second intervals while running. |
| **Risk Assessment** | Accepts six machine operating inputs and displays a predicted failure-risk score, risk priority and maintenance guidance. |
| **Model Intelligence** | Displays model evaluation metrics, the test confusion matrix and feature-importance information. |

**Note:** “Live Monitoring” is a controlled replay of dataset observations, **not** a real-time IoT connection.

## Dataset and inputs

The project uses the [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). It contains **10,000** observations, including **339** records labelled as machine failures. The dataset is synthetically generated to reflect industrial predictive-maintenance conditions.

The prediction form uses these six inputs:

- Product type (`Type`)
- Air temperature (`Air temperature [K]`)
- Process temperature (`Process temperature [K]`)
- Rotational speed (`Rotational speed [rpm]`)
- Torque (`Torque [Nm]`)
- Tool wear (`Tool wear [min]`)

The target is `Machine failure`. The trained pipeline also uses engineered operating-condition features. Failure-mode labels are not used as model inputs.

## Technology

- Python
- Streamlit
- pandas
- scikit-learn (Random Forest)
- joblib

The Streamlit application **loads a previously trained model**; it does not retrain the model whenever the dashboard starts.

## Project structure

```text
MaintenIQ/
├── app.py
├── pdm/
│   ├── __init__.py
│   └── data.py              # Dataset loading and feature definitions
├── data/
│   └── ai4i2020.csv
├── artifacts/
│   ├── random_forest.joblib
│   ├── metrics.json
│   └── feature_importance.csv
├── requirements.txt
├── .gitignore
└── README.md
```

This layout shows the **runtime files required by `app.py`**. Keep any additional training scripts, tests or configuration files already in your project. The `pdm` package must be included because `app.py` imports `pdm.data`.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/dshrestha23/machine-maintenance-abi
cd machine-maintenance-abi
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -e
```

The `pyproject.toml` contain the packages required by the project, including `streamlit`, `pandas`, `scikit-learn`, and `joblib`. Use compatible versions of scikit-learn for the saved model; ideally match the training environment. Use `python 3.13`.

### 4. Check the required files

Ensure these files exist:

```text
data/ai4i2020.csv
artifacts/random_forest.joblib
artifacts/metrics.json
artifacts/feature_importance.csv
pdm/data.py
```

Download the AI4I CSV from the UCI dataset page if it is not already in your project. Only load `.joblib` files you created or trust.

### 5. Start the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

## How to use MaintenIQ

1. Open **Operations Overview** to inspect the dataset summary and historical outcomes.
2. Open **Live Monitoring** and select **Start Monitoring** for automatic replay, or **Process 25 Readings** to process a batch. Use **Stop Monitoring**, **Resume Monitoring**, or **Reset Feed** as needed.
3. Open **Risk Assessment**, enter the six operating conditions, and click **Assess Conditions**.
4. Open **Model Intelligence** to review the model's test performance and feature importance.

### Risk priority levels

| Priority | Model score |
| --- | --- |
| **Routine** | Below 20% |
| **Review** | 20% to below 50% |
| **High** | 50% or above |

These are application-defined decision-support thresholds, **not** independently validated industrial safety limits. Any additional failure-mode checks are rule-based guidance rather than separate trained failure-mode classifiers.

## Model evaluation

The project uses a stratified **70% training / 15% validation / 15% test** split. The following figures are from the held-out test set:

| Metric | Result |
| --- | ---: |
| Accuracy | 99.40% |
| Precision | 100.00% |
| Recall | 82.35% |
| F1-score | 90.32% |
| ROC-AUC | 0.9926 |
| PR-AUC | 0.9301 |

Test confusion matrix: **1,449 true negatives, 0 false positives, 9 false negatives and 42 true positives**.
