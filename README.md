# Reducing Traffic Mortality Using Machine Learning

## Overview
A Predictive Framework for Accident Severity Classification & Risk Hotspot Identification. This machine learning project predicts the severity of traffic accidents based on environmental and temporal factors, identifies geographical hotspots using spatial clustering, and provides an explainable composite "Risk Score" for decision support.

## Features
*   **Unified Data Pipeline:** Ingests and standardizes data from US Accidents, UK STATS19, and MoRTH.
*   **Severity Prediction:** Classifies accident severity (Minor, Serious, Fatal) using ensemble models (Random Forest, XGBoost, LightGBM).
*   **Class Imbalance Handling:** Utilizes SMOTE to ensure high recall for critical (Fatal) minority classes.
*   **Hotspot Detection:** Implements DBSCAN, K-Means, and KDE for spatial-temporal risk mapping.
*   **Explainable AI:** Uses SHAP and LIME for transparent model decision-making.
*   **Risk Scoring:** Generates a 0-100 actionable risk metric combining model predictions, environment, and history.

## Tech Stack
*   **Language:** Python 3.10+
*   **Data Processing:** Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE)
*   **Models:** XGBoost, LightGBM
*   **Explainability:** SHAP, LIME
*   **Visualization/Geospatial:** Matplotlib, Seaborn, Folium

## Project Structure
```
traffic-mortality-ml/
│
├── config/                 # Configuration YAML and Python loader
├── data/                   # Raw and processed datasets (ignored in git)
├── docs/                   # Documentation and Viva Prep
├── notebooks/              # Jupyter notebooks for EDA and prototyping
├── src/                    # Main source code
│   ├── data/               # Data ingestion and preprocessing scripts
│   ├── features/           # Feature engineering
│   ├── models/             # Model training and evaluation
│   ├── risk/               # Risk scoring and hotspot logic
│   └── visualization/      # Plotting and dashboard logic
├── README.md               # This file
├── PROJECT_PLAN.md         # Full project execution plan
└── requirements.txt        # Python dependencies
```

## Setup Instructions (Windows / VS Code)
1. **Clone the repository.**
2. **Create a virtual environment:** `python -m venv venv`
3. **Activate the environment:** `.\venv\Scripts\activate`
4. **Install dependencies:** `pip install -r requirements.txt`
5. **Configure Kaggle API:** Place your `kaggle.json` in `C:\Users\<USER>\.kaggle\` to download the US dataset.

## How to Run
1. Configure settings in `config/risk_config.yaml`.
2. Run data pipeline: `python src/data/make_dataset.py`
3. Train models: `python src/models/train_model.py`
4. View outputs in the `outputs/` and `figures/` directories.

## Dashboards
[PLACEHOLDER: Insert Dashboard Screenshot Here]

## Citations & License
*   US Accidents Dataset: Moosavi, Sobhan, et al. (Kaggle)
*   Code licensed under MIT License.
