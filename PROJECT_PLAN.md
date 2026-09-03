# Project Plan: Traffic Mortality ML

## 1. Objective
Develop a robust machine learning framework to predict traffic accident severity (Minor, Serious, Fatal) and identify dynamic geographical risk hotspots, ultimately producing an explainable 0-100 Risk Score for emergency resource allocation.

## 2. Data Sources
*   **MoRTH (India):** State-level aggregate baseline.
*   **US Accidents (Kaggle):** 7.7M records, detailed environmental/temporal features.
*   **UK STATS19:** High-quality verified injury severity data.

## 3. Data Pipeline
*   **Ingestion:** Automated download scripts (Kaggle API, HTTP).
*   **Unification:** Map disparate schemas to a single target standard.
*   **Preprocessing:** Median/Mode imputation, Label/Ordinal Encoding, StandardScaler (fit on train set only).

## 4. Feature Engineering
*   **Temporal:** Extract hour, day, month, `is_weekend`, `rush_hour`, `is_night`.
*   **Environmental:** Bin weather conditions (`weather_severity`), group visibility and temperature.
*   **Spatial:** Flag presence of junctions, crossings, and signals.

## 5. ML Models
*   **Baseline:** Logistic Regression.
*   **Ensembles:** Random Forest, XGBoost, LightGBM.
*   **Imbalance Handling:** Apply SMOTE on the training set to boost Fatal-class learning.

## 6. Evaluation Metrics
*   Primary Metric: **Fatal-Class Recall** (minimizing False Negatives).
*   Secondary Metrics: Precision, F1-Score, Overall Accuracy, ROC-AUC.

## 7. Risk-Score Methodology
Develop a formula combining:
*   $P_s$ (Model prediction of severity) - 40%
*   $R_t$ (Temporal risk factors) - 20%
*   $R_e$ (Environmental risk factors) - 20%
*   $R_g$ (Geographic historical density) - 20%
Normalize to 0-100 (Low, Moderate, High, Critical).

## 8. Hotspot Methodology
*   Implement K-Means (baseline clustering).
*   Implement DBSCAN (density-based, noise filtering).
*   Implement KDE (visual heatmaps).
*   Compare effectiveness for road networks.

## 9. Explainability
*   Integrate SHAP for global feature importance reporting.
*   Integrate LIME for local, real-time dashboard explanations.

## 10. Dashboard & Visualization
*   Generate static evaluation plots (Confusion Matrices, ROC curves).
*   Generate interactive Folium maps for hotspots.

## 11. Limitations & Assumptions
*   Assuming US traffic delay proxy maps reasonably to injury severity.
*   Assuming historical hotspot data is predictive of future risk.

## 12. Future Scope
*   GNN integration for road topology.
*   Real-time API ingestion.
