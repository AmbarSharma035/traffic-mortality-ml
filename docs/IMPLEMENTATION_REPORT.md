# Implementation Report

**Project Title:** Reducing Traffic Mortality Using Machine Learning: A Predictive Framework for Accident Severity Classification & Risk Hotspot Identification

## 1. Introduction
Traffic accidents remain a leading cause of unnatural deaths globally. Traditional reactive measures (improving roads after accidents happen) are insufficient. This project proposes a proactive approach using machine learning to predict accident severity and identify geographical hotspots based on environmental, temporal, and spatial factors.

## 2. Problem Statement
Emergency response systems currently lack the ability to predict the severity of an accident before first responders arrive, leading to sub-optimal resource allocation. Furthermore, static accident maps do not account for dynamic variables like weather and time.

## 3. Objectives
*   To develop a unified ML pipeline capable of predicting accident severity across diverse datasets.
*   To solve the class imbalance problem inherent in accident data to ensure high recall for fatal accidents.
*   To implement spatial clustering techniques for dynamic hotspot detection.
*   To build a composite Risk Scoring system for decision support.
*   To ensure model transparency using Explainable AI (XAI) techniques.

## 4. Literature Survey
Previous studies have extensively used Logistic Regression and basic clustering for traffic analysis. Recent advancements highlight the superiority of ensemble methods like Random Forest and XGBoost in handling non-linear spatial-temporal data. However, many studies evaluate models using raw accuracy, failing to address the critical class imbalance problem, which this project tackles via SMOTE.

## 5. Dataset
The project utilizes three datasets to ensure robustness:
1.  **US Accidents (Kaggle):** 7.7 million records (2016-2023) containing weather, POI, and temporal data.
2.  **UK STATS19:** Highly structured governmental data detailing injury severity.
3.  **MoRTH (India):** Aggregate data used for macro-level analysis.
A unified schema was created mapping target variables to three classes: Minor (0), Serious (1), and Fatal (2).

## 6. Data Preprocessing
*   **Imputation:** Median for numeric variables, Mode for categorical variables.
*   **Encoding:** Label encoding for high-cardinality features, ordinal encoding where applicable.
*   **Scaling:** StandardScaler applied to numeric features to optimize gradient-descent algorithms.
*   **Data Leakage Prevention:** All scaling and imputation parameters were derived exclusively from the training fold after a train-test split.

## 7. Feature Engineering
Temporal features (hour, day_of_week, is_weekend, rush_hour) and environmental categories (weather_severity, visibility_category) were derived from timestamps and raw text data to provide clearer signals to the models.

## 8. Methodology
The system architecture consists of a data ingestion pipeline, a preprocessing and SMOTE engine, a model training module evaluating multiple algorithms, a spatial clustering module for hotspot detection, and a scoring module that combines outputs into a final Risk Score.

## 9. ML Models
Four machine learning models were implemented, trained, and comparatively evaluated:
1. **Logistic Regression (Multinomial)**: Linear decision boundaries with L2 regularization (`max_iter=1000`).
2. **Random Forest Classifier**: Ensemble of 100 decision trees (`max_depth=12`, `n_jobs=-1`).
3. **XGBoost Classifier**: Gradient boosted decision trees (`n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, multi-class log-loss).
4. **LightGBM Classifier**: Leaf-wise gradient boosting (`n_estimators=100`, `num_leaves=31`, `learning_rate=0.1`).

All models were trained on SMOTE-balanced training data and tested on an untouched test set of 15,639 samples.

## 10. SMOTE (Synthetic Minority Over-sampling Technique)
Accident severity exhibits extreme class imbalance:
* **Pre-SMOTE Training Distribution**: Minor: 55,508 (76.1%), Serious: 16,406 (22.5%), Fatal: 1,066 (1.5%).
* **Post-SMOTE Training Distribution**: Minor: 55,508 (33.3%), Serious: 55,508 (33.3%), Fatal: 55,508 (33.3%).

**CRITICAL PROTOCOL**: SMOTE was applied **exclusively to the training split** after stratified 70/15/15 train/val/test partitioning. The validation (15,639) and test (15,639) sets remained strictly un-oversampled to preserve real-world natural class distributions and prevent data leakage.

## 11. Evaluation
Evaluation was conducted on the untouched test partition (15,639 collisions: 11,895 Minor, 3,516 Serious, 228 Fatal). Because failing to predict a fatal crash carries catastrophic consequences in emergency dispatch, **Fatal-Class Recall** was defined as the primary model selection metric rather than raw overall accuracy.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Fatal Recall (Primary) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | 43.05% | **0.3437** | **0.4041** | 0.2781 | 0.5119 | **56.14% (128 / 228)** |
| **Random Forest** | 60.27% | 0.3393 | 0.3399 | 0.3379 | 0.6181 | 1.75% (4 / 228) |
| **XGBoost** | 67.04% | 0.3496 | 0.3418 | **0.3410** | 0.6523 | 0.88% (2 / 228) |
| **LightGBM** | **70.44%** | 0.3471 | 0.3381 | 0.3299 | **0.6613** | 0.44% (1 / 228) |

**Key Finding**: Complex gradient-boosted trees achieved high nominal accuracy (67–70%) by overwhelmingly predicting the majority 'Minor' class, virtually ignoring fatal accidents (0.4–1.8% recall). In contrast, Logistic Regression with SMOTE achieved **56.14% Fatal Recall**, successfully flagging over half of all fatal crashes. Consequently, Logistic Regression was selected as the operational deployment model.

## 12. Risk Scoring
A composite exposure-based Risk Score (0–100) was computed across all 104,258 records:
$$\text{Risk Score} = 0.4 \times S_{\text{severity}} + 0.2 \times S_{\text{temporal}} + 0.2 \times S_{\text{environmental}} + 0.2 \times S_{\text{geographic}}$$

* **Score Distribution**: Mean: 33.87, Std: 5.37, Min: 21.89, Max: 57.16.
* **Risk Categorization**:
  * Low (0–30): 26,381 locations (25.3%)
  * Moderate (31–60): 77,877 locations (74.7%)
  * High (61–80): 0
  * Critical (81–100): 0

## 13. Hotspot Detection
* **K-Means Clustering ($k=20$)**: Partitioned 104,258 collision coordinates into 20 regional zones. The dominant cluster centered on Greater London with 26,300 collisions (25.2% of all UK accidents).
* **DBSCAN Clustering ($\epsilon=5\text{ km}$, $\text{min\_samples}=10$)**: Identified **173 dense accident corridors** while filtering 3,801 isolated collisions as background noise.
* **Kernel Density Estimation (KDE)**: Computed a 2D spatial probability density surface ($\text{bandwidth}=0.01^\circ$) exported as an interactive Folium heatmap overlay.

## 14. SHAP/LIME Explainability
* **Global Interpretability (SHAP)**: KernelExplainer evaluated feature importance on sample records. Lighting condition, road features, hour of day, and junction presence emerged as the strongest drivers of fatal severity predictions.
* **Local Interpretability (LIME)**: LimeTabularExplainer generated feature attribution bar charts for individual accident profiles, enabling traffic safety engineers to diagnose specific causal risk factors (e.g., unlit dark road + night hour contributing positively toward fatal severity).

## 15. Dashboard
A full-stack, interactive Streamlit application (`app/streamlit_app.py`) was developed featuring 7 comprehensive pages:
1. **Overview & KPI Dashboard**: Total records (104,258), model accuracy, fatal recall, class breakdown.
2. **Real-Time Severity Prediction**: Interactive form allowing input of time, weather, road conditions, yielding predicted class and probabilities.
3. **Interactive Risk Map**: Geospatial Folium map with clustered markers color-coded by risk category.
4. **Hotspot Analysis**: Visualization of K-Means cluster centers and DBSCAN corridors.
5. **Temporal Patterns**: Hourly, daily, monthly, and weekend accident distribution charts.
6. **Model Explainability**: Embedded SHAP beeswarm/bar plots and LIME case studies.
7. **Model Comparison**: Full comparative metric tables and confusion matrix heatmaps.

## 16. Results
* **Primary Metric Achieved**: 56.14% fatal accident recall with Logistic Regression on unseen test data, up from near 0% without SMOTE.
* **Data Scale**: 104,258 real collisions processed and harmonized with 17 engineered features.
* **Test Suite Verification**: 56 comprehensive automated tests passing with 100% success rate across data loading, preprocessing, feature matrices, models, risk scoring, hotspots, and dashboard components.

## 17. Limitations
*   **Proxy Severity Mapping:** The US dataset measures traffic delay, not physical injury. Mapping long delays to 'Fatal' is a proxy assumption that introduces noise.
*   **Computational Cost:** Kernel Density Estimation (KDE) and SHAP are computationally expensive on datasets exceeding millions of rows.

## 18. Future Scope
*   Integration with real-time API feeds (e.g., TomTom, OpenWeather).
*   Implementation of Graph Neural Networks (GNNs) to better model the spatial topology of road networks.
*   Development of a mobile application for real-time driver alerts.

## 19. Conclusion
This project successfully demonstrates that machine learning can move traffic safety analysis from a reactive statistical exercise to a predictive, dynamic system. By prioritizing Fatal-class recall through SMOTE and providing explainable composite risk scores, the framework offers a viable prototype for modern emergency response and proactive infrastructure planning.
