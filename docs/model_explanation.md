# Model & Algorithm Explanations (Beginner Friendly)

This document provides simple explanations of the algorithms and concepts used in this project, designed for clear communication during a viva.

## 1. Machine Learning Models

### 1.1 Logistic Regression
*   **Analogy:** Drawing a single straight line through data to separate "safe" from "dangerous."
*   **How it works:** It uses a mathematical equation to estimate the probability that an event belongs to a certain class (e.g., is it a Fatal accident or not?). If the probability is > 50%, it says 'Yes'.
*   **Pros:** Very fast, easy to understand.
*   **Cons:** Cannot handle complex, non-linear relationships (like when risk goes up in the morning, down at noon, and up again at night).

### 1.2 Random Forest
*   **Analogy:** A committee of experts. Instead of asking one person, you ask 100 people (decision trees), and they vote on the outcome.
*   **How it works:** It builds many individual "Decision Trees." Each tree looks at a random subset of the data and makes a prediction. The final answer is decided by a majority vote.
*   **Pros:** Highly accurate, handles complex data well, resistant to overfitting.
*   **Cons:** Can be slow to train, hard to interpret exactly how the final decision was reached (black box).

### 1.3 XGBoost (Extreme Gradient Boosting)
*   **Analogy:** A team learning from mistakes. The first person tries, fails slightly. The second person focuses *only* on the mistakes of the first. The third focuses on the mistakes of the second, and so on.
*   **How it works:** It builds trees sequentially. Each new tree is designed specifically to correct the errors made by the previous trees.
*   **Pros:** Usually provides the highest accuracy in competitions, handles missing data automatically.
*   **Cons:** Needs careful tuning, prone to overfitting if not tuned properly.

### 1.4 LightGBM
*   **Analogy:** Similar to XGBoost, but it builds its knowledge (trees) differently—growing the most promising branch deeply rather than growing all branches evenly.
*   **How it works:** It uses "leaf-wise" tree growth instead of "level-wise" (like XGBoost). 
*   **Pros:** Much faster training speed than XGBoost, uses less memory.
*   **Cons:** Can overfit easily on very small datasets.

## 2. Key Concepts

### 2.1 SMOTE (Synthetic Minority Over-sampling Technique)
*   **Why needed:** 95% of accidents might be 'Minor' and 5% 'Fatal'. The model might just learn to guess 'Minor' every time and get 95% accuracy!
*   **How it works:** It creates *fake but realistic* new data points for the minority class (Fatal). It looks at a Fatal accident, looks at its nearest Fatal neighbors, and creates a new point somewhere in between them.

### 2.2 Recall & Fatal-Class Recall
*   **Why accuracy isn't enough:** As above, guessing "Minor" yields 95% accuracy but misses every single fatality.
*   **Recall:** Out of all the *actual* Fatal accidents, how many did our model successfully identify?
*   **Cost:** Missing a Fatal prediction means no ambulance is dispatched. A false alarm (predicting Fatal when it's Minor) is a wasted trip, but missing a real fatality costs lives. Therefore, Recall is our most critical metric.

### 2.3 Clustering Algorithms (Hotspots)
*   **K-Means:** You tell the algorithm you want '5' hotspots. It groups all accidents into 5 circular zones based on average distance.
*   **DBSCAN:** You tell the algorithm "Find me areas where at least 10 accidents happened within 500 meters." It finds these dense clusters (of any shape, like along a road) and ignores isolated accidents as 'noise'.

### 2.4 Explainability (SHAP & LIME)
*   **SHAP:** Uses game theory to fairly distribute the "credit" for a prediction among all the features. Shows the global impact of features (e.g., "Speed is the #1 cause").
*   **LIME:** Creates a simple, temporary model just to explain *one specific prediction* (e.g., "This specific crash happened mainly because of the rain").

### 2.5 Risk Score
*   A custom number from 0-100 that combines the model's prediction with temporal (time of day), environmental (weather), and geographic (historical hotspot) factors to give an overall danger rating for a situation.
