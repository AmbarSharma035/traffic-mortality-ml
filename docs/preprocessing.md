# Data Preprocessing Pipeline

This document details the preprocessing steps applied to the unified data before model training.

## 1. Missing Value Handling
Missing data is a common issue in real-world datasets. We use robust imputation strategies:
*   **Numeric Features:** Missing values in numeric columns (e.g., `temperature`, `humidity`, `visibility`) are imputed using the **median** value of that column in the training set. The median is resistant to outliers.
*   **Categorical Features:** Missing values in categorical columns (e.g., `weather_condition`, `road_surface`) are imputed using the **mode** (most frequent value) in the training set.

## 2. Categorical Encoding
Machine learning algorithms require numerical input.
*   **Ordinal Encoding:** Categorical variables with a natural order (e.g., `weather_severity`, `time_of_day`) are mapped to ordered integers.
*   **Label Encoding:** High-cardinality non-ordinal variables (e.g., `region`, `weather_condition` if not ordinal) are encoded using integers. *(Note: One-Hot Encoding is avoided for high-cardinality features to prevent the curse of dimensionality).*

## 3. Feature Scaling
Distance-based algorithms (like K-Means) and gradient descent-based algorithms (like Logistic Regression) are sensitive to the scale of features.
*   We apply **StandardScaler** to numeric features to ensure they have a mean of 0 and a standard deviation of 1.
*   **Crucial Rule:** Scaling parameters (mean and standard deviation) are learned **only from the training data** and then applied to the validation/test data to prevent data leakage. Tree-based models (RF, XGBoost) generally do not require scaling, but it is applied for consistency and to support models that do.

## 4. Class Imbalance Handling (SMOTE)
Traffic accident severity is highly imbalanced (e.g., Minor accidents vastly outnumber Fatal accidents).
*   We use **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic examples of the minority classes (Serious, Fatal).
*   **Crucial Rule:** SMOTE is applied **ONLY to the training data** after the train-test split. Applying it before splitting causes severe data leakage, leading to artificially inflated evaluation metrics. The validation and test sets must represent the true, imbalanced real-world distribution.

## 5. Data Leakage Prevention
Data leakage occurs when information from outside the training dataset is used to create the model. We prevent this by:
*   Splitting the data into train/validation/test sets *before* any imputation, scaling, or SMOTE operations.
*   Fitting imputers and scalers exclusively on the training set.
