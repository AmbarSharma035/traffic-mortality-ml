# Viva Questions and Answers

This document prepares you for a B.Tech project viva with short, clear answers.

## Section 1: Project Overview & Objectives

**1. What is the main objective of your project?**
To develop a machine learning framework that predicts traffic accident severity and identifies high-risk geographical hotspots. The ultimate goal is to provide decision-support tools for better resource allocation, like ambulance routing.

**2. Why did you choose ML for traffic mortality reduction?**
Traditional traffic safety analysis relies on historical statistics. Machine learning allows us to be predictive—combining weather, time, and location to forecast danger dynamically before it happens.

**3. What is the difference between predicting severity and calculating a risk score?**
Severity prediction is the raw ML output (e.g., 80% chance of a Fatal accident). The Risk Score is a broader composite metric (0-100) that combines this ML prediction with historical hotspot data, temporal factors, and environmental conditions.

## Section 2: Data Sources & Preprocessing

**4. What datasets did you use?**
We integrated three sources to prove the model's adaptability: MoRTH aggregate data for India, the US Accidents dataset (7.7 million records), and UK STATS19 data.

**5. How did you handle missing values?**
We used robust imputation techniques to prevent data loss. Numeric features were imputed using the median (which is resistant to outliers), and categorical features were imputed using the mode.

**6. Why scale data, and when did you do it?**
Scaling ensures features with large numbers (like distance) don't dominate small numbers (like hour of day). We scaled data *after* the train-test split, fitting only on the training data, to prevent data leakage.

**7. How did you unify severity labels across different datasets?**
Since datasets use different scales (e.g., US uses 1-4 for delay, UK uses 1-3 for injury), we mapped them to a common set: Minor (0), Serious (1), and Fatal (2).

## Section 3: Handling Imbalance (SMOTE)

**8. What is the class imbalance problem in your dataset?**
In the real world, minor accidents vastly outnumber fatal ones. If 95% of data is 'Minor', a model could just guess 'Minor' every time and achieve 95% accuracy while failing to identify any fatalities.

**9. What is SMOTE and why did you use it?**
SMOTE stands for Synthetic Minority Over-sampling Technique. It creates synthetic, realistic examples of the minority classes (Serious, Fatal) by interpolating between existing minority data points.

**10. When applying SMOTE, why must it be done after train-test splitting?**
If applied before splitting, synthetic data generated from the test set leaks into the training set. This causes the model to overfit and artificially inflates evaluation metrics. 

## Section 4: Machine Learning Models

**11. Which models did you test?**
We tested Logistic Regression (as a baseline), Random Forest, XGBoost, and LightGBM.

**12. Explain Random Forest in simple terms.**
It's an ensemble method that builds many decision trees using random subsets of data. The final prediction is determined by a majority vote from all the trees, which makes it robust against overfitting.

**13. What is the difference between XGBoost and LightGBM?**
Both are gradient boosting algorithms. XGBoost grows trees level-by-level, while LightGBM grows trees leaf-wise (choosing the most promising branch to split). LightGBM is generally faster and uses less memory.

**14. Why might tree-based models perform better than Logistic Regression here?**
Traffic risk involves non-linear relationships (e.g., risk is high early morning, low midday, high late night). Tree-based models capture these complex, non-linear interactions automatically.

## Section 5: Evaluation Metrics

**15. Why is Accuracy a poor metric for this project?**
Due to high class imbalance. An accuracy of 90% is useless if the 10% it got wrong represents every single fatal accident in the dataset.

**16. Why is Fatal-class Recall your primary metric?**
Recall measures how many actual positive cases we correctly identified. In traffic safety, a False Negative (predicting Minor when it's actually Fatal) means ambulances aren't dispatched. Maximizing Fatal-class Recall minimizes these life-threatening errors.

**17. What is the F1-Score?**
It is the harmonic mean of Precision and Recall. It provides a balanced measure when you care about both false positives and false negatives.

## Section 6: Hotspot Detection

**18. What is a hotspot?**
A geographical location with a statistically high concentration of severe accidents.

**19. What is DBSCAN and how is it different from K-Means?**
DBSCAN is a density-based clustering algorithm. Unlike K-Means, you don't need to specify the number of clusters in advance, and it can identify arbitrarily shaped clusters (like a long stretch of highway) while ignoring isolated accidents as 'noise'.

**20. Why use KDE (Kernel Density Estimation)?**
KDE creates a continuous probability surface rather than discrete clusters. It is excellent for generating visual heatmaps for dashboards.

## Section 7: Explainability (SHAP & LIME)

**21. Why is explainability important in your project?**
Models like XGBoost are "black boxes." If we advise authorities to change infrastructure based on a prediction, we must be able to explain *why* the model made that prediction.

**22. What does SHAP do?**
SHAP uses game theory to calculate the marginal contribution of every feature to the final prediction, providing consistent global and local explanations.

**23. What does LIME do?**
LIME creates a simple, interpretable model locally around a single prediction. It is fast and useful for explaining individual predictions on a dashboard.

## Section 8: System & Limitations

**24. How is your Risk Score calculated?**
It's a weighted sum of four components: The model's severity prediction probability, Temporal risk (time of day/week), Environmental risk (weather/lighting), and Geographic density (historical hotspots).

**25. What is the main limitation of your project?**
The proxy mapping of the US Accidents dataset. The original data measures traffic delay (1-4), not explicitly injury severity. Mapping '4' to 'Fatal' is an assumption that introduces noise.

**26. How would you deploy this project in the real world?**
It could be deployed as a REST API that feeds into a real-time dashboard used by emergency dispatchers, or integrated into navigation apps to route drivers away from dynamic high-risk zones.

*(Note: These are 26 core questions. You can expand on these based on the other documentation provided).*
