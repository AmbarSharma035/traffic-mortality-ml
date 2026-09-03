# Explainability: SHAP vs LIME

Complex machine learning models (like XGBoost or Random Forest) are often considered "black boxes." Explainable AI (XAI) techniques are essential to build trust and understand *why* a model made a specific prediction.

## 1. SHAP (SHapley Additive exPlanations)
*   **What it is:** A game-theoretic approach to explain the output of any ML model.
*   **How it works:** It calculates "Shapley values" from cooperative game theory. It imagines each feature as a "player" in a game where the "payout" is the model's prediction. It calculates the marginal contribution of each feature across all possible combinations of features.
*   **Scope:** 
    *   **Global Explainability:** Shows which features are most important overall across the entire dataset (e.g., "Weather is the most important factor globally").
    *   **Local Explainability:** Explains a single prediction (e.g., "For this specific accident, the rain increased the risk by 15%, but the daylight decreased it by 5%").

## 2. LIME (Local Interpretable Model-agnostic Explanations)
*   **What it is:** A technique that explains the predictions of any classifier in an interpretable and faithful manner by learning an interpretable model locally around the prediction.
*   **How it works:** To explain a single prediction, LIME generates a new dataset consisting of perturbed samples (slight variations) of the instance being explained. It gets the black-box model's predictions for these variations. Then, it trains a simple, interpretable model (like a linear regression) on this local dataset.
*   **Scope:** 
    *   **Local Explainability ONLY.** It is excellent for understanding individual predictions but does not provide a robust global view.

## Comparison

| Feature | SHAP | LIME |
| :--- | :--- | :--- |
| **Foundation** | Cooperative Game Theory | Local Surrogate Models |
| **Scope** | Global & Local | Local only |
| **Consistency** | Highly consistent (mathematically proven) | Can be unstable (different explanations for similar points) |
| **Speed** | Slow (especially for large datasets, though TreeSHAP is faster) | Fast (only trains a simple model locally) |
| **Output Interpretation**| Additive values that sum to the model's output | Coefficients of a local linear model |

## When to use each in this project:
*   **Use SHAP** for the final project report to show the overall feature importance (Global) and for generating stable explanations for critical, high-risk predictions.
*   **Use LIME** for rapid, real-time explanations in a dashboard where speed is prioritized over mathematical consistency, or when explaining very specific edge cases.
