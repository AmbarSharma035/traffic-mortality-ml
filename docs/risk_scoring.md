# Risk Scoring Methodology

The risk scoring module provides a composite score (0-100) representing the danger level of a specific road segment or set of conditions. It is designed as a decision-support metric.

## 1. Components and Weights
The total Risk Score is calculated as a weighted sum of four components. These weights can be adjusted in `config/risk_config.yaml`.

*   **Severity Probability ($P_s$) - Weight: 40%**
    *   Derived from the ML model's predicted probability of a 'Serious' or 'Fatal' accident occurring under the given conditions.
*   **Temporal Risk ($R_t$) - Weight: 20%**
    *   Assesses risk based on time features (e.g., night-time, rush hour, weekends).
*   **Environmental Risk ($R_e$) - Weight: 20%**
    *   Assesses risk based on weather, lighting, and road surface conditions (e.g., rain, snow, poor visibility, darkness).
*   **Geographic Density/History ($R_g$) - Weight: 20%**
    *   Derived from hotspot analysis. How frequently do severe accidents happen in this specific location?

## 2. Formula
$$ Risk Score = (W_s \times P_s) + (W_t \times R_t) + (W_e \times R_e) + (W_g \times R_g) $$
Where $W_s + W_t + W_e + W_g = 1.0$.

## 3. Normalization
Each component is scaled to a 0-100 range before applying weights, ensuring the final Risk Score naturally falls between 0 and 100.

## 4. Risk Categories
Based on the final score (0-100), situations are classified into categories:

| Category | Score Threshold | Description |
| :--- | :--- | :--- |
| **Low** | 0 - 30 | Routine driving conditions. Normal caution required. |
| **Moderate**| 31 - 60 | Elevated risk (e.g., light rain, moderate traffic). |
| **High** | 61 - 85 | Significant risk (e.g., poor weather, known dangerous junction). |
| **Critical**| 86 - 100 | Extreme danger (e.g., severe storm, high-speed night driving in a hotspot). Immediate intervention recommended. |

## 5. Disclaimer
> [!WARNING]
> This Risk Score is a **project-defined decision-support metric** developed for academic purposes. It is NOT an official safety rating from any government or transport authority. It is intended to highlight patterns and assist in hypothetical resource allocation (e.g., ambulance positioning).
