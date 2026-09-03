# Severity Mapping Decisions

To train a unified model on datasets from different countries, we must standardize the target variable: accident severity. We mapped the original severity classes to a common set of labels: **Minor (0), Serious (1), and Fatal (2)**.

## 1. US Accidents (Sobhan Moosavi Dataset)
The original dataset uses a 1-4 scale, which primarily measures **traffic impact (delay)**, not necessarily injury severity. 
*   1: Short delay.
*   2: Moderate delay.
*   3: Long delay.
*   4: Severe delay / road closure.

**Mapping:**
*   1 & 2 -> **Minor (0)**
*   3 -> **Serious (1)**
*   4 -> **Fatal (2)**

## 2. UK STATS19
The UK dataset records injury severity explicitly.
*   3: Slight injury.
*   2: Serious injury.
*   1: Fatal injury.

**Mapping:**
*   3 (Slight) -> **Minor (0)**
*   2 (Serious) -> **Serious (1)**
*   1 (Fatal) -> **Fatal (2)**

## 3. Unified Labels Summary
*   **Minor (0):** Minor traffic impact, slight or no injuries.
*   **Serious (1):** Significant traffic delay, serious injuries.
*   **Fatal (2):** Severe road closure, fatal injuries.

## Limitations of Proxy Mapping
The US Accidents dataset mapping is a **proxy mapping**. Because the original labels (1-4) measure traffic delay rather than confirmed injuries/fatalities, labeling a class '4' as 'Fatal' is an approximation. A class 4 accident might involve an overturned truck causing a massive delay without fatalities, while a class 2 might involve a fatality that was quickly cleared from the roadway. This introduces noise into the model's target variable, which is a known limitation of using this specific dataset for injury-severity prediction.
