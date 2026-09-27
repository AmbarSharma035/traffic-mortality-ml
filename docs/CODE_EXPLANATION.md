# Code Explanation

This covers the code worth being able to explain confidently in a viva or interview —
not every line, but every non-obvious decision.

## 1. Why labels are converted to integers before training

**File:** `src/models/severity_model.py`

```python
CLASS_ORDER = config.SEVERITY_CLASSES  # ["Minor", "Serious", "Fatal"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}   # Minor=0, Serious=1, Fatal=2
IDX_TO_LABEL = {i: c for c, i in LABEL_TO_IDX.items()}

def encode_labels(y: pd.Series) -> pd.Series:
    return y.map(LABEL_TO_IDX)
```

**Why:** XGBoost and LightGBM's multi-class objectives require integer class labels
(`0, 1, 2, ...`), not strings. Logistic Regression and Random Forest would happily
accept the strings directly, but training all four models on the *same* encoded labels
keeps the comparison fair and the code uniform. Everything downstream (metrics,
predictions) maps back to `"Minor"/"Serious"/"Fatal"` using `IDX_TO_LABEL`, so nothing
user-facing ever shows a bare `0`, `1`, `2`.

**Watch out for:** `pipeline.predict_proba()` returns columns in `pipeline.classes_`
order, which is the *sorted* order of whatever labels you trained on — for integers
`[0, 1, 2]` that's conveniently already `[Minor, Serious, Fatal]` order, so no reordering
was needed for the classifiers. But **`roc_auc_score` also needs `labels` sorted
ascending**, which came up as a real bug during development:

```python
# src/utils/metrics.py
order = np.argsort(labels)
sorted_labels = list(np.asarray(labels)[order])
y_proba_sorted = np.asarray(y_proba)[:, order]
roc_auc = roc_auc_score(y_true, y_proba_sorted, multi_class="ovr", labels=sorted_labels)
```
This explicitly reorders the probability columns to match sorted label order before
calling `roc_auc_score`, rather than assuming the caller already passed them sorted.

## 2. SMOTE inside the pipeline, not before the split

**File:** `src/models/severity_model.py`

```python
def _make_pipeline(classifier) -> ImbPipeline:
    return ImbPipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("smote", SMOTE(random_state=config.RANDOM_SEED, k_neighbors=5)),
        ("classifier", classifier),
    ])
```

**Why `imblearn.pipeline.Pipeline` and not `sklearn.pipeline.Pipeline`:** a normal
scikit-learn pipeline doesn't know how to skip a step at prediction time. `SMOTE` should
only ever run during `.fit()` on the *training* data — never during `.predict()`, and
never on the test/validation data. `imblearn`'s `Pipeline` handles this automatically:
during `.fit()` it resamples between the preprocessor and the classifier; during
`.predict()`/`.predict_proba()` it's a no-op passthrough. This is what SRS requirement
FR-3.3-05 ("imbalance-handling techniques only to the training split") means in code,
and getting this wrong (e.g. running SMOTE on the whole dataset before splitting) is
one of the most common accidental sources of data leakage in imbalanced classification
projects — worth being able to explain *why* it would be wrong if asked.

## 3. Why Logistic Regression beat the tree ensembles on Fatal recall

This is a genuine, reproducible finding from this project's own benchmark
(`models/metrics.json`), not a bug:

| Model | Fatal recall | Macro F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | **0.53** | 0.337 | 0.611 |
| Random Forest | 0.03 | 0.352 | 0.613 |
| XGBoost | 0.00 | 0.320 | 0.590 |
| LightGBM | 0.00 | 0.320 | 0.579 |

All four models have a **similar ROC-AUC** (~0.58–0.61) — meaning they all learned a
broadly similar ability to *rank* Fatal cases as riskier. But their default decision
(`argmax` of predicted probability) behaves very differently: the tree ensembles almost
never actually predict "Fatal" as their top class, even for genuinely high-risk rows,
while Logistic Regression does so much more often.

**The likely explanation** (worth stating plainly if asked "why," rather than treating
it as mysterious): SMOTE synthesizes new minority-class points by linearly interpolating
between existing ones **in the one-hot-encoded, scaled feature space**. Tree-based
models split on sharp thresholds and can carve out very tight, confident decision
regions around those synthetic points — regions that describe the *synthetic* Fatal
examples well but don't generalize to the real, messier Fatal examples in the test set.
A linear model like Logistic Regression can't build such a sharp, overfit boundary, so
its probability estimates end up better calibrated on unseen data even though its
overall discriminative power (ROC-AUC) isn't higher.

**Why this is a legitimate result to present, not something to hide:** it directly
demonstrates why the project's evaluation strategy (`docs/PROJECT_QA.md` — "recall on
the Fatal class is the north star, not overall accuracy") matters — accuracy alone
(see the table: RF/XGBoost/LightGBM all have *higher* accuracy than Logistic
Regression, ~0.73–0.75 vs 0.51) would have picked exactly the wrong model for this
problem.

## 4. Cost-sensitive decision thresholding

**File:** `src/models/severity_model.py`

```python
FATAL_IDX = LABEL_TO_IDX["Fatal"]  # 2

def apply_fatal_threshold(proba_ordered: np.ndarray, threshold: float = None) -> np.ndarray:
    threshold = config.FATAL_DECISION_THRESHOLD if threshold is None else threshold
    argmax_pred = np.argmax(proba_ordered, axis=1)
    fatal_triggered = proba_ordered[:, FATAL_IDX] >= threshold
    return np.where(fatal_triggered, FATAL_IDX, argmax_pred)
```

**What it does:** instead of trusting `argmax(probabilities)` — which implicitly uses a
50% threshold and, under ~2–3% Fatal prevalence, will rarely pick Fatal even when the
model's own `P(Fatal)` is meaningfully elevated — this flags a case as Fatal whenever
`P(Fatal)` clears a much lower bar (`config.FATAL_DECISION_THRESHOLD = 0.15`). Below
that bar, the ordinary argmax between Minor/Serious still applies.

**Why this is defensible, not "cheating":** the whole project's evaluation philosophy
(slide 5: *"a missed fatal-risk case costs far more than a false alarm"*) is a
statement about **asymmetric costs**. A fixed 0.5 threshold is optimal only when a false
positive and a false negative cost the same — which is explicitly not true here. Lowering
the decision threshold for the costly class is a standard, textbook cost-sensitive
learning technique, and it's implemented as a clearly separate, labelled function
(`apply_fatal_threshold`) rather than silently baked into training, so the trade-off
being made is visible and explainable rather than hidden.

**In the dashboard:** both the raw `argmax_prediction` and the
`recall_oriented_prediction` are shown side-by-side (see `predict_single()`), so a user
can see when the threshold rule actually changed the answer.

## 5. Laplace/credibility smoothing for the risk score

**File:** `src/models/risk_model.py`

```python
k = config.RISK_SMOOTHING_K  # 8
agg["smoothed_avg_weight"] = (
    (agg["weighted_sum"] + k * global_avg_weight) / (agg["n_accidents"] + k)
)
```

**The problem this solves:** imagine one road/time/weather segment has exactly 2
recorded accidents, and both happened to be Fatal. Its *raw* average severity weight
would be the maximum possible (6.0, since Fatal=6 in `config.RISK_SEVERITY_WEIGHTS`) —
higher than a segment with 500 accidents where only 30 were Fatal. That's misleading:
2 data points is not enough evidence to call a segment more dangerous than one with
hundreds of observations.

**How the fix works:** this is Bayesian-flavoured smoothing (closely related to
"Laplace smoothing" from Naive Bayes, and to what's sometimes called "credibility
weighting" in insurance risk-scoring). It blends the segment's own observed average
with the *dataset-wide* average (`global_avg_weight`), weighted by a virtual sample size
`k=8`. A segment with very few real accidents (n << 8) gets pulled strongly toward the
global average; a segment with hundreds of accidents (n >> 8) is barely affected,
because its own data dominates the formula. This is the same underlying idea as
`(successes + k*prior) / (trials + k)` — it can be re-derived on demand as
"a weighted average between what we observed and what we'd expect by default,
weighted by how much we trust the observation."

## 6. Why hotspot clustering never looks at severity, but ranking does

**File:** `src/hotspot/hotspot_detection.py`

```python
def run_kmeans(df, k=None):
    coords = df[["latitude", "longitude"]].values   # ONLY lat/lon
    ...

def rank_hotspots(df, cluster_labels, top_n=None):
    ...
    weights = work["severity"].map(config.RISK_SEVERITY_WEIGHTS)  # severity used HERE
```

**Why this separation matters (and is worth stating explicitly if asked):**
clustering algorithms group points by *similarity in the feature space you give them*.
If severity were included as a clustering feature, the algorithm would partly be
grouping "similarly-severe" accidents together rather than "geographically close"
accidents — muddying the very question hotspot detection is meant to answer
("where do crashes concentrate", independent of any single cause — slide 7). Keeping
clustering spatial-only, and using severity only afterward to *rank* the clusters that
were found, keeps each step honest about what question it's answering.

## 7. The LIME wrapper — why it needs its own encoding step

**File:** `src/explainability/explain.py`

LIME's `LimeTabularExplainer` needs every column in its `training_data` to be numeric —
it does its own internal scaling and perturbation math, and it treats
`categorical_features` as small integer *codes*, not as strings, and not as one-hot
columns. But the actual model pipeline (`pipeline.predict_proba`) expects the *original*
string columns (`"weather": "Foggy"`, etc.) so its own internal `OneHotEncoder` can do
its job. These two requirements are incompatible unless something sits between them —
that's what this function does:

```python
def build_lime_explainer(pipeline, background_df):
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        encoded[col] = le.fit_transform(df[col].astype(str))   # e.g. "Foggy" -> 2
        categorical_names[idx] = list(le.classes_)              # so LIME can display names
        encoders[col] = le                                       # kept for decoding later
    ...
```

Then, at explanation time, `predict_fn` — the function LIME actually calls repeatedly
while perturbing the input — decodes LIME's integer codes back into strings *before*
handing the row to the real pipeline:

```python
def predict_fn(rows):
    rows_df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    for col, le in encoders.items():
        idx = rows_df[col].round().clip(0, len(le.classes_) - 1).astype(int)
        rows_df[col] = le.inverse_transform(idx)   # 2 -> "Foggy", back to a real string
    ...
    return pipeline.predict_proba(rows_df)
```

The `.round().clip(...)` matters because LIME perturbs categorical codes as
floating-point values close to a valid integer (e.g. `1.97` instead of `2`) — rounding
and clipping guarantees a valid category is always decoded, even from LIME's noisy
perturbations.

## 8. The shared `ColumnTransformer` — one definition, four models, no duplication

**File:** `src/features/feature_engineering.py`

```python
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
```

**Why `handle_unknown="ignore"` matters:** without it, if the dashboard's input form
ever produces a category combination that happened not to appear in the training data
(unlikely with this feature set, but a real risk in general), `.transform()` would raise
an error instead of returning a valid (all-zero) encoding. This one keyword argument is
what makes the dashboard robust to arbitrary user input rather than crashing on an edge
case.

**Why one function, called from three places:** `severity_model.py`'s classifiers,
`risk_model.py`'s regressor (a separate `ColumnTransformer` instance, but same design
pattern), and `explain.py`'s SHAP feature-name lookup all rely on `FEATURE_COLUMNS`
and/or `build_preprocessor()` being defined in exactly one place. If a new feature is
added to `add_derived_features()` and to `CATEGORICAL_FEATURES`/`NUMERIC_FEATURES`, it
becomes available to every model automatically — there's no second list to remember to
update.

## 9. `main.py` as the project's table of contents

`main.py` is deliberately written to be read top-to-bottom as a narrative — each
`section("STEP N - ...")` print statement is a checkpoint a presenter can literally
read off during a demo: "here's ingestion, here's cleaning, here's Pillar 1 training and
model selection, here's Pillar 2, here's Pillar 3, here's the explainability sanity
check." If you're asked to walk through the project live, running `python main.py` and
narrating each printed section is close to a ready-made script.

## 10. Test design philosophy

`tests/test_models.py` doesn't test "does the model get high accuracy" (that would make
tests flaky and tie them to a specific random seed's luck). It tests **properties that
must always hold regardless of the exact numbers**:
```python
def test_risk_lookup_scores_are_bounded_0_to_100(...):
    assert lookup["risk_score"].between(0, 100).all()

def test_dbscan_runs_and_ranking_is_sorted_descending(...):
    assert scores == sorted(scores, reverse=True)

def test_apply_fatal_threshold_flags_fatal_when_probability_clears_bar():
    # exact, hand-picked probability arrays with a known expected outcome
```
This is a useful distinction to be able to articulate: testing ML code isn't usually
about asserting a specific accuracy number (which is sensitive to data, seed, and
library versions) — it's about asserting invariants (bounded ranges, sorted output,
correct behaviour on a known hand-crafted input) that would catch a real bug without
being brittle.
