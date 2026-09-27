# Project Q&A (Interview / Viva Prep)

## Core Questions

### What problem does the project solve?
Road-safety analysis today is reactive: accidents are investigated after the fact and
statistics are published once a year, which is too slow and too coarse to guide
day-to-day intervention. This project builds three coordinated, usable outputs from the
same accident data instead: (1) predict how severe an accident is likely to be from its
conditions, (2) score how risky a road/time/weather combination is *before* an accident
happens, and (3) find and rank the physical locations where accidents keep
concentrating. See `docs/PROJECT_OVERVIEW.md` for the full problem statement.

### How does the project work, end to end?
Raw accident data → validated (`src/data/load_data.py`) → cleaned
(`src/data/preprocess.py`) → feature-engineered (`src/features/feature_engineering.py`)
→ split into three independent pipelines: Severity Prediction (4 benchmarked
classifiers, SMOTE, picked by Fatal recall), Risk Prediction (segment aggregation +
regressor), Hotspot Detection (K-Means/DBSCAN/KDE + severity-weighted ranking). Every
pillar's output is saved to `models/`. A Streamlit dashboard reads those files and
exposes all three, plus SHAP/LIME explanations for the severity predictions. Full trace
in `docs/DATA_FLOW.md`.

### Why did you choose these technologies?
Short version: scikit-learn for one consistent API across preprocessing, classical ML,
clustering *and* density estimation; XGBoost/LightGBM because the synopsis specifically
named them as the ensemble methods to benchmark; SMOTE (via imbalanced-learn) because
Fatal is ~2–3% of the data; SHAP+LIME as two independent explainability methods rather
than one; Streamlit+Folium because they turn a Python pipeline into a clickable
dashboard with no separate frontend build. Full rationale table in
`docs/PROJECT_OVERVIEW.md`.

### Explain the architecture.
It's a batch ML pipeline plus a read-only dashboard — no network API, no database
server. `main.py` runs the whole pipeline once and writes every artifact as a file
(`.joblib` for models, `.csv` for lookup/ranking tables, `.json` for metrics);
`dashboard/app.py` only ever reads those files, never retrains. Full ASCII diagram and
component-by-component breakdown in `docs/ARCHITECTURE.md`.

### Explain the database.
There isn't a database server (no PostgreSQL/SQLite/etc.) — this is a deliberate scope
choice. Every persisted artifact is a plain file: two CSVs (`data/raw/`,
`data/processed/`), two joblib-serialized scikit-learn/imblearn pipelines
(`severity_model.joblib`, `risk_model.joblib`), two CSV lookup tables
(`risk_lookup.csv`, `hotspots_ranked.csv`), and one JSON (`metrics.json`). This is
appropriate for a single-user, single-machine ML pipeline; a real multi-user deployment
(as sketched in the project's SRS document) would add a relational database for user
accounts, audit logs and dataset metadata — that's future-scope, not part of this
deliverable.

### Explain the APIs.
There is no REST/HTTP API in this implementation. The "API" is Python function calls
between modules (e.g. `dashboard/app.py` calls `predict_single()` from
`src/models/severity_model.py` directly, in the same process). The project's SRS
document specifies a REST API with JWT authentication for a future productionised
version; building that was out of scope for this academic ML deliverable, whose focus
is the modelling pipeline itself.

### Explain the important algorithms.
- **SMOTE** (Synthetic Minority Over-sampling Technique): generates synthetic minority
  class examples by interpolating between real minority examples and their nearest
  neighbours, used to counter the ~2–3% Fatal class prevalence.
- **Logistic Regression / Random Forest / XGBoost / LightGBM**: four different severity
  classifiers benchmarked against each other; see `docs/CODE_EXPLANATION.md` §3 for why
  Logistic Regression won on this dataset's Fatal-recall metric.
- **K-Means**: partitions accident coordinates into `k` clusters by minimizing
  within-cluster variance — good for broad, roughly circular zones.
- **DBSCAN**: density-based clustering that finds irregularly-shaped dense regions and
  explicitly labels sparse points as noise (`-1`) — better suited to real black-spots,
  which aren't circular.
- **Kernel Density Estimation (KDE)**: estimates a continuous density surface over the
  map rather than discrete clusters, used for the dashboard's heatmap layer.
- **Laplace/credibility smoothing**: blends a segment's own observed risk with the
  dataset-wide average, weighted by how many observations back it up — see
  `docs/CODE_EXPLANATION.md` §5.
- **SHAP / LIME**: two independent methods for explaining individual predictions —
  see `docs/CODE_EXPLANATION.md` §7.

### What challenges were faced?
1. **`sklearn`/`xgboost` API drift during development** — `LogisticRegression`'s
   `multi_class` parameter was removed in newer scikit-learn, and `roc_auc_score`
   requires sorted integer labels for multi-class ROC-AUC — both caused real runtime
   errors during development, fixed and documented in `docs/SETUP_GUIDE.md`'s
   troubleshooting section.
2. **SMOTE + tree ensembles generalizing poorly on the rare class** — RF/XGBoost/
   LightGBM all achieved near-zero Fatal recall on the test set despite SMOTE properly
   balancing the training data; Logistic Regression didn't have this problem. See
   `docs/CODE_EXPLANATION.md` §3 for the likely cause (trees overfitting to synthetic
   interpolated minority points) and why this is a legitimate finding worth presenting,
   not a bug to hide.
3. **LIME needs numeric input, the model pipeline needs strings** — required writing a
   label-encode/decode wrapper around LIME's explainer; see
   `docs/CODE_EXPLANATION.md` §7.
4. **No real dataset access in the development environment** — MoRTH/Kaggle/STATS19 all
   require a manual download or API key not available here, so a synthetic generator
   with realistic structure (including deliberately injected black-spot junctions) was
   built instead. This is disclosed prominently, not hidden — see
   `docs/PROJECT_OVERVIEW.md`.
5. **The risk score has no true exposure baseline** — see "What could be improved"
   below.

### What could be improved?
- **A real dataset.** The synthetic generator is structurally realistic but is not real
  accident data; results (exact metric values) would change with MoRTH/Kaggle/STATS19
  data, though the pipeline itself would not need to change.
- **A true exposure baseline for the risk model.** The current risk score measures "how
  severe are the accidents that DO happen in this segment" using only accident records —
  it cannot measure true incidence rate per vehicle-km travelled, because that would
  require traffic-volume data this project doesn't have. This is stated plainly in
  `src/models/risk_model.py`'s module docstring rather than glossed over.
  averaged over more folds/seeds), and possibly a cost-sensitive loss function (e.g.
  `sample_weight` proportional to `RISK_SEVERITY_WEIGHTS`) as an alternative to SMOTE
  for the tree-based models, which the current benchmark suggests SMOTE alone doesn't
  suit well.
- **A haversine-based DBSCAN metric** instead of the flat-earth degrees-to-km
  approximation currently used (fine at city scale, would matter at larger scale).
- **A REST API and a real multi-user web front end**, exactly as scoped in the
  project's SRS document, for a genuinely deployable version.

### What happens if something fails?
- **Bad input data:** `validate_schema()` raises `DataValidationError` immediately with
  a specific message (missing columns / bad severity labels / out-of-range coordinates)
  rather than letting a malformed dataset silently corrupt training.
- **Dashboard launched before training:** `load_artifacts()` checks that every expected
  file exists in `models/`; if not, the dashboard shows a clear on-screen error
  ("Run `python main.py` first") instead of crashing with a stack trace.
- **An unseen category value at inference time:** the shared `ColumnTransformer` is
  built with `OneHotEncoder(handle_unknown="ignore")`, so a category combination the
  model never saw in training encodes to all-zeros rather than raising an exception.
- **A pipeline step failing mid-run:** `main.py` runs as a plain top-to-bottom script —
  if any step raises, the run stops with a normal Python traceback at that exact step,
  and no partial/corrupted `models/` artifacts from *later* steps get written (each
  step's outputs are only saved after that step completes).

---

## 20 Beginner Questions

1. **Q: What does this project predict?**
   A: Three things: accident severity (Minor/Serious/Fatal) for a given crash, a 0–100
   risk score for a road/time/weather combination, and a ranked list of geographic
   accident hotspots.

2. **Q: What programming language is this built in?**
   A: Python 3 (developed and tested on 3.12; requires 3.10+).

3. **Q: What is the target/label for the severity model?**
   A: The `severity` column, with three possible values: `Minor`, `Serious`, `Fatal`.

4. **Q: Why is Fatal the rarest class?**
   A: Because in real accident data (and in this project's realistic synthetic
   generator), fatal accidents are a small fraction of all accidents — around 2–3% here
   — which is exactly why the project doesn't just optimize for overall accuracy.

5. **Q: What is class imbalance?**
   A: When one class in a classification problem has far fewer examples than the
   others — here, Fatal (~2–3%) versus Minor (~76%) and Serious (~21%).

6. **Q: What is SMOTE?**
   A: Synthetic Minority Over-sampling Technique — it creates new synthetic examples of
   the minority class by interpolating between existing minority examples and their
   nearest neighbours, so the training data has a more balanced class distribution.

7. **Q: Why is SMOTE only applied to the training data, not the test data?**
   A: To avoid data leakage — if synthetic points derived from test data leaked into
   training (or vice versa), the reported evaluation metrics would be misleadingly
   optimistic.

8. **Q: What does "recall" mean, in plain terms?**
   A: Out of all the actual Fatal accidents, what fraction did the model correctly
   identify as Fatal? High recall on Fatal means few real fatal cases are missed.

9. **Q: Why does this project care more about recall than accuracy?**
   A: Because a model can get very high accuracy just by always predicting "Minor" (the
   majority class) — that's useless in practice, since it means every Fatal case is
   missed. Recall on the Fatal class directly measures how many of the costliest cases
   are actually caught.

10. **Q: What is a confusion matrix?**
    A: A table showing, for each true class, how many predictions landed in each
    predicted class — it shows exactly which classes get confused with which.

11. **Q: What four models were compared for severity prediction?**
    A: Logistic Regression, Random Forest, XGBoost, and LightGBM.

12. **Q: Which one performed best, and by what measure?**
    A: Logistic Regression, measured by recall on the Fatal class (0.53, versus 0.00–0.03
    for the other three) — see `docs/CODE_EXPLANATION.md` §3 for why.

13. **Q: What is K-Means clustering used for here?**
    A: Grouping accident GPS coordinates into a fixed number (12, by default) of
    zones/regions based on geographic proximity, as one of two spatial hotspot methods.

14. **Q: What is DBSCAN, and how is it different from K-Means?**
    A: A density-based clustering algorithm that finds clusters of arbitrary shape and
    explicitly marks isolated points as "noise" — unlike K-Means, it doesn't need you
    to specify the number of clusters in advance, and it doesn't force every point into
    a cluster.

15. **Q: What does "hotspot" mean in this project?**
    A: A geographic location where accidents cluster together in unusually high
    numbers — a "black spot" in road-safety terminology.

16. **Q: What is SHAP used for?**
    A: Explaining which input features (speed, weather, time of day, etc.) pushed a
    specific severity prediction higher or lower — both across the whole dataset
    (global) and for one specific prediction (local).

17. **Q: What is a Streamlit dashboard?**
    A: A way to build an interactive web UI purely in Python, without writing separate
    HTML/CSS/JavaScript — used here to let a non-technical user query all three model
    outputs through forms, tables and a map.

18. **Q: Where is the trained model actually stored?**
    A: As a `.joblib` file (e.g. `models/severity_model.joblib`) — a serialized Python
    object that can be loaded back into memory without retraining.

19. **Q: Does the dashboard train the model itself?**
    A: No — training only happens when you run `python main.py`. The dashboard only
    loads the already-trained model files from `models/`.

20. **Q: What testing is done on this project?**
    A: 17 automated unit tests (via `pytest`), covering data cleaning, feature
    engineering, and correctness properties of all three pillars (e.g. risk scores stay
    between 0 and 100, hotspot rankings are sorted correctly).

---

## 20 Intermediate Questions

1. **Q: Why is feature engineering kept in a separate module from data cleaning?**
   A: Single-responsibility design — `preprocess.py` only fixes what's *wrong* with the
   data (duplicates, missing values, dtypes); `feature_engineering.py` only *derives*
   new information from already-clean data. This makes each easier to test and reason
   about independently, and means you can change the imputation strategy without
   touching what features get built, or vice versa.

2. **Q: Why use a `ColumnTransformer` instead of manually one-hot encoding each column?**
   A: It bundles categorical encoding and numeric scaling into one fitted object that's
   guaranteed to apply the *exact same* transformation at inference time as it learned
   at training time (e.g. the same set of one-hot columns, the same mean/std for
   scaling) — manually redoing this at inference risks subtle train/inference mismatch.

3. **Q: Why `handle_unknown="ignore"` on the `OneHotEncoder`?**
   A: So an unseen category combination at inference time (e.g. from the dashboard's
   form) encodes to all-zero columns instead of raising an exception — makes the
   pipeline robust to inputs it wasn't explicitly trained on.

4. **Q: Why does `evaluate_classifier()` compute ROC-AUC with `multi_class="ovr"`?**
   A: "One-vs-rest" — for a 3-class problem, it computes a separate AUC per class
   (each class vs. all others) and macro-averages them, since a single, shared decision
   threshold doesn't naturally extend to more than two classes.

5. **Q: Why does `roc_auc_score` need the `labels` argument sorted?**
   A: It's an internal scikit-learn requirement for `multi_class="ovr"` — the function
   raises `ValueError: Parameter 'labels' must be ordered` otherwise. This project's
   severity classes are ordinal (`Minor, Serious, Fatal`), not alphabetically sorted,
   so `src/utils/metrics.py` explicitly reorders the probability columns to match
   sorted label order before calling it.

6. **Q: What is the risk model's "segment", precisely?**
   A: A unique combination of `road_type`, `junction_type`, `weather`, `hour_bucket`,
   and `is_weekend` — defined by `SEGMENT_KEYS` in `src/models/risk_model.py`. Every
   accident falls into exactly one segment.

7. **Q: Why use a `GradientBoostingRegressor` on top of the aggregated risk lookup
   table, instead of just using the lookup table directly?**
   A: The lookup table only has a score for segment combinations that actually appear
   in the historical data. The regressor learns the general relationship between
   segment attributes and risk score, so it can produce a reasonable estimate for a
   combination that has few or zero historical accidents.

8. **Q: What does `sample_weight=np.log1p(n_accidents)` do in the risk regressor?**
   A: Gives more training influence to segments backed by more observations (a
   log-scale, so a segment with 1,000 accidents doesn't dominate 1,000x more than one
   with 10 — it's weighted, not simply duplicated).

9. **Q: Why is Laplace/credibility smoothing needed for the risk score?**
   A: Without it, a segment with only 2 recorded accidents (both Fatal) would score as
   the *maximum* possible risk — misleadingly, since 2 data points isn't enough
   evidence. Smoothing pulls low-count segments toward the dataset-wide average,
   proportional to how few observations back them up.

10. **Q: Why does hotspot clustering use only latitude/longitude, and not severity or
    time?**
    A: To keep the clustering question purely spatial ("where do accidents
    concentrate physically") rather than blending it with "where do similarly-severe
    accidents happen," which would answer a different, muddier question. Severity is
    used only afterward, to rank the clusters that were found spatially.

11. **Q: Why convert DBSCAN's `eps` parameter from kilometres to degrees?**
    A: `DBSCAN` operates on raw coordinate distances; latitude/longitude are in
    degrees, not kilometres, so `eps_km / 111.0` (a rough km-per-degree-of-latitude
    constant) converts a human-meaningful "1.5 km neighbourhood" into the units DBSCAN
    actually uses. It's an approximation (flat-earth, not haversine) — accurate enough
    at city scale, a documented limitation at larger scale.

12. **Q: Why does `rank_hotspots()` drop DBSCAN's noise points (label `-1`)?**
    A: Noise points, by DBSCAN's definition, don't belong to any dense cluster — they're
    isolated accidents, not part of a "black spot." Including them in the ranked
    black-spot table would misrepresent isolated incidents as hotspots.

13. **Q: What's the difference between the SHAP explainer used for tree models versus
    Logistic Regression?**
    A: Tree models (RF/XGBoost/LightGBM) use `shap.TreeExplainer`, which exploits the
    tree structure for fast, exact SHAP value computation. Logistic Regression uses a
    generic `shap.Explainer` wrapping `predict_proba` directly, since it isn't a tree —
    slower, but model-agnostic.

14. **Q: Why is a "background sample" needed for SHAP, rather than the whole dataset?**
    A: SHAP values are computed relative to a baseline ("what would the prediction be
    with no information") approximated from a representative sample. Using the full
    dataset would be more accurate but far slower; a few hundred rows is enough to
    get a stable estimate for a project at this scope.

15. **Q: How does LIME fundamentally differ from SHAP in *how* it explains a
    prediction?**
    A: SHAP is based on cooperative game theory (Shapley values) applied to the model's
    actual internal structure or its input/output behaviour. LIME instead perturbs the
    input many times, observes how the model's output changes, and fits a simple local
    linear model to approximate the real model's behaviour *just around that one
    point* — a genuinely different method, which is why running both provides a
    meaningful cross-check rather than a redundant second opinion.

16. **Q: Why does the dashboard use `@st.cache_resource`?**
    A: So the trained pipelines, background sample, and explainers are loaded from disk
    only once per running dashboard process (not re-loaded on every single user
    interaction/page rerun, which is Streamlit's default behaviour) — this is what
    keeps the dashboard responsive.

17. **Q: What would happen if you removed the `weather_severity_index` engineered
    feature and just left `weather`, `light_condition`, `road_surface` as separate
    one-hot columns?**
    A: The model could probably still learn similar patterns (the one-hot columns carry
    the same underlying information) — but the composite index gives a single, human
    interpretable "how nasty are conditions right now" number that's easier for a
    non-technical stakeholder (and the SHAP/LIME explanations) to reason about directly,
    and reduces the number of separate categorical levels the model has to combine on
    its own.

18. **Q: Why does `main.py` check `if not config.DATA_RAW_PATH.exists()` before
    generating synthetic data, rather than always regenerating it?**
    A: So that once you've either generated data once, or dropped in a real dataset, the
    pipeline never silently overwrites it on a subsequent run — regeneration is opt-in
    (delete the file yourself) rather than automatic.

19. **Q: What's stored in `models/metrics.json`, and who reads it?**
    A: Every severity model's benchmark metrics (accuracy, macro-F1, ROC-AUC, Fatal
    recall/precision/F1, confusion matrix), the winning model's name, its 5-fold CV
    Fatal-recall scores, and the hotspot clustering quality numbers. It's read by the
    dashboard's "Model Performance" tab to render the benchmark table without needing
    to re-run any model.

20. **Q: Why does `train_and_evaluate()` return `y_test` mapped back to string labels,
    even though training happened on integer-encoded labels internally?**
    A: So that any caller downstream (the dashboard, `main.py`'s printed summaries,
    tests) can work with human-readable `"Minor"/"Serious"/"Fatal"` values without
    needing to know about the internal integer encoding at all — the encoding is an
    implementation detail of `severity_model.py`, not something other modules should
    have to care about.

---

## 10 Advanced Questions

1. **Q: The four severity models have similar ROC-AUC (~0.58–0.61) but wildly different
   Fatal recall (0.00–0.53). What does that gap actually tell you about the models?**
   A: ROC-AUC measures ranking ability across *all* possible decision thresholds —
   it's threshold-independent. Fatal recall (at the default argmax threshold) measures
   one specific operating point. A similar ROC-AUC with very different recall means all
   four models learned similar *relative* risk ordering, but the tree ensembles'
   probability outputs are miscalibrated at the specific 1/3-way decision boundary
   argmax uses for 3-class problems — their Fatal probabilities rarely exceed the other
   two classes' probabilities even when they're relatively elevated. This is exactly
   why `apply_fatal_threshold()` exists: it moves the operating point to somewhere the
   models' ROC-AUC says is actually informative, rather than trusting the default.

2. **Q: If you retrained with a class-weighted loss instead of SMOTE for XGBoost/
   LightGBM, would you expect the Fatal-recall gap to close? Why or why not?**
   A: Plausibly yes, and it's a reasonable next experiment (flagged in "What could be
   improved?"). Class weighting penalizes misclassifying the minority class directly in
   the loss function, without synthesizing new interpolated points in feature space —
   it sidesteps the specific failure mode hypothesized here (tree ensembles overfitting
   to SMOTE's synthetic minority points). It's not guaranteed, since tree ensembles can
   still find sharp, high-confidence splits under class weighting too — but it removes
   one plausible cause and is worth testing empirically rather than assumed.

3. **Q: Why is `TreeExplainer`'s output shape different for a 3-class problem than a
   binary one, and how does `global_feature_importance()` handle that?**
   A: For multi-class problems, SHAP returns either a list of per-class arrays or a
   single array with an extra class dimension (`(n_samples, n_features, n_classes)`),
   depending on the SHAP version/explainer type — `global_feature_importance()`
   explicitly checks both shapes (`isinstance(shap_values, list)` vs.
   `np.ndim(shap_values) == 3`) and extracts the requested class's values either way,
   rather than assuming one fixed shape.

4. **Q: The risk score's smoothing constant `k=8` is a fixed value in `config.py`. How
   would you go about choosing it more rigorously?**
   A: `k` represents how many "virtual" observations of the global average you're
   effectively adding as a prior. A more rigorous approach: treat it as a
   hyperparameter, hold out a validation split of segments, and choose the `k` that
   minimizes prediction error (e.g. on a segment-level cross-validation) between the
   smoothed score and a ground-truth signal — or, more principled still, fit an
   empirical-Bayes model (e.g. a Beta-Binomial or Gamma-Poisson conjugate prior) where
   the smoothing strength is derived from the observed between-segment variance rather
   than chosen by hand.

5. **Q: DBSCAN found 53 clusters with 2,213 noise points and a *negative* silhouette
   score (-0.10) on this dataset. Is that a failure?**
   A: Not necessarily — silhouette score penalizes clusters that aren't well-separated
   from each other, but DBSCAN's whole design intentionally allows irregularly-shaped,
   variable-density clusters and large amounts of noise, which silhouette isn't built
   to reward. A negative score here reflects that a lot of the underlying data is
   genuinely diffuse background scatter around city centers (by design in the synthetic
   generator) rather than tightly clustered — DBSCAN correctly labels most of that as
   noise. The more meaningful validation for this project's actual goal is whether the
   top-ranked clusters by `severity_weighted_score` correspond to real danger
   concentrations (they do, by construction, correspond to the deliberately injected
   black-spot junctions) — silhouette is a generic clustering-quality metric, not a
   business-relevant one here.

6. **Q: Why does `build_risk_lookup()` use `include_groups=False` in its
   `.groupby().apply()` call?**
   A: To avoid pandas re-including the grouping columns inside the group's DataFrame
   passed to `_severity_weighted_score()`, which changed behaviour across recent pandas
   versions (older pandas silently included them; the parameter makes the exclusion
   explicit rather than relying on default behaviour that has been in flux).

7. **Q: The project separates "severity" (per-accident) and "risk" (per-segment,
   exposure-based). What's the fundamental statistical difference between what each one
   estimates?**
   A: Severity prediction estimates `P(severity class | accident occurred, its
   conditions)` — it's conditioned on an accident already having happened. Risk
   prediction, as implemented, estimates something closer to
   `E[severity weight | accident occurred in this segment]` — still conditioned on an
   accident happening, just aggregated to segment level instead of per-instance. A
   *true* exposure-based risk score would need `P(severe accident | segment, time
   window)` unconditioned on an accident happening — which requires a
   traffic-volume/exposure denominator this project doesn't have (see "What could be
   improved?"). Being able to state this distinction precisely is a strong signal of
   understanding the project's actual scope and its honest limitations.

8. **Q: Why might Precision-Recall AUC be a more appropriate primary metric than
   ROC-AUC for the Fatal class specifically, and why doesn't this project report it?**
   A: Under severe class imbalance, ROC-AUC can look deceptively good because the
   true-negative rate is easy to keep high (there are so many negatives) — PR-AUC
   focuses only on the positive (Fatal) class's precision/recall trade-off and is
   generally considered more informative for rare-class problems (the project's own SRS
   document actually names PR-AUC as "the primary comparison metric given class
   imbalance" in section 3.7). This implementation reports ROC-AUC and per-class recall/
   precision/F1 instead — a reasonable, defensible choice for the same underlying goal,
   but adding PR-AUC explicitly (via `sklearn.metrics.average_precision_score`) would be
   a direct, well-scoped improvement.

9. **Q: The synthetic data generator injects black-spot junctions using a fixed random
   seed. What's a concrete way to check whether the hotspot pipeline is actually finding
   those specific injected black-spots, rather than just finding *some* dense-looking
   region?**
   A: Compare `rank_hotspots()`'s output centroids against the known
   `_build_blackspot_junctions()` coordinates (both are deterministic given the same
   seed) — e.g. compute the distance from each top-ranked cluster centroid to the
   nearest injected black-spot coordinate, and check it's within a small tolerance
   (a few hundred metres). This is a genuinely stronger validation than silhouette
   score, because it checks the pipeline against *known ground truth* rather than a
   generic clustering-quality heuristic — and would make a good addition to
   `tests/test_models.py`.

10. **Q: If this were re-architected as the SRS's larger multi-user system, what's the
    single biggest structural change required, beyond just "add a database and a web
    framework"?**
    A: The current pipeline assumes a **single, static dataset and a single set of
    trained models shared by everyone** (`main.py` runs once, produces one
    `models/` directory). A real multi-user system needs **per-dataset, versioned
    model artifacts** (SRS FR-3.4-05, NFR-5.5-02: "support model versioning") — meaning
    `config.py`'s flat `MODEL_DIR` constant would need to become a per-user/per-dataset
    namespace, every training run would need to record which dataset and
    hyperparameters produced it, and the dashboard would need to know *which* model
    version to load for *which* user's request rather than assuming one global answer.
    That's a genuinely different data-modelling problem, not just "add auth on top."
