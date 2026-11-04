
"""
Pillar 3 - Accident Hotspot Detection.

Three complementary spatial methods:
1. K-Means: broad geographic zones.
2. DBSCAN: dense geographic clusters and noise detection.
3. KDE: continuous accident-density surface.

Clustering uses accident coordinates only.
Severity is used afterwards to rank discovered clusters.

Performance safeguards:
- Silhouette scores use a reproducible sample.
- DBSCAN uses a reproducible spatial sample.
- DBSCAN uses haversine distance in kilometres.
- DBSCAN labels apply only to the records actually clustered.
"""

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity

import config

EARTH_RADIUS_KM = 6371.0088
SILHOUETTE_SAMPLE_SIZE = 5000
DBSCAN_SAMPLE_SIZE = 30000
KMEANS_SAMPLE_SIZE = 100000


def _valid_coordinates(df):
    """Return rows with valid geographic coordinates."""
    coords = df[["latitude", "longitude"]].apply(
        pd.to_numeric, errors="coerce"
    )

    valid = (
        coords["latitude"].between(-90, 90)
        & coords["longitude"].between(-180, 180)
    )

    return coords.loc[valid].to_numpy(dtype=float), valid


def _silhouette_sample(coords, labels):
    """Calculate silhouette on a reproducible sample."""
    if len(set(labels)) < 2:
        return None

    sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(coords))

    if sample_size < 2:
        return None

    try:
        return float(
            silhouette_score(
                coords,
                labels,
                sample_size=sample_size,
                random_state=config.RANDOM_SEED,
            )
        )
    except ValueError:
        return None


def run_kmeans(df, k=None):
    """Fit K-Means on valid coordinates.

    Returns:
        labels: labels aligned to the original dataframe.
        model: fitted K-Means model.
        score: sampled silhouette score.
    """
    k = config.KMEANS_K if k is None else k

    coords, valid = _valid_coordinates(df)

    if len(coords) < k:
        raise ValueError(
            f"K-Means needs at least {k} valid coordinates."
        )

    # Fit on a reproducible sample to control memory and runtime.
    rng = np.random.default_rng(config.RANDOM_SEED)

    sample_size = min(KMEANS_SAMPLE_SIZE, len(coords))

    sample_indices = rng.choice(
        len(coords),
        size=sample_size,
        replace=False,
    )

    model = KMeans(
        n_clusters=k,
        random_state=config.RANDOM_SEED,
        n_init=10,
    )

    model.fit(coords[sample_indices])

    # Assign every valid record to its nearest fitted centroid.
    labels_valid = model.predict(coords)

    score = _silhouette_sample(
        coords[sample_indices],
        model.labels_,
    )

    labels = np.full(len(df), -1, dtype=int)
    labels[np.flatnonzero(valid.to_numpy())] = labels_valid

    return labels, model, score


def run_dbscan(df, eps_km=None, min_samples=None):
    """Run DBSCAN using haversine geographic distance.

    Important:
    DBSCAN is fitted on a reproducible sample to avoid excessive
    memory use. Returned labels are -2 for records not sampled,
    -1 for sampled noise records, and non-negative cluster IDs
    for sampled cluster members.

    Therefore, hotspot ranking must use only sampled records.
    """
    eps_km = (
        config.DBSCAN_EPS_KM if eps_km is None else eps_km
    )

    min_samples = (
        config.DBSCAN_MIN_SAMPLES
        if min_samples is None else min_samples
    )

    coords, valid = _valid_coordinates(df)

    if len(coords) < min_samples:
        raise ValueError(
            "Not enough valid coordinates for DBSCAN."
        )

    rng = np.random.default_rng(config.RANDOM_SEED)

    sample_size = min(DBSCAN_SAMPLE_SIZE, len(coords))

    sample_indices = rng.choice(
        len(coords),
        size=sample_size,
        replace=False,
    )

    sample_coords = coords[sample_indices]

    # Haversine expects [latitude, longitude] in radians.
    radians = np.radians(sample_coords)

    model = DBSCAN(
        eps=eps_km / EARTH_RADIUS_KM,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1,
    )

    sample_labels = model.fit_predict(radians)

    non_noise = sample_labels != -1

    score = None

    if len(set(sample_labels[non_noise])) > 1:
        score = _silhouette_sample(
            radians[non_noise],
            sample_labels[non_noise],
        )

    # -2 means not included in the DBSCAN sample.
    labels = np.full(len(df), -2, dtype=int)

    valid_positions = np.flatnonzero(valid.to_numpy())
    original_positions = valid_positions[sample_indices]

    labels[original_positions] = sample_labels

    n_noise = int((sample_labels == -1).sum())

    return labels, model, score, n_noise


def fit_kde(df, bandwidth=None):
    """Fit KDE using projected approximate kilometre coordinates.

    A local equirectangular projection is used for this UK dataset.
    The bandwidth is expressed in kilometres.
    """
    bandwidth = (
        config.KDE_BANDWIDTH
        if bandwidth is None else bandwidth
    )

    coords, valid = _valid_coordinates(df)

    if len(coords) == 0:
        raise ValueError("No valid coordinates available for KDE.")

    reference_lat = np.mean(coords[:, 0])

    x = (
        coords[:, 1]
        * np.cos(np.radians(reference_lat))
        * 111.320
    )

    y = coords[:, 0] * 110.574

    projected = np.column_stack([x, y])

    kde = KernelDensity(
        bandwidth=bandwidth,
        kernel="gaussian",
    )

    kde.fit(projected)

    return kde


def kde_grid(kde, df, grid_size=60):
    """Evaluate KDE on a geographic grid.

    Returns latitude grid, longitude grid and density.
    """
    coords, valid = _valid_coordinates(df)

    if len(coords) == 0:
        raise ValueError("No valid coordinates available.")

    reference_lat = np.mean(coords[:, 0])

    lat_grid = np.linspace(
        coords[:, 0].min(),
        coords[:, 0].max(),
        grid_size,
    )

    lon_grid = np.linspace(
        coords[:, 1].min(),
        coords[:, 1].max(),
        grid_size,
    )

    mesh_lat, mesh_lon = np.meshgrid(lat_grid, lon_grid)

    x = (
        mesh_lon.ravel()
        * np.cos(np.radians(reference_lat))
        * 111.320
    )

    y = mesh_lat.ravel() * 110.574

    grid_points = np.column_stack([x, y])

    log_density = kde.score_samples(grid_points)

    density = np.exp(log_density).reshape(mesh_lat.shape)

    return mesh_lat, mesh_lon, density


def rank_hotspots(df, cluster_labels, top_n=None):
    """Rank sampled DBSCAN clusters using severity-weighted scores.

    Labels:
    -2: not included in DBSCAN sample
    -1: DBSCAN noise
    0 or higher: discovered cluster
    """
    top_n = (
        config.TOP_N_HOTSPOTS
        if top_n is None else top_n
    )

    if len(cluster_labels) != len(df):
        raise ValueError(
            "Cluster labels must match dataframe length."
        )

    work = df.copy()
    work["cluster"] = cluster_labels

    # Exclude noise and records not included in DBSCAN.
    work = work[work["cluster"] >= 0].copy()

    if work.empty:
        return pd.DataFrame(
            columns=[
                "cluster",
                "n_accidents",
                "fatal_count",
                "serious_count",
                "centroid_lat",
                "centroid_lon",
                "dominant_district",
                "severity_weighted_score",
                "rank",
            ]
        )

    weights = work["severity"].map(
        config.RISK_SEVERITY_WEIGHTS
    )

    if weights.isna().any():
        raise ValueError(
            "Some severity labels have no configured risk weight."
        )

    work["_weight"] = weights

    # The prepared dataset currently stores district names in city.
    district_column = (
        "district" if "district" in work.columns else "city"
    )

    grouped = work.groupby("cluster").agg(
        n_accidents=("cluster", "size"),
        fatal_count=(
            "severity",
            lambda s: (s == "Fatal").sum(),
        ),
        serious_count=(
            "severity",
            lambda s: (s == "Serious").sum(),
        ),
        centroid_lat=("latitude", "mean"),
        centroid_lon=("longitude", "mean"),
        dominant_district=(
            district_column,
            lambda s: (
                s.mode().iloc[0] if not s.mode().empty else "Unknown"
            ),
        ),
        severity_weighted_score=("_weight", "sum"),
    ).reset_index()

    grouped = grouped.sort_values(
        "severity_weighted_score",
        ascending=False,
    ).head(top_n)

    grouped["rank"] = np.arange(1, len(grouped) + 1)

    return grouped.reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv(config.DATA_PROCESSED_PATH)

    print("Running K-Means...")
    km_labels, km_model, km_score = run_kmeans(df)

    print(
        f"K-Means: {config.KMEANS_K} clusters, "
        f"sampled silhouette={km_score}"
    )

    print("\nRunning sampled DBSCAN...")
    db_labels, db_model, db_score, n_noise = run_dbscan(df)

    n_clusters = len(set(db_model.labels_)) - (
        1 if -1 in db_model.labels_ else 0
    )

    print(
        f"DBSCAN: {n_clusters} clusters, "
        f"{n_noise} noise points, "
        f"sampled silhouette={db_score}"
    )

    ranked = rank_hotspots(df, db_labels)

    print("\nTop hotspots:")
    print(ranked.head(10).to_string(index=False))
