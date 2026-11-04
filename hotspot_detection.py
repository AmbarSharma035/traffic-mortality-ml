"""
Pillar 3 - Accident Hotspot Detection.

"Independent of any single cause, where on the map do crashes keep
concentrating?" (slide 7). Unsupervised, spatial-only - no severity label
or time window is used to form the clusters themselves; severity is used
afterwards only to *rank* the clusters that were found.

Three complementary methods, matching the synopsis:
  - K-Means  : broad zonal clusters (fixed k, roughly circular)
  - DBSCAN   : irregularly-shaped danger clusters that don't fit a circle,
               plus explicit noise/outlier detection
  - KDE      : a continuous risk-density surface for the map's heatmap layer
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity

import config

KM_PER_DEGREE_LAT = 111.0  # rough constant used to convert DBSCAN's eps from km to degrees


def run_kmeans(df: pd.DataFrame, k: int = None):
    k = k or config.KMEANS_K
    coords = df[["latitude", "longitude"]].values
    model = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
    labels = model.fit_predict(coords)
    score = silhouette_score(coords, labels) if k > 1 else None
    return labels, model, score


def run_dbscan(df: pd.DataFrame, eps_km: float = None, min_samples: int = None):
    """DBSCAN in (approximate) km via a simple degrees-per-km conversion.
    Good enough at city scale; a production system would use a proper
    haversine metric instead of this flat-earth approximation."""
    eps_km = eps_km or config.DBSCAN_EPS_KM
    min_samples = min_samples or config.DBSCAN_MIN_SAMPLES
    eps_deg = eps_km / KM_PER_DEGREE_LAT
    coords = df[["latitude", "longitude"]].values
    model = DBSCAN(eps=eps_deg, min_samples=min_samples)
    labels = model.fit_predict(coords)

    non_noise = labels != -1
    score = None
    if non_noise.sum() > 1 and len(set(labels[non_noise])) > 1:
        score = silhouette_score(coords[non_noise], labels[non_noise])
    n_noise = int((labels == -1).sum())
    return labels, model, score, n_noise


def fit_kde(df: pd.DataFrame, bandwidth: float = None) -> KernelDensity:
    """Fit a Gaussian KDE over accident coordinates for the density-surface
    heatmap layer shown on the dashboard map."""
    bandwidth = bandwidth or config.KDE_BANDWIDTH
    coords = df[["latitude", "longitude"]].values
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(coords)
    return kde


def kde_grid(kde: KernelDensity, df: pd.DataFrame, grid_size: int = 60):
    """Evaluate the fitted KDE on a regular lat/lon grid spanning the data,
    for drawing a density heatmap. Returns (lat_grid, lon_grid, density)."""
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
    lat_grid = np.linspace(lat_min, lat_max, grid_size)
    lon_grid = np.linspace(lon_min, lon_max, grid_size)
    mesh_lat, mesh_lon = np.meshgrid(lat_grid, lon_grid)
    grid_points = np.column_stack([mesh_lat.ravel(), mesh_lon.ravel()])
    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density).reshape(mesh_lat.shape)
    return mesh_lat, mesh_lon, density


def rank_hotspots(df: pd.DataFrame, cluster_labels: np.ndarray, top_n: int = None) -> pd.DataFrame:
    """Turn raw cluster assignments into the ranked "black-spot" table
    (SRS FR-3.6-04): ordered by a severity-weighted score combining crash
    density and how many of those crashes were Fatal/Serious.
    """
    top_n = top_n or config.TOP_N_HOTSPOTS
    work = df.copy()
    work["cluster"] = cluster_labels
    work = work[work["cluster"] != -1]  # drop DBSCAN noise points, if any

    weights = work["severity"].map(config.RISK_SEVERITY_WEIGHTS)
    work = work.assign(_weight=weights)

    grouped = work.groupby("cluster").agg(
        n_accidents=("cluster", "size"),
        fatal_count=("severity", lambda s: (s == "Fatal").sum()),
        serious_count=("severity", lambda s: (s == "Serious").sum()),
        centroid_lat=("latitude", "mean"),
        centroid_lon=("longitude", "mean"),
        dominant_city=("city", lambda s: s.mode().iloc[0]),
        severity_weighted_score=("_weight", "sum"),
    ).reset_index()

    grouped = grouped.sort_values("severity_weighted_score", ascending=False).head(top_n)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return grouped.reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv(config.DATA_PROCESSED_PATH, parse_dates=["date"])

    km_labels, km_model, km_score = run_kmeans(df)
    print(f"K-Means (k={config.KMEANS_K}): silhouette={km_score:.3f}")

    db_labels, db_model, db_score, n_noise = run_dbscan(df)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points, silhouette={db_score}")

    ranked = rank_hotspots(df, db_labels)
    print(ranked.head(10)[["rank", "cluster", "dominant_city", "n_accidents", "fatal_count", "severity_weighted_score"]])
