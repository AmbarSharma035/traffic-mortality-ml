# Hotspot Detection Methodology

Identifying locations with a high density of severe accidents ("hotspots") is crucial for proactive risk management. We compare three spatial analysis techniques.

## 1. K-Means Clustering
*   **Concept:** Partitions the spatial data (latitude/longitude) into $K$ distinct clusters based on distance to the cluster center (centroid).
*   **Parameters:** Requires specifying $K$ (number of clusters) in advance. We use the **Elbow Method** (plotting within-cluster sum of squares against K) to estimate the optimal number of clusters for a given region.
*   **Pros:** Simple, fast, forces data into neat regional zones.
*   **Cons:** Assumes spherical clusters. Doesn't handle noise well (outlier accidents pull the centroid).

## 2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
*   **Concept:** Groups points that are closely packed together, marking points in low-density regions as outliers (noise).
*   **Parameters:** 
    *   `eps`: The maximum distance between two samples for one to be considered as in the neighborhood of the other. We configure this in kilometers (converted to coordinate distance).
    *   `min_samples`: The number of samples in a neighborhood for a point to be considered a core point.
*   **Pros:** Does NOT require specifying the number of clusters in advance. Can find arbitrarily shaped clusters (e.g., along a winding highway). Explicitly identifies isolated accidents as 'noise' rather than forcing them into a cluster.
*   **Cons:** Struggles if clusters have varying densities.

## 3. KDE (Kernel Density Estimation)
*   **Concept:** Creates a continuous probability density surface over the geographic area.
*   **Parameters:** `bandwidth` (controls the smoothness of the surface).
*   **Pros:** Excellent for visual heatmaps. Doesn't create hard boundaries like clustering.
*   **Cons:** Computationally intensive over large areas.

## Comparison

| Feature | K-Means | DBSCAN | KDE |
| :--- | :--- | :--- | :--- |
| Requires specifying 'K'?| Yes | No | No |
| Handles Outliers/Noise? | Poorly | Excellently | N/A (Smooths them) |
| Cluster Shape | Spherical | Arbitrary (e.g., roads) | Continuous Surface |
| Best Used For | Regional macro-zones | Pinpointing dangerous intersections/stretches | Visual Heatmaps |

## Temporal Hotspot Analysis
Beyond spatial clustering, we analyze hotspots temporally:
*   **Hourly:** Identifying intersections that are only dangerous during rush hour.
*   **Daily:** Identifying routes that are dangerous on weekends (e.g., due to drunk driving).
*   **Monthly:** Identifying regions prone to winter weather accidents.
