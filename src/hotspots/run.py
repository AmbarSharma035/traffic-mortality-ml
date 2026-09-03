import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import gaussian_kde
import joblib
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (DATA_PROCESSED, OUTPUTS_DIR, FIGURES_DIR, MAPS_DIR, MODELS_DIR,
                            setup_logging, get_config_section, ensure_dirs)

logger = setup_logging("hotspots")

def run_kmeans(coords: np.ndarray, n_clusters: int = 20) -> tuple:
    logger.info(f"Running K-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(coords)
    centers = kmeans.cluster_centers_
    
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(f"K-Means cluster sizes: {dict(zip(unique, counts))}")
    
    return labels, centers, kmeans

def run_dbscan(coords: np.ndarray, eps_km: float = 5.0, min_samples: int = 10) -> tuple:
    eps_deg = eps_km / 111.0
    logger.info(f"Running DBSCAN with eps={eps_deg:.4f} degrees, min_samples={min_samples}...")
    dbscan = DBSCAN(eps=eps_deg, min_samples=min_samples)
    labels = dbscan.fit_predict(coords)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    logger.info(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points.")
    
    return labels, n_clusters, n_noise

def run_kde(coords: np.ndarray, bandwidth: float = 0.01, grid_resolution: int = 200) -> tuple:
    logger.info("Running KDE...")
    kde = gaussian_kde(coords.T, bw_method=bandwidth)
    
    lon_min, lon_max = coords[:, 0].min(), coords[:, 0].max()
    lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()
    
    xx, yy = np.mgrid[lon_min:lon_max:complex(0, grid_resolution), 
                      lat_min:lat_max:complex(0, grid_resolution)]
    
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(xx.shape)
    
    return xx, yy, density, kde

def temporal_analysis(df: pd.DataFrame) -> dict:
    logger.info("Running temporal analysis...")
    results = {}
    
    if 'hour' in df.columns:
        hourly = df['hour'].value_counts().sort_index()
        results['hourly'] = hourly.to_dict()
        plt.figure()
        hourly.plot(kind='bar')
        plt.title('Accidents per Hour')
        plt.savefig(FIGURES_DIR / "temporal_hourly.png")
        plt.close()
        
    if 'day_of_week' in df.columns:
        daily = df['day_of_week'].value_counts().sort_index()
        results['daily'] = daily.to_dict()
        plt.figure()
        daily.plot(kind='bar')
        plt.title('Accidents per Day of Week')
        plt.savefig(FIGURES_DIR / "temporal_daily.png")
        plt.close()
        
    if 'month' in df.columns:
        monthly = df['month'].value_counts().sort_index()
        results['monthly'] = monthly.to_dict()
        plt.figure()
        monthly.plot(kind='bar')
        plt.title('Accidents per Month')
        plt.savefig(FIGURES_DIR / "temporal_monthly.png")
        plt.close()
        
    if 'is_weekend' in df.columns:
        weekend = df['is_weekend'].value_counts()
        results['weekend'] = weekend.to_dict()
        plt.figure()
        weekend.plot(kind='bar')
        plt.title('Weekend vs Weekday')
        plt.savefig(FIGURES_DIR / "temporal_weekend.png")
        plt.close()
        
    if 'is_night' in df.columns:
        night = df['is_night'].value_counts()
        results['night'] = night.to_dict()
        
    if 'hour' in df.columns: results['top_hours'] = hourly.nlargest(3).index.tolist()
    if 'day_of_week' in df.columns: results['top_days'] = daily.nlargest(3).index.tolist()
    if 'month' in df.columns: results['top_months'] = monthly.nlargest(3).index.tolist()
    
    return results

try:
    import folium
    from folium.plugins import HeatMap
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

def create_hotspot_map(coords, kmeans_labels, kmeans_centers, dbscan_labels, kde_density=None):
    logger.info("Creating hotspot map...")
    if not HAS_FOLIUM:
        logger.warning("folium not installed, skipping map creation.")
        return
    
    try:
        center_lat = np.median(coords[:, 1])
        center_lon = np.median(coords[:, 0])
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
        
        kmeans_group = folium.FeatureGroup(name='K-Means Clusters')
        for idx, center in enumerate(kmeans_centers):
            folium.Marker(
                [center[1], center[0]], 
                popup=f"Cluster {idx}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(kmeans_group)
        kmeans_group.add_to(m)
        
        if len(coords) > 0:
            heat_group = folium.FeatureGroup(name='Accident Heatmap')
            heat_data = [[row[1], row[0]] for row in coords[:5000]]
            HeatMap(heat_data, radius=12, blur=8).add_to(heat_group)
            heat_group.add_to(m)
            
        folium.LayerControl().add_to(m)
        map_path = MAPS_DIR / "hotspot_map.html"
        m.save(str(map_path))
        logger.info(f"Hotspot map saved to {map_path}")
    except Exception as e:
        logger.error(f"Error creating map: {e}")

def main():
    ensure_dirs()
    try:
        df_path = DATA_PROCESSED / "unified_accidents.csv"
        df = pd.read_csv(df_path)
        
        if 'longitude' not in df.columns or 'latitude' not in df.columns:
            logger.error("Missing coordinates in data.")
            return
            
        coords = df[['longitude', 'latitude']].dropna().values
        
        config_hotspots = get_config_section('hotspots')
        
        kmeans_cfg = config_hotspots.get('kmeans', {})
        dbscan_cfg = config_hotspots.get('dbscan', {})
        kde_cfg = config_hotspots.get('kde', {})
        
        kmeans_labels, kmeans_centers, kmeans_model = run_kmeans(
            coords, n_clusters=kmeans_cfg.get('n_clusters', 20))
        dbscan_labels, n_clusters, n_noise = run_dbscan(
            coords, eps_km=dbscan_cfg.get('eps_km', 5.0), 
            min_samples=dbscan_cfg.get('min_samples', 10))
        xx, yy, density, kde_model = run_kde(
            coords, bandwidth=kde_cfg.get('bandwidth', 0.01))
        
        temp_stats = temporal_analysis(df)
        # Save to data/processed/ so dashboard can find it
        temporal_stats_path = DATA_PROCESSED / "temporal_stats.json"
        with open(temporal_stats_path, "w") as f:
            json.dump(temp_stats, f)
            
        create_hotspot_map(coords, kmeans_labels, kmeans_centers, dbscan_labels)
        
        # Save KDE density grid (the KDE object itself is not picklable)
        try:
            kde_data = {
                'xx': xx, 'yy': yy, 'density': density,
                'coords_shape': coords.shape
            }
            np.savez(MODELS_DIR / "kde_data.npz", **kde_data)
            logger.info("Saved KDE density grid to kde_data.npz")
        except Exception as e:
            logger.warning(f"Could not save KDE data: {e}")
        
        coords_df = df[['longitude', 'latitude']].dropna().copy()
        coords_df['kmeans_label'] = kmeans_labels
        coords_df['dbscan_label'] = dbscan_labels
        coords_df.to_csv(DATA_PROCESSED / "hotspot_labels.csv", index=False)
        
        logger.info("Hotspot analysis complete.")
    except Exception as e:
        logger.exception("Error in hotspots main")

if __name__ == '__main__':
    main()

