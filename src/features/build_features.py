"""Feature engineering and matrix generation."""
import sys
import os
import json
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (DATA_PROCESSED, UNIFIED_DATA_PATH, FEATURE_MATRIX_PATH,
                            FEATURE_COLUMNS_PATH, TARGET_PATH, FIGURES_DIR,
                            setup_logging, get_config_section, ensure_dirs)

logger = setup_logging(__name__)

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features like rush hour, time of day, season."""
    df = df.copy()
    
    # rush_hour
    rush_hours = [7, 8, 9, 16, 17, 18]
    df['rush_hour'] = df['hour'].isin(rush_hours).astype(int)
    
    # time_of_day
    def get_time_of_day(h):
        if pd.isna(h): return 'unknown'
        if 6 <= h <= 11: return 'morning'
        if 12 <= h <= 16: return 'afternoon'
        if 17 <= h <= 20: return 'evening'
        return 'night'
    df['time_of_day'] = df['hour'].apply(get_time_of_day)
    
    # season (12,1,2: Winter, 3,4,5: Spring, 6,7,8: Summer, 9,10,11: Fall)
    def get_season(m):
        if pd.isna(m): return 'unknown'
        if m in [12, 1, 2]: return 'winter'
        if m in [3, 4, 5]: return 'spring'
        if m in [6, 7, 8]: return 'summer'
        return 'fall'
    df['season'] = df['month'].apply(get_season)
    
    return df

def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create weather related features."""
    df = df.copy()
    
    # weather_severity (0-3)
    def map_weather(w):
        if pd.isna(w): return 0
        w = str(w).lower()
        if any(x in w for x in ['severe', 'storm', 'tornado', 'hurricane', 'heavy snow', 'heavy rain']):
            return 3
        if any(x in w for x in ['rain', 'fog', 'snow', 'hail', 'mist', 'precipitation']):
            return 2
        if any(x in w for x in ['cloud', 'overcast', 'light rain', 'drizzle', 'scattered']):
            return 1
        return 0
    
    if 'weather_condition' in df.columns:
        df['weather_severity'] = df['weather_condition'].apply(map_weather)
    else:
        df['weather_severity'] = 0
        
    # visibility_category
    def map_visibility(v):
        if pd.isna(v): return 'unknown'
        if v > 5: return 'good'
        if v >= 2: return 'moderate'
        return 'poor'
        
    if 'visibility' in df.columns:
        df['visibility_category'] = df['visibility'].apply(map_visibility)
    else:
        df['visibility_category'] = 'unknown'
        
    # temperature_category
    def map_temp(t):
        if pd.isna(t): return 'unknown'
        if t < 32: return 'cold'
        if t <= 59: return 'cool'
        if t <= 79: return 'moderate'
        return 'hot'
        
    if 'temperature' in df.columns:
        df['temperature_category'] = df['temperature'].apply(map_temp)
    else:
        df['temperature_category'] = 'unknown'
        
    return df

def create_road_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create road infrastructure features."""
    df = df.copy()
    
    road_cols = []
    if 'has_junction' in df.columns: road_cols.append('has_junction')
    if 'has_crossing' in df.columns: road_cols.append('has_crossing')
    if 'has_traffic_signal' in df.columns: road_cols.append('has_traffic_signal')
    
    if road_cols:
        df['has_road_feature'] = df[road_cols].any(axis=1).astype(int)
    else:
        df['has_road_feature'] = 0
        
    return df

def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Build the final feature matrix and target vector."""
    logger.info("Building feature matrix.")
    
    df = create_temporal_features(df)
    df = create_weather_features(df)
    df = create_road_features(df)
    
    # Encode Categoricals
    # ordinal encoders mapped manually or factorized
    time_of_day_map = {'unknown': -1, 'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
    season_map = {'unknown': -1, 'spring': 0, 'summer': 1, 'fall': 2, 'winter': 3}
    vis_map = {'unknown': -1, 'poor': 0, 'moderate': 1, 'good': 2}
    temp_map = {'unknown': -1, 'cold': 0, 'cool': 1, 'moderate': 2, 'hot': 3}
    
    df['time_of_day_enc'] = df['time_of_day'].map(time_of_day_map)
    df['season_enc'] = df['season'].map(season_map)
    df['visibility_category_enc'] = df['visibility_category'].map(vis_map)
    df['temperature_category_enc'] = df['temperature_category'].map(temp_map)
    
    if 'lighting_condition' in df.columns:
        df['lighting_condition_enc'] = pd.factorize(df['lighting_condition'])[0]
    else:
        df['lighting_condition_enc'] = -1
        
    if 'weather_condition' in df.columns:
        df['weather_condition_enc'] = pd.factorize(df['weather_condition'])[0]
    else:
        df['weather_condition_enc'] = -1
        
    # Target Encoding
    severity_map = {'Minor': 0, 'Serious': 1, 'Fatal': 2}
    df['severity_enc'] = df['severity'].map(severity_map)
    # Drop rows without valid severity
    df = df.dropna(subset=['severity_enc'])
    
    y = df['severity_enc'].astype(int)
    
    # Feature columns
    base_features = [
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_night', 
        'temperature', 'humidity', 'visibility', 'wind_speed',
        'has_junction', 'has_crossing', 'has_traffic_signal',
        'rush_hour', 'weather_severity', 'has_road_feature',
        'time_of_day_enc', 'season_enc', 'visibility_category_enc', 
        'temperature_category_enc', 'lighting_condition_enc', 'weather_condition_enc'
    ]
    
    features = [f for f in base_features if f in df.columns]
    X = df[features].copy()
    
    # Drop columns that are entirely NaN (e.g., temperature when only UK data)
    all_nan_cols = [col for col in X.columns if X[col].isnull().all()]
    if all_nan_cols:
        logger.info(f"Dropping entirely-NaN columns: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)
    
    # Handle remaining NaNs: fill with median, fallback to 0
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            fill_val = median_val if pd.notna(median_val) else 0
            X[col] = X[col].fillna(fill_val)
            
    # Ensure boolean columns are numeric
    for col in X.select_dtypes(include=bool).columns:
        X[col] = X[col].astype(int)
    
    # Ensure all values are finite
    X = X.fillna(0)
        
    feature_names = list(X.columns)
    
    return X, y, feature_names

def main():
    ensure_dirs()
    logger.info(f"Loading unified dataset from {UNIFIED_DATA_PATH}")
    
    if not os.path.exists(UNIFIED_DATA_PATH):
        logger.error(f"Unified data not found at {UNIFIED_DATA_PATH}")
        return
        
    df = pd.read_csv(UNIFIED_DATA_PATH)
    
    X, y, feature_names = build_feature_matrix(df)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Feature names: {feature_names}")
    
    # Save outputs
    X.to_csv(FEATURE_MATRIX_PATH, index=False)
    y.to_csv(TARGET_PATH, index=False)
    
    with open(FEATURE_COLUMNS_PATH, 'w') as f:
        json.dump(feature_names, f, indent=4)
        
    # Save coords
    coords_path = Path(DATA_PROCESSED) / "coordinates.csv"
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df[['latitude', 'longitude']].to_csv(coords_path, index=False)
        logger.info(f"Saved coordinates to {coords_path}")
        
    logger.info("Feature engineering completed successfully.")

if __name__ == '__main__':
    main()
