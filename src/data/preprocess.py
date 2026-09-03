import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (RAW_UK_STATS19, RAW_US_ACCIDENTS, UNIFIED_DATA_PATH,
                           FIGURES_DIR, setup_logging, ensure_dirs, get_config_section)

logger = setup_logging(__name__)

def preprocess_us_accidents(sample_size=500000) -> pd.DataFrame:
    """Preprocess US Accidents dataset."""
    logger.info("Preprocessing US Accidents dataset.")
    us_dir = Path(RAW_US_ACCIDENTS)
    csv_files = list(us_dir.glob("*.csv"))
    if not csv_files:
        logger.error("US Accidents CSV not found.")
        return pd.DataFrame()
        
    filepath = csv_files[0]
    
    usecols = [
        'ID', 'Severity', 'Start_Time', 'Start_Lat', 'Start_Lng', 
        'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)', 
        'Weather_Condition', 'Sunrise_Sunset', 'Civil_Twilight', 
        'Junction', 'Crossing', 'Traffic_Signal', 'City', 'County', 'State'
    ]
    
    logger.info(f"Loading {filepath} (this may take a while)...")
    try:
        df = pd.read_csv(filepath, usecols=usecols)
    except ValueError as e:
        logger.error(f"Error reading CSV: {e}. Attempting to read without usecols constraint.")
        df = pd.read_csv(filepath)
        missing_cols = [c for c in usecols if c not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in US data: {missing_cols}")
        usecols = [c for c in usecols if c in df.columns]
        df = df[usecols]

    if sample_size is not None and len(df) > sample_size:
        logger.info(f"Sampling {sample_size} records from {len(df)} total.")
        df = df.sample(n=sample_size, random_state=42)
        
    # Temporal features
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed', errors='coerce')
    df['hour'] = df['Start_Time'].dt.hour
    df['day_of_week'] = df['Start_Time'].dt.dayofweek
    df['month'] = df['Start_Time'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    if 'Sunrise_Sunset' in df.columns:
        df['is_night'] = (df['Sunrise_Sunset'] == 'Night').astype(int)
    else:
        df['is_night'] = ((df['hour'] >= 18) | (df['hour'] < 6)).astype(int)
        
    # Severity mapping
    severity_map = {1: 'Minor', 2: 'Minor', 3: 'Serious', 4: 'Fatal'}
    df['Severity'] = df['Severity'].map(severity_map)
    
    # Missing values
    num_cols = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    if 'Weather_Condition' in df.columns:
        df['Weather_Condition'] = df['Weather_Condition'].fillna(df['Weather_Condition'].mode()[0])
        
    bool_cols = ['Junction', 'Crossing', 'Traffic_Signal']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False)

    # Rename
    rename_map = {
        'Start_Lat': 'latitude', 'Start_Lng': 'longitude', 'Start_Time': 'timestamp',
        'Temperature(F)': 'temperature', 'Humidity(%)': 'humidity',
        'Visibility(mi)': 'visibility', 'Wind_Speed(mph)': 'wind_speed',
        'Weather_Condition': 'weather_condition', 'Sunrise_Sunset': 'lighting_condition',
        'Junction': 'has_junction', 'Crossing': 'has_crossing', 'Traffic_Signal': 'has_traffic_signal',
        'State': 'region', 'ID': 'accident_id', 'Severity': 'severity'
    }
    df = df.rename(columns=rename_map)
    df['country'] = 'US'
    
    logger.info(f"Completed US preprocessing. Shape: {df.shape}")
    return df

def preprocess_uk_stats19() -> pd.DataFrame:
    """Preprocess UK STATS19 dataset."""
    logger.info("Preprocessing UK STATS19 dataset.")
    uk_dir = Path(RAW_UK_STATS19)
    collision_file = uk_dir / "dft-road-casualty-statistics-collision-2023.csv"
    
    if not collision_file.exists():
        logger.error(f"UK collision file not found at {collision_file}")
        return pd.DataFrame()
    
    # 2023 format uses 'collision_' prefix instead of 'accident_'
    usecols = [
        'collision_index', 'collision_severity', 'date', 'time', 'latitude', 'longitude',
        'number_of_vehicles', 'number_of_casualties', 'road_surface_conditions',
        'weather_conditions', 'light_conditions', 'junction_detail', 
        'local_authority_district', 'speed_limit'
    ]
    
    try:
        df = pd.read_csv(collision_file, usecols=usecols, low_memory=False)
    except ValueError as e:
        logger.warning(f"Some columns missing: {e}. Reading all columns.")
        df = pd.read_csv(collision_file, low_memory=False)
        usecols = [c for c in usecols if c in df.columns]
        df = df[usecols]
    
    logger.info(f"Loaded UK data: {df.shape[0]} records, {df.shape[1]} columns")

    # Severity mapping: 1=Fatal, 2=Serious, 3=Slight
    severity_map = {1: 'Fatal', 2: 'Serious', 3: 'Minor'}
    df['collision_severity'] = df['collision_severity'].map(severity_map)
    
    # Temporal features
    if 'date' in df.columns and 'time' in df.columns:
        df['time'] = df['time'].fillna('12:00')
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'].astype(str), 
                                          format='mixed', errors='coerce')
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    if 'light_conditions' in df.columns:
        df['is_night'] = df['light_conditions'].isin([4, 5, 6, 7]).astype(int)
    else:
        df['is_night'] = ((df.get('hour', 12) >= 18) | (df.get('hour', 12) < 6)).astype(int)
        
    # Decode weather and light conditions
    weather_map = {1: 'Fine', 2: 'Rain', 3: 'Snow', 4: 'Fine+High Winds', 
                   5: 'Rain+High Winds', 6: 'Snow+High Winds', 7: 'Fog/Mist', 
                   8: 'Other', 9: 'Unknown', -1: 'Unknown'}
    light_map = {1: 'Daylight', 4: 'Darkness: lit', 5: 'Darkness: unlit', 
                 6: 'Darkness: no lighting', 7: 'Darkness: unknown', -1: 'Unknown'}
    
    if 'weather_conditions' in df.columns:
        df['weather_condition'] = df['weather_conditions'].map(weather_map).fillna('Unknown')
    else:
        df['weather_condition'] = 'Unknown'
    if 'light_conditions' in df.columns:
        df['lighting_condition'] = df['light_conditions'].map(light_map).fillna('Unknown')
    else:
        df['lighting_condition'] = 'Unknown'
        
    # Rename to unified schema
    rename_map = {
        'collision_index': 'accident_id',
        'collision_severity': 'severity',
        'road_surface_conditions': 'road_surface',
        'junction_detail': 'junction_type',
        'local_authority_district': 'region'
    }
    df = df.rename(columns=rename_map)
    
    # Road features
    df['has_junction'] = (df['junction_type'].notna() & (df['junction_type'] != 0)).astype(int) if 'junction_type' in df.columns else 0
    df['has_crossing'] = 0
    df['has_traffic_signal'] = 0
    
    # Missing columns for UK (weather numerics not in STATS19)
    df['temperature'] = np.nan
    df['humidity'] = np.nan
    df['visibility'] = np.nan
    df['wind_speed'] = np.nan
    
    df['country'] = 'UK'
    
    # Drop rows with missing severity
    df = df.dropna(subset=['severity'])
    
    logger.info(f"Completed UK preprocessing. Shape: {df.shape}")
    return df

def harmonize_datasets(us_df, uk_df) -> pd.DataFrame:
    """Combine US and UK datasets."""
    logger.info("Harmonizing datasets.")
    
    combined = pd.concat([us_df, uk_df], ignore_index=True)
    
    # Ensure consistent order
    common_cols = [
        'accident_id', 'country', 'region', 'timestamp', 'latitude', 'longitude',
        'severity', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_night',
        'temperature', 'humidity', 'visibility', 'wind_speed', 'weather_condition',
        'lighting_condition', 'has_junction', 'has_crossing', 'has_traffic_signal'
    ]
    
    # Keep columns that exist in the combined dataframe
    final_cols = [c for c in common_cols if c in combined.columns]
    
    combined = combined[final_cols]
    
    logger.info(f"Harmonized shape: {combined.shape}")
    
    # Log counts
    logger.info("\nCounts per country:")
    logger.info(combined['country'].value_counts())
    
    logger.info("\nCounts per severity:")
    logger.info(combined['severity'].value_counts())
    
    return combined

def main():
    ensure_dirs()
    logger.info("Starting preprocessing.")
    
    # Get sample size from config
    data_config = get_config_section('data')
    sample_size = data_config.get('us_accidents_sample_size', 500000)
    
    us_df = preprocess_us_accidents(sample_size=sample_size)
    uk_df = preprocess_uk_stats19()
    
    # Handle case where one or both datasets may be empty
    dfs = [df for df in [us_df, uk_df] if not df.empty]
    if not dfs:
        logger.error("No datasets available for preprocessing. Download data first.")
        return
    
    if len(dfs) == 1:
        logger.warning("Only one dataset available. Proceeding with single dataset.")
        unified_df = dfs[0]
    else:
        unified_df = harmonize_datasets(us_df, uk_df)
    
    logger.info(f"Saving unified dataset to {UNIFIED_DATA_PATH}")
    os.makedirs(Path(UNIFIED_DATA_PATH).parent, exist_ok=True)
    unified_df.to_csv(UNIFIED_DATA_PATH, index=False)
    
    logger.info(f"Final dataset: {unified_df.shape[0]} records, {unified_df.shape[1]} columns")
    logger.info(f"Severity distribution:\n{unified_df['severity'].value_counts()}")
    
    # Generate plot
    import matplotlib
    matplotlib.use('Agg')
    plt.figure(figsize=(8, 5))
    severity_counts = unified_df['severity'].value_counts()
    colors = {'Minor': '#2ecc71', 'Serious': '#f39c12', 'Fatal': '#e74c3c'}
    bar_colors = [colors.get(s, '#3498db') for s in severity_counts.index]
    severity_counts.plot(kind='bar', color=bar_colors)
    plt.title('Severity Class Distribution (Raw)')
    plt.xlabel('Severity')
    plt.ylabel('Count')
    for i, (idx, val) in enumerate(severity_counts.items()):
        plt.text(i, val + val*0.01, f'{val:,}', ha='center', va='bottom', fontsize=9)
    
    fig_path = Path(FIGURES_DIR) / "class_distribution_raw.png"
    os.makedirs(Path(FIGURES_DIR), exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"Saved class distribution figure to {fig_path}")

if __name__ == '__main__':
    main()
