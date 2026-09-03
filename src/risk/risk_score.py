import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, FIGURES_DIR,
                            setup_logging, get_config_section, ensure_dirs)

logger = setup_logging("risk_score")

def compute_severity_score(fatal_probability: float) -> float:
    return float(fatal_probability * 100)

def compute_temporal_score(hour: int, day_of_week: int, temporal_stats: dict) -> float:
    if not temporal_stats or 'hourly' not in temporal_stats or 'daily' not in temporal_stats:
        return 50.0
    hour_risk = temporal_stats['hourly'].get(str(hour), 0.5)
    day_risk = temporal_stats['daily'].get(str(day_of_week), 0.5)
    return (hour_risk * 0.6 + day_risk * 0.4) * 100

def compute_environmental_score(weather_severity: int, visibility_cat: str, is_night: bool, road_surface: str = None) -> float:
    score = 0.0
    if weather_severity > 2: score += 30
    elif weather_severity > 0: score += 10
    
    if visibility_cat in ['Poor', 'Bad']: score += 30
    if is_night: score += 20
    if road_surface in ['Wet', 'Snow', 'Ice']: score += 20
    
    return min(100.0, score)

def compute_geographic_score(lat: float, lon: float, kde_model=None) -> float:
    if kde_model is None:
        return 50.0
    try:
        density = kde_model([lon, lat])[0]
        return min(100.0, float(density * 100))
    except Exception:
        return 50.0

def compute_risk_score(severity_score, temporal_score, env_score, geo_score) -> float:
    risk_config = get_config_section('risk')
    w = risk_config.get('weights', {'severity': 0.4, 'temporal': 0.2, 'environmental': 0.2, 'geographic': 0.2})
    
    score = (w.get('severity', 0.4) * severity_score + 
             w.get('temporal', 0.2) * temporal_score + 
             w.get('environmental', 0.2) * env_score + 
             w.get('geographic', 0.2) * geo_score)
    return max(0.0, min(100.0, float(score)))

def get_risk_category(score: float) -> str:
    risk_config = get_config_section('risk')
    thresholds = risk_config.get('thresholds', {'Low': 30, 'Moderate': 60, 'High': 80})
    if score <= thresholds.get('Low', 30):
        return 'Low'
    elif score <= thresholds.get('Moderate', 60):
        return 'Moderate'
    elif score <= thresholds.get('High', 80):
        return 'High'
    else:
        return 'Critical'

def compute_risk_for_dataset(df, fatal_probs, kde_model=None) -> pd.DataFrame:
    """Compute risk scores for all records (vectorized)."""
    logger.info("Computing risk scores for dataset...")
    df = df.copy()
    
    # Severity scores from model probabilities
    sev_scores = fatal_probs * 100
    
    # Temporal scores (simplified - use neutral 50 without per-location stats)
    temp_scores = np.full(len(df), 50.0)
    
    # Environmental scores
    env_scores = np.zeros(len(df))
    if 'weather_severity' in df.columns:
        env_scores += np.where(df['weather_severity'] > 2, 30, 
                              np.where(df['weather_severity'] > 0, 10, 0))
    if 'is_night' in df.columns:
        env_scores += np.where(df['is_night'] == 1, 20, 0)
    env_scores = np.clip(env_scores, 0, 100)
    
    # Geographic scores (neutral without KDE)
    geo_scores = np.full(len(df), 50.0)
    
    # Combine
    risk_config = get_config_section('risk')
    w = risk_config.get('weights', {'severity': 0.4, 'temporal': 0.2, 'environmental': 0.2, 'geographic': 0.2})
    
    risk_scores = (w.get('severity', 0.4) * sev_scores + 
                   w.get('temporal', 0.2) * temp_scores + 
                   w.get('environmental', 0.2) * env_scores + 
                   w.get('geographic', 0.2) * geo_scores)
    risk_scores = np.clip(risk_scores, 0, 100)
    
    df['risk_score'] = risk_scores
    df['risk_category'] = pd.cut(risk_scores, 
                                  bins=[-1, 30, 60, 80, 101], 
                                  labels=['Low', 'Moderate', 'High', 'Critical'])
    
    return df

def main():
    logger.info("Starting risk score computation...")
    ensure_dirs()
    
    try:
        import json
        
        # Load model artifacts
        model_path = MODELS_DIR / "selected_model.pkl"
        scaler_path = MODELS_DIR / "scaler.pkl"
        feature_names_path = MODELS_DIR / "feature_names.json"
        
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}. Run training first.")
            return
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
        
        # Load unified dataset and feature matrix
        df = pd.read_csv(DATA_PROCESSED / "unified_accidents.csv")
        X = pd.read_csv(DATA_PROCESSED / "feature_matrix.csv")
        
        # Scale features and get probabilities
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)
        fatal_probs = probs[:, -1]  # Fatal class probability
        
        # Merge feature columns needed for risk scoring
        for col in ['weather_severity', 'is_night']:
            if col in X.columns and col not in df.columns:
                df[col] = X[col].values
        
        # Compute risk scores
        augmented_df = compute_risk_for_dataset(df, fatal_probs)
        
        out_path = DATA_PROCESSED / "accidents_with_risk.csv"
        augmented_df.to_csv(out_path, index=False)
        logger.info(f"Saved augmented dataset to {out_path}")
        
        plt.figure(figsize=(10, 6))
        plt.hist(augmented_df['risk_score'], bins=50, color='skyblue', edgecolor='black')
        plt.title('Distribution of Risk Scores')
        plt.xlabel('Risk Score')
        plt.ylabel('Frequency')
        fig_path = FIGURES_DIR / "risk_distribution.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
        logger.info("Summary stats for risk scores:")
        logger.info(str(augmented_df['risk_score'].describe()))
        logger.info(str(augmented_df['risk_category'].value_counts()))
        
    except Exception as e:
        logger.exception("Error in risk score main")

if __name__ == '__main__':
    main()

