import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (MODELS_DIR, FEATURE_COLUMNS_PATH, setup_logging, get_config_section)

import joblib
import json
import numpy as np
import pandas as pd

logger = setup_logging('predict_pipeline')

def load_selected_model():
    """
    Load selected model, scaler, and feature names.
    """
    trained_dir = MODELS_DIR / 'trained'
    
    try:
        model = joblib.load(MODELS_DIR / 'selected_model.pkl')
        scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
        with open(MODELS_DIR / 'feature_names.json', 'r') as f:
            feature_names = json.load(f)
        return model, scaler, feature_names
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise

def predict_severity(features_dict, model=None, scaler=None, feature_names=None):
    """
    Predict severity for a single instance (dict of features).
    """
    if model is None or scaler is None or feature_names is None:
        model, scaler, feature_names = load_selected_model()
        
    # Ensure correct column order
    df = pd.DataFrame([features_dict])
    
    # Handle missing features by filling with 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
            
    df = df[feature_names]
    
    # Scale features
    scaled_features = scaler.transform(df)
    
    # Predict
    pred_code = int(model.predict(scaled_features)[0])
    probs = model.predict_proba(scaled_features)[0]
    
    severity_map = {0: 'Minor', 1: 'Serious', 2: 'Fatal'}
    
    return {
        'predicted_severity': severity_map[pred_code],
        'probabilities': {
            'Minor': float(probs[0]),
            'Serious': float(probs[1]) if len(probs) > 1 else 0.0,
            'Fatal': float(probs[2]) if len(probs) > 2 else 0.0
        },
        'severity_code': pred_code,
        'confidence': float(max(probs))
    }

def predict_batch(df, model=None, scaler=None, feature_names=None):
    """
    Predict severity for a batch of instances.
    """
    if model is None or scaler is None or feature_names is None:
        model, scaler, feature_names = load_selected_model()
        
    df_pred = df.copy()
    
    # Ensure all features exist
    for col in feature_names:
        if col not in df_pred.columns:
            df_pred[col] = 0
            
    X_input = df_pred[feature_names]
    
    # Scale
    X_scaled = scaler.transform(X_input)
    
    # Predict
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)
    
    severity_map = {0: 'Minor', 1: 'Serious', 2: 'Fatal'}
    
    df_pred['predicted_severity_code'] = preds
    df_pred['predicted_severity'] = df_pred['predicted_severity_code'].map(severity_map)
    df_pred['prob_minor'] = probs[:, 0]
    df_pred['prob_serious'] = probs[:, 1] if probs.shape[1] > 1 else 0.0
    df_pred['prob_fatal'] = probs[:, 2] if probs.shape[1] > 2 else 0.0
    df_pred['confidence'] = probs.max(axis=1)
    
    return df_pred
