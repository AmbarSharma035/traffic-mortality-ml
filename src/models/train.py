import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (DATA_PROCESSED, FEATURE_MATRIX_PATH, FEATURE_COLUMNS_PATH,
                            TARGET_PATH, MODELS_DIR, FIGURES_DIR,
                            setup_logging, get_config_section, ensure_dirs)

import numpy as np
import pandas as pd
import json
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logger = setup_logging('train_pipeline')

def split_data(X, y):
    """
    Stratified train/val/test split: 70/15/15
    """
    logger.info("Splitting data into train/val/test...")
    data_config = get_config_section('data')
    test_size = data_config.get('test_size', 0.15)
    val_size = data_config.get('val_size', 0.15)
    random_seed = data_config.get('random_seed', 42)
    
    # Calculate proportion for first split: test vs train+val
    val_train_prop = 1.0 - test_size
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_seed
    )
    
    # Calculate proportion for second split: val vs train within train+val
    relative_val_size = val_size / val_train_prop
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, stratify=y_temp, random_state=random_seed
    )
    
    logger.info(f"Split sizes -> Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def apply_smote(X_train, y_train):
    """
    Apply SMOTE only to training data.
    """
    logger.info("Applying SMOTE to training data ONLY...")
    smote_config = get_config_section('smote')
    random_seed = get_config_section('data').get('random_seed', 42)
    sampling_strategy = smote_config.get('sampling_strategy', 'auto')
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_seed)
    
    logger.info(f"Class distribution before SMOTE: {np.bincount(y_train)}")
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    logger.info(f"Class distribution after SMOTE: {np.bincount(y_resampled)}")
    
    # Save before/after plot
    ensure_dirs()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    classes = ['Minor', 'Serious', 'Fatal']
    
    counts_before = np.bincount(y_train)
    ax[0].bar(classes, counts_before)
    ax[0].set_title('Class Distribution Before SMOTE')
    ax[0].set_ylabel('Count')
    
    counts_after = np.bincount(y_resampled)
    ax[1].bar(classes, counts_after)
    ax[1].set_title('Class Distribution After SMOTE')
    ax[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'smote_distribution.png')
    plt.close()
    
    return X_resampled, y_resampled

def build_preprocessing_pipeline():
    """
    Build scikit-learn preprocessing pipeline.
    """
    logger.info("Building preprocessing pipeline...")
    scaler = StandardScaler()
    return scaler

def train_all_models(X_train, y_train, feature_names):
    """
    Train 4 models using config hyperparameters.
    """
    logger.info("Training models...")
    models_config = get_config_section('models')
    random_seed = get_config_section('data').get('random_seed', 42)
    
    models = {}
    
    # Logistic Regression
    lr_params = models_config.get('logistic_regression', {}).copy()
    lr = LogisticRegression(random_state=random_seed, **lr_params)
    
    # Random Forest
    rf_params = models_config.get('random_forest', {}).copy()
    rf = RandomForestClassifier(random_state=random_seed, **rf_params)
    
    # XGBoost - sklearn API auto-detects num_class
    xgb_params = models_config.get('xgboost', {}).copy()
    xgb_params.pop('eval_metric', None)  # handle separately
    xgb = XGBClassifier(
        random_state=random_seed, 
        eval_metric='mlogloss',
        **xgb_params
    )
    
    # LightGBM
    lgbm_params = models_config.get('lightgbm', {}).copy()
    lgbm_params.pop('verbose', None)  # handle separately
    lgbm = LGBMClassifier(random_state=random_seed, verbose=-1, **lgbm_params)
    
    models_to_train = {
        'logistic_regression': lr,
        'random_forest': rf,
        'xgboost': xgb,
        'lightgbm': lgbm
    }
    
    trained_models = {}
    for name, model in models_to_train.items():
        logger.info(f"Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        logger.info(f"{name} trained in {end_time - start_time:.2f} seconds.")
        trained_models[name] = model
        
    return trained_models

def select_best_model(models, X_val, y_val):
    """
    Evaluate each model on validation set, select best by Fatal-class recall.
    """
    logger.info("Evaluating models on validation set...")
    best_model_name = None
    best_model = None
    best_fatal_recall = -1.0
    
    for name, model in models.items():
        y_pred = model.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        # Assuming classes are 0: Minor, 1: Serious, 2: Fatal
        recalls = recall_score(y_val, y_pred, average=None)
        fatal_recall = recalls[2] if len(recalls) > 2 else 0
        
        logger.info(f"Model: {name} | Accuracy: {acc:.4f} | Fatal Recall: {fatal_recall:.4f}")
        
        if fatal_recall > best_fatal_recall:
            best_fatal_recall = fatal_recall
            best_model_name = name
            best_model = model
            
    logger.info(f"Selected best model: {best_model_name} with Fatal Recall: {best_fatal_recall:.4f}")
    return best_model_name, best_model

def main():
    ensure_dirs()
    logger.info("Starting training pipeline...")
    
    # Load data
    logger.info("Loading processed data...")
    X = pd.read_csv(FEATURE_MATRIX_PATH).values
    y = pd.read_csv(TARGET_PATH).squeeze().values.astype(int)
    with open(FEATURE_COLUMNS_PATH, 'r') as f:
        feature_names = json.load(f)
        
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Save split indices/data shapes for reproducibility (simplification: saving splits as arrays)
    # Fit scaler on train, transform all
    scaler = build_preprocessing_pipeline()
    logger.info("Fitting scaler on training data and transforming datasets...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE ONLY to scaled X_train
    X_train_smote, y_train_smote = apply_smote(X_train_scaled, y_train)
    
    # Train models
    models = train_all_models(X_train_smote, y_train_smote, feature_names)
    
    # Select best model
    best_name, best_model = select_best_model(models, X_val_scaled, y_val)
    
    # Save models, scaler, feature names
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Saving trained models and artifacts...")
    for name, model in models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")
        
    joblib.dump(best_model, MODELS_DIR / 'selected_model.pkl')
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    
    with open(MODELS_DIR / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f)
        
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
