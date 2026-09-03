import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (MODELS_DIR, DATA_PROCESSED, FIGURES_DIR, REPORTS_DIR,
                            FEATURE_MATRIX_PATH, TARGET_PATH, FEATURE_COLUMNS_PATH,
                            setup_logging, get_config_section, ensure_dirs)

import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

logger = setup_logging('evaluate_pipeline')

def evaluate_model(model, X_test, y_test, model_name, label_names):
    """
    Evaluate model and compute detailed metrics.
    """
    logger.info(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    # Precision
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    
    # Recall
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    fatal_recall = rec_per_class[2] if len(rec_per_class) > 2 else 0
    
    # F1
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    clf_report_str = classification_report(y_test, y_pred, target_names=label_names, zero_division=0)
    clf_report_dict = classification_report(y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0)
    
    return {
        'model_name': model_name,
        'accuracy': acc,
        'prec_macro': prec_macro,
        'prec_per_class': prec_per_class,
        'rec_macro': rec_macro,
        'rec_per_class': rec_per_class,
        'fatal_recall': fatal_recall,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'classification_report_str': clf_report_str,
        'classification_report_dict': clf_report_dict
    }

def compare_models(results_dict):
    """
    Create comparison DataFrame.
    """
    rows = []
    for name, res in results_dict.items():
        rows.append({
            'Model': name,
            'Accuracy': res['accuracy'],
            'Macro_Precision': res['prec_macro'],
            'Macro_Recall': res['rec_macro'],
            'Macro_F1': res['f1_macro'],
            'Weighted_F1': res['f1_weighted'],
            'Fatal_Recall': res['fatal_recall']
        })
        
    df = pd.DataFrame(rows)
    df = df.sort_values(by='Fatal_Recall', ascending=False).reset_index(drop=True)
    return df

def generate_evaluation_plots(results_dict, comparison_df, label_names):
    """
    Generate and save evaluation plots.
    """
    ensure_dirs()
    
    # 1. Confusion Matrix Heatmaps
    n_models = len(results_dict)
    fig, axes = plt.subplots(int(np.ceil(n_models/2)), 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (name, res) in enumerate(results_dict.items()):
        if i < len(axes):
            sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                        xticklabels=label_names, yticklabels=label_names, ax=axes[i])
            axes[i].set_title(f'Confusion Matrix: {name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'confusion_matrices.png')
    plt.close()
    
    # 2. Fatal Recall Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=comparison_df, x='Model', y='Fatal_Recall', palette='viridis')
    plt.title('Fatal Class Recall Comparison')
    plt.ylabel('Recall (Fatal)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fatal_recall_comparison.png')
    plt.close()
    
    # 3. Model Comparison across metrics
    metrics_to_plot = ['Accuracy', 'Macro_Precision', 'Macro_Recall', 'Macro_F1', 'Fatal_Recall']
    plot_df = comparison_df.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='Model', y='Score', hue='Metric', palette='Set2')
    plt.title('Model Comparison Across Metrics')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_comparison.png')
    plt.close()

def main():
    logger.info("Starting evaluation pipeline...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data to get test set
    X = pd.read_csv(FEATURE_MATRIX_PATH).values
    y = pd.read_csv(TARGET_PATH).squeeze().values.astype(int)
    
    data_config = get_config_section('data')
    test_size = data_config.get('test_size', 0.15)
    random_seed = data_config.get('random_seed', 42)
    
    # We only need the test set
    _, X_test_raw, _, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_seed
    )
    
    # Load scaler and transform test set
    scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
    X_test = scaler.transform(X_test_raw)
    
    label_names = ["Minor", "Serious", "Fatal"]
    
    models_to_eval = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    results_dict = {}
    
    for name in models_to_eval:
        model_path = MODELS_DIR / f"{name}.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            res = evaluate_model(model, X_test, y_test, name, label_names)
            results_dict[name] = res
        else:
            logger.warning(f"Model {name} not found at {model_path}.")
            
    if not results_dict:
        logger.error("No models found to evaluate.")
        return
        
    comp_df = compare_models(results_dict)
    
    # Generate plots
    generate_evaluation_plots(results_dict, comp_df, label_names)
    
    # Save reports
    comp_df.to_csv(REPORTS_DIR / 'model_comparison.csv', index=False)
    
    with open(REPORTS_DIR / 'evaluation_report.txt', 'w') as f:
        f.write("=== Model Evaluation Report ===\n\n")
        f.write(comp_df.to_string())
        f.write("\n\n")
        
        for name, res in results_dict.items():
            f.write(f"\n--- {name.upper()} ---\n")
            f.write(res['classification_report_str'])
            f.write("\n")
            
    logger.info("Evaluation pipeline completed successfully.")
    print("\nSummary of Models (Sorted by Fatal Recall):")
    print(comp_df[['Model', 'Accuracy', 'Fatal_Recall', 'Macro_F1']])

if __name__ == "__main__":
    main()
