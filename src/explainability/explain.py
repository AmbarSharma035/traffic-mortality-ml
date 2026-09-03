import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (MODELS_DIR, DATA_PROCESSED, FIGURES_DIR,
                            setup_logging, get_config_section, ensure_dirs)

logger = setup_logging("explainability")

def shap_global_explanation(model, X_sample, feature_names, model_name) -> list:
    logger.info("Running SHAP global explanation...")
    try:
        import shap
        if any(name in model_name.lower() for name in ['forest', 'xgb', 'lgb', 'tree']):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_sample, 50))
            shap_values = explainer.shap_values(X_sample)
            
        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list):
            sv = shap_values[-1]
        else:
            sv = shap_values
            
        shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False)
        plt.savefig(FIGURES_DIR / "shap_summary.png", bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_sample, feature_names=feature_names, plot_type="bar", show=False)
        plt.savefig(FIGURES_DIR / "shap_feature_importance.png", bbox_inches='tight')
        plt.close()
        
        return shap_values
    except Exception as e:
        logger.error(f"Error in SHAP global: {e}")
        return None

def shap_individual_explanation(model, instance, feature_names, model_name, index=0) -> dict:
    logger.info(f"Running SHAP individual explanation for index {index}...")
    try:
        import shap
        if any(name in model_name.lower() for name in ['forest', 'xgb', 'lgb', 'tree']):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(instance)
            expected_value = explainer.expected_value
        else:
            return {}
            
        if isinstance(shap_values, list):
            sv = shap_values[-1][0]
            ev = expected_value[-1]
        else:
            sv = shap_values[0]
            ev = expected_value
            
        exp = shap.Explanation(values=sv, base_values=ev, data=instance.values[0], feature_names=feature_names)
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(exp, show=False)
        plt.savefig(FIGURES_DIR / f"shap_individual_{index}.png", bbox_inches='tight')
        plt.close()
        
        contributions = {feature_names[i]: float(sv[i]) for i in range(len(feature_names))}
        return contributions
    except Exception as e:
        logger.error(f"Error in SHAP individual: {e}")
        return {}

def lime_individual_explanation(model, instance, X_train, feature_names, class_names, index=0) -> dict:
    logger.info(f"Running LIME individual explanation for index {index}...")
    try:
        import lime
        import lime.lime_tabular
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )
        
        exp = explainer.explain_instance(
            instance.values[0],
            model.predict_proba,
            num_features=10
        )
        
        fig = exp.as_pyplot_figure()
        fig.savefig(FIGURES_DIR / f"lime_individual_{index}.png", bbox_inches='tight')
        plt.close(fig)
        
        pred_probs = model.predict_proba(instance)[0]
        pred_class = class_names[np.argmax(pred_probs)]
        
        return {
            'predicted_class': pred_class,
            'class_probabilities': pred_probs.tolist(),
            'feature_contributions': exp.as_list()
        }
    except Exception as e:
        logger.error(f"Error in LIME individual: {e}")
        return {}

def main():
    ensure_dirs()
    try:
        # Load model artifacts using same paths as train.py
        model_path = MODELS_DIR / "selected_model.pkl"
        scaler_path = MODELS_DIR / "scaler.pkl"
        feature_names_path = MODELS_DIR / "feature_names.json"
        
        if not model_path.exists():
            logger.error("Selected model not found. Run training first.")
            return
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
            
        # Load feature matrix (saved by build_features.py)
        X_df = pd.read_csv(DATA_PROCESSED / "feature_matrix.csv").head(1000)
        
        X_scaled_np = scaler.transform(X_df)
        X_scaled = pd.DataFrame(X_scaled_np, columns=feature_names)
        
        exp_config = get_config_section('explainability')
        max_samples = exp_config.get('shap_max_samples', 100)
        X_sample = X_scaled.sample(min(max_samples, len(X_scaled)), random_state=42)
        
        # Determine model name from filename
        model_name = "selected_model"
        shap_global_explanation(model, X_sample, feature_names, model_name)
        
        class_names = ["Minor", "Serious", "Fatal"]
        for i in range(3):
            if i >= len(X_scaled): break
            instance = X_scaled.iloc[[i]]
            shap_individual_explanation(model, instance, feature_names, model_name, i)
            lime_individual_explanation(model, instance, X_scaled, feature_names, class_names, i)
            
        logger.info("Explainability main completed.")
    except Exception as e:
        logger.exception("Error in explain main")

if __name__ == '__main__':
    main()
