import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import FIGURES_DIR, ensure_dirs

def plot_class_distribution(y, title, save_path):
    plt.figure(figsize=(8, 6))
    counts = y.value_counts()
    ax = counts.plot(kind='bar', color=['#4C72B0', '#DD8452', '#C44E52'])
    plt.title(title)
    plt.ylabel('Count')
    
    for i, v in enumerate(counts):
        ax.text(i, v + len(y)*0.01, str(v), ha='center')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_confusion_matrix(cm, labels, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_model_comparison(comparison_df, save_path):
    if comparison_df.empty or 'Model' not in comparison_df.columns:
        return
        
    df_melt = comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melt, x='Model', y='Score', hue='Metric')
    plt.title('Model Comparison')
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_fatal_recall_comparison(comparison_df, save_path):
    if comparison_df.empty or 'Model' not in comparison_df.columns or 'Fatal_Recall' not in comparison_df.columns:
        return
        
    plt.figure(figsize=(10, 6))
    sns.barplot(data=comparison_df, x='Fatal_Recall', y='Model', color='darkred')
    plt.title('Comparison of Recall for Fatal Accidents')
    plt.xlabel('Recall')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_risk_distribution(risk_scores, save_path):
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(risk_scores, bins=50, edgecolor='black')
    
    for i, p in enumerate(patches):
        val = p.get_x()
        if val <= 30: p.set_facecolor('#2ecc71')
        elif val <= 60: p.set_facecolor('#f1c40f')
        elif val <= 80: p.set_facecolor('#e67e22')
        else: p.set_facecolor('#e74c3c')
        
    plt.title('Distribution of Risk Scores')
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_temporal_patterns(temporal_stats, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    if 'hourly' in temporal_stats:
        x = list(temporal_stats['hourly'].keys())
        y = list(temporal_stats['hourly'].values())
        axes[0].bar(x, y, color='skyblue')
        axes[0].set_title('Hourly')
        axes[0].tick_params(axis='x', rotation=45)
        
    if 'daily' in temporal_stats:
        x = list(temporal_stats['daily'].keys())
        y = list(temporal_stats['daily'].values())
        axes[1].bar(x, y, color='lightgreen')
        axes[1].set_title('Daily')
        
    if 'monthly' in temporal_stats:
        x = list(temporal_stats['monthly'].keys())
        y = list(temporal_stats['monthly'].values())
        axes[2].bar(x, y, color='salmon')
        axes[2].set_title('Monthly')
        
    if 'weekend' in temporal_stats:
        x = list(temporal_stats['weekend'].keys())
        y = list(temporal_stats['weekend'].values())
        axes[3].bar([str(val) for val in x], y, color='purple')
        axes[3].set_title('Weekend')
        
    plt.tight_layout()
    plt.savefig(save_dir / "temporal_patterns_combined.png", dpi=150)
    plt.close()

def create_risk_map(df_with_risk, save_path):
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        return None
        
    if 'latitude' not in df_with_risk.columns or 'longitude' not in df_with_risk.columns:
        return None
        
    df_plot = df_with_risk.dropna(subset=['latitude', 'longitude']).copy()
    if len(df_plot) > 5000:
        df_plot = df_plot.sample(5000)
        
    center_lat = df_plot['latitude'].median()
    center_lon = df_plot['longitude'].median()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    marker_cluster = MarkerCluster().add_to(m)
    
    def get_color(cat):
        if cat == 'Low': return 'green'
        elif cat == 'Moderate': return 'orange'
        elif cat == 'High': return 'red'
        elif cat == 'Critical': return 'darkred'
        return 'blue'
        
    for _, row in df_plot.iterrows():
        cat = row.get('risk_category', 'Low')
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=get_color(cat),
            fill=True,
            fill_color=get_color(cat),
            fill_opacity=0.7,
            popup=f"Risk: {cat}<br>Score: {row.get('risk_score', 0):.1f}"
        ).add_to(marker_cluster)
        
    m.save(str(save_path))
    return m
