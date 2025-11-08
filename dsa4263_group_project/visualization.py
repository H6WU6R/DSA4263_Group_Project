"""
Visualization Module

This module contains functions for creating visualizations
for EDA, feature importance, and model comparison.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Dict, List


def create_temporal_eda_plots(df: pd.DataFrame, save_path: str, verbose: bool = True) -> str:
    """
    Create comprehensive temporal EDA visualizations.
    
    Args:
        df: Email dataframe with temporal features
        save_path: Directory to save the plot
        verbose: Print progress messages
        
    Returns:
        Path to saved visualization file
    """
    if verbose:
        print("\n[Visualization] Creating temporal EDA plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Temporal Patterns in Email Spam Detection', fontsize=16, fontweight='bold')
    
    # Plot 1: Regional distribution
    ax1 = axes[0, 0]
    region_counts = df['timezone_region'].value_counts()
    ax1.bar(range(len(region_counts)), region_counts.values, color='steelblue', alpha=0.7)
    ax1.set_xticks(range(len(region_counts)))
    ax1.set_xticklabels(region_counts.index, rotation=45, ha='right')
    ax1.set_ylabel('Email Count')
    ax1.set_title('Email Distribution by Region')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Spam rate by region
    ax2 = axes[0, 1]
    region_spam = df.groupby('timezone_region')['label'].mean().sort_values(ascending=False)
    ax2.bar(range(len(region_spam)), region_spam.values, color='coral', alpha=0.7)
    ax2.set_xticks(range(len(region_spam)))
    ax2.set_xticklabels(region_spam.index, rotation=45, ha='right')
    ax2.set_ylabel('Spam Rate')
    ax2.set_title('Spam Rate by Region')
    ax2.axhline(y=df['label'].mean(), color='red', linestyle='--', alpha=0.5, label='Overall')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Hourly patterns
    ax3 = axes[0, 2]
    hour_spam = df.groupby('hour')['label'].mean()
    ax3.plot(hour_spam.index, hour_spam.values, marker='o', linewidth=2, markersize=6)
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Spam Rate')
    ax3.set_title('Spam Rate by Hour')
    ax3.set_xticks(range(0, 24, 3))
    ax3.grid(alpha=0.3)
    
    # Plot 4: Weekday patterns
    ax4 = axes[1, 0]
    weekday_spam = df.groupby('day_of_week')['label'].mean()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax4.bar(range(7), weekday_spam.values, color='lightgreen', alpha=0.7)
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(day_names)
    ax4.set_ylabel('Spam Rate')
    ax4.set_title('Spam Rate by Day of Week')
    ax4.axhline(y=df['label'].mean(), color='red', linestyle='--', alpha=0.5)
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: Sender volume distribution
    ax5 = axes[1, 1]
    sender_stats = df.groupby('sender')['label'].count()
    sender_volumes = sender_stats.value_counts().sort_index()
    ax5.bar(sender_volumes.index[:20], sender_volumes.values[:20], color='purple', alpha=0.6)
    ax5.set_xlabel('Emails per Sender')
    ax5.set_ylabel('Number of Senders')
    ax5.set_title('Sender Volume Distribution (Top 20)')
    ax5.set_yscale('log')
    ax5.grid(alpha=0.3)
    
    # Plot 6: Weekend vs Weekday
    ax6 = axes[1, 2]
    weekend_data = df.groupby('is_weekend')['label'].mean()
    labels = ['Weekday', 'Weekend']
    colors_pie = ['lightblue', 'lightcoral']
    ax6.pie([weekend_data[0], weekend_data[1]], labels=labels, autopct='%1.1f%%',
            colors=colors_pie, startangle=90)
    ax6.set_title('Spam Rate: Weekday vs Weekend')
    
    plt.tight_layout()
    eda_file = os.path.join(save_path, 'temporal_eda_analysis.png')
    plt.savefig(eda_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"  ✓ Saved to: {eda_file}")
    
    return eda_file


def create_feature_importance_plots(
    importance_df: pd.DataFrame,
    save_path: str,
    verbose: bool = True
) -> str:
    """
    Create feature importance visualizations.
    
    Args:
        importance_df: Dataframe with 'feature', 'importance', 'type' columns
        save_path: Directory to save the plot
        verbose: Print progress messages
        
    Returns:
        Path to saved visualization file
    """
    if verbose:
        print("\n[Visualization] Creating feature importance plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Top 20 features
    ax1 = axes[0]
    top_20 = importance_df.head(20)
    colors_map = {
        'temporal': 'steelblue',
        'url_domain': 'mediumseagreen',
        'text_meta': 'mediumpurple',
        'graph': 'coral'
    }
    bar_colors = [colors_map[t] for t in top_20['type']]
    
    bars = ax1.barh(range(len(top_20)), top_20['importance'].values, color=bar_colors, alpha=0.7)
    ax1.set_yticks(range(len(top_20)))
    ax1.set_yticklabels(top_20['feature'].values)
    ax1.set_xlabel('Importance', fontweight='bold')
    ax1.set_title('Top 20 Most Important Features', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='steelblue', label='Temporal'),
        Patch(facecolor='mediumseagreen', label='URL/Domain'),
        Patch(facecolor='mediumpurple', label='Text Meta'),
        Patch(facecolor='coral', label='Graph')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Plot 2: Feature type comparison
    ax2 = axes[1]
    type_importance = importance_df.groupby('type')['importance'].sum()
    colors_pie = ['coral', 'steelblue', 'mediumpurple', 'mediumseagreen']
    ax2.pie(type_importance.values, labels=type_importance.index, autopct='%1.1f%%',
            colors=colors_pie, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('Feature Type Contribution', fontweight='bold')
    
    plt.tight_layout()
    importance_file = os.path.join(save_path, 'feature_importance_analysis.png')
    plt.savefig(importance_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"  ✓ Saved to: {importance_file}")
    
    return importance_file


def create_model_comparison_plots(
    model_results: List[Dict],
    save_path: str,
    verbose: bool = True
) -> str:
    """
    Create model comparison visualizations.
    
    Args:
        model_results: List of model result dictionaries
        save_path: Directory to save the plot
        verbose: Print progress messages
        
    Returns:
        Path to saved visualization file
    """
    if verbose:
        print("\n[Visualization] Creating model comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison Analysis', fontsize=16, fontweight='bold')
    
    models = [r['Model'] for r in model_results]
    accuracies = [r['Accuracy'] for r in model_results]
    f1_scores = [r['F1-Score'] for r in model_results]
    precisions = [r['Precision'] for r in model_results]
    recalls = [r['Recall'] for r in model_results]
    times = [r['Training Time (s)'] for r in model_results]
    
    colors_acc = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    
    # Plot 1: Accuracy Comparison
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(models)), accuracies, color=colors_acc, alpha=0.7)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Model')
    ax1.set_ylim([min(accuracies)*0.95, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    best_acc_idx = accuracies.index(max(accuracies))
    bars[best_acc_idx].set_edgecolor('red')
    bars[best_acc_idx].set_linewidth(3)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: F1-Score Comparison
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(models)), f1_scores, color=colors_acc, alpha=0.7)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('F1-Score by Model')
    ax2.set_ylim([min(f1_scores)*0.95, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    best_f1_idx = f1_scores.index(max(f1_scores))
    bars[best_f1_idx].set_edgecolor('red')
    bars[best_f1_idx].set_linewidth(3)
    for i, v in enumerate(f1_scores):
        ax2.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 3: Precision vs Recall
    ax3 = axes[1, 0]
    for i, (p, r, model) in enumerate(zip(precisions, recalls, models)):
        ax3.scatter(r, p, s=200, alpha=0.6, c=[colors_acc[i]], label=model)
        ax3.annotate(model, (r, p), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision vs Recall Trade-off')
    ax3.grid(alpha=0.3)
    ax3.set_xlim([min(recalls)*0.95, 1.0])
    ax3.set_ylim([min(precisions)*0.95, 1.0])
    
    # Plot 4: Training Time Comparison
    ax4 = axes[1, 1]
    bars = ax4.bar(range(len(models)), times, color=colors_acc, alpha=0.7)
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.set_ylabel('Training Time (seconds)')
    ax4.set_title('Training Time by Model')
    ax4.grid(axis='y', alpha=0.3)
    for i, v in enumerate(times):
        ax4.text(i, v + max(times)*0.02, f'{v:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    model_comparison_file = os.path.join(save_path, 'model_comparison_analysis.png')
    plt.savefig(model_comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"  ✓ Saved to: {model_comparison_file}")
    
    return model_comparison_file
