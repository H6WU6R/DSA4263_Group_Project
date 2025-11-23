# %%
import sys
import os
import time
import argparse
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Import our modules
from data_cleaning_finalized import DataCleaner
from feature_engineering_finalized import FeatureEngineer
from modeling_finalized import ModelTrainer

warnings.filterwarnings('ignore')

# Setup NLTK resources
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        print(f"  • Downloading {resource}...")
        nltk.download(resource, quiet=True)

stop_words = set(stopwords.words('english'))
stop_words.update(['re', 'fwd', 'subject'])
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrices(y_test, predictions_dict, save_dir="reports/figures"):
    """
    Plot confusion matrices for all models.
    
    Args:
        y_test: True test labels
        predictions_dict: Dictionary mapping model names to predictions
        save_dir: Directory to save plots
    """
    # Get project root
    project_root = Path(__file__).parent.parent
    save_path = project_root / save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING CONFUSION MATRICES")
    print("="*80)
    
    # Calculate grid size
    n_models = len(predictions_dict)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        ax = axes[idx]
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar=True, square=True)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xticklabels(['Ham', 'Spam'])
        ax.set_yticklabels(['Ham', 'Spam'])
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = save_path / "confusion_matrices.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Confusion matrices saved to: {fig_path}")
    
    plt.close()


def plot_metrics_comparison(results_df, save_dir="reports/figures"):
    """
    Plot grouped bar chart comparing metrics across models.
    
    Args:
        results_df: DataFrame with model results
        save_dir: Directory to save plots
    """
    # Get project root
    project_root = Path(__file__).parent.parent
    save_path = project_root / save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING METRICS COMPARISON CHART")
    print("="*80)
    
    # Prepare data
    metrics_to_plot = ['F1-Score', 'Precision', 'Recall', 'Accuracy']
    plot_data = results_df[['Model', 'Type'] + metrics_to_plot].copy()
    
    # Create model labels with type
    plot_data['Model_Label'] = plot_data['Model'] + ' (' + plot_data['Type'] + ')'
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for grouped bar chart
    x = np.arange(len(plot_data))
    width = 0.2
    
    # Plot each metric
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    for i, metric in enumerate(metrics_to_plot):
        offset = width * (i - 1.5)
        ax.bar(x + offset, plot_data[metric], width, 
               label=metric, color=colors[i], alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data['Model_Label'], rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = save_path / "metrics_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Metrics comparison saved to: {fig_path}")
    
    plt.close()

# %%
cleaner = DataCleaner(stop_words=stop_words, lemmatizer=lemmatizer, stemmer=stemmer)

# Load raw datasets
print("\n📂 Loading raw datasets...")
raw_datasets = [
    cleaner.load_raw_data('CEAS_08.csv'),
    cleaner.load_raw_data('Nazario.csv'),
    cleaner.load_raw_data('Nigerian_Fraud.csv'),
    cleaner.load_raw_data('SpamAssasin.csv')
]
print(f"   Loaded {len(raw_datasets)} datasets")
print(f"   Total rows: {sum(len(df) for df in raw_datasets):,}")

# Step 1: Clean and merge datasets
print("\n" + "=" * 64)
print("STEP 1: CLEAN AND MERGE DATASETS")
print("=" * 64)
merged_df = cleaner.clean_and_merge(raw_datasets, sender_col='sender', receiver_col='receiver')
print(f"Merged rows              : {len(merged_df):,}")

# Add email_id after merging
merged_df['email_id'] = merged_df.index

# Step 2: Clean dates
print("\n" + "=" * 64)
print("STEP 2: CLEAN DATES")
print("=" * 64)
cleaned_df = cleaner.clean_dates(merged_df, date_col='date')

# Step 3: Clean text
print("\n" + "=" * 64)
print("STEP 3: CLEAN TEXT")
print("=" * 64)
if 'full_text' in cleaned_df.columns:
    print("Cleaning 'full_text' column...")
    cleaned_df['full_text'] = cleaner.clean_text_column(cleaned_df['full_text'], show_progress=True)
    empty_after = (cleaned_df['full_text'] == "").sum()
    print(f"Empty after cleaning     : {empty_after:,} rows")
else:
    print("⚠️  'full_text' column not found!")

# Final summary
print("\n" + "=" * 64)
print("FINAL SUMMARY")
print("=" * 64)
print(f"Output rows              : {len(cleaned_df):,}")
print(f"Columns                  : {list(cleaned_df.columns)}")

# Save cleaned data
cleaner.save_processed_data(cleaned_df, 'cleaned_date_merge.csv')
print(f"\n💾 Cleaned data saved to: data/processed/cleaned_date_merge.csv")
print("=" * 64 + "\n")
# %%
# ============================================================================
# FEATURE ENGINEERING WITH AUTOMATIC BEST FEATURE SELECTION
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING PIPELINE")
print("=" * 80)

# Initialize feature engineer
engineer = FeatureEngineer(verbose=True)

# Prepare data for feature engineering
df_for_features = cleaned_df.copy()
df_for_features = df_for_features.dropna(subset=['sender', 'receiver', 'date', 'label'])
df_for_features = df_for_features.sort_values('date').reset_index(drop=True)

print(f"\n📊 Data ready for feature engineering: {len(df_for_features):,} rows")
print(f"   Spam rate: {df_for_features['label'].mean():.2%}")
# %%
# ============================================================================
# STEP 4: GRAPH FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: GRAPH FEATURES + ABLATION STUDY")
print("=" * 80)

start_time = time.time()
df_graph = engineer.compute_graph_features_parallel(df_for_features.copy())
graph_time = time.time() - start_time
print(f"\n⏱️  Graph features computed in {graph_time:.2f} seconds")

# Run ablation study to find best graph features
graph_results = engineer.graph_feature_ablation_study(df_graph)
print("\n📊 Graph Feature Ablation Results:")
print(graph_results.to_string(index=False))

# Automatically select best graph features (by F1-Score)
best_graph_name, best_graph_features, best_graph_metrics = engineer.select_best_feature_group(
    graph_results,
    engineer._graph_feature_groups,
    metric='F1-Score'
)

# Save ablation results
graph_results.to_csv('../data/processed/graph_ablation_results.csv', index=False)
print(f"\n💾 Graph ablation results saved")
# %%
# ============================================================================
# STEP 5: TIMESERIES FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: TIMESERIES FEATURES + ABLATION STUDY")
print("=" * 80)

start_time = time.time()
df_ts = engineer.compute_timeseries_features_parallel(df_for_features.copy())
ts_time = time.time() - start_time
print(f"\n⏱️  Timeseries features computed in {ts_time:.2f} seconds")

# Run ablation study to find best timeseries features
ts_results = engineer.timeseries_feature_ablation_study(df_ts)
print("\n📊 Timeseries Feature Ablation Results:")
print(ts_results.to_string(index=False))

# Automatically select best timeseries features (by F1-Score)
best_ts_name, best_ts_features, best_ts_metrics = engineer.select_best_feature_group(
    ts_results,
    engineer._timeseries_feature_groups,
    metric='F1-Score'
)

# Save ablation results
ts_results.to_csv('../data/processed/timeseries_ablation_results.csv', index=False)
print(f"\n💾 Timeseries ablation results saved")
# %%
# ============================================================================
# STEP 6: TEXT FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: TEXT FEATURES + ABLATION STUDY")
print("=" * 80)

start_time = time.time()
df_text = engineer.compute_text_features_parallel(df_for_features.copy())
text_time = time.time() - start_time
print(f"\n⏱️  Text features computed in {text_time:.2f} seconds")

# Run ablation study to find best text features
text_results = engineer.text_feature_ablation_study(df_text)
print("\n📊 Text Feature Ablation Results:")
print(text_results.to_string(index=False))

# Automatically select best text features (by F1-Score)
best_text_name, best_text_features, best_text_metrics = engineer.select_best_feature_group(
    text_results,
    engineer._text_feature_groups,
    metric='F1-Score'
)

# Save ablation results
text_results.to_csv('../data/processed/text_ablation_results.csv', index=False)
print(f"\n💾 Text ablation results saved")

# ============================================================================
# STEP 7: MERGE BEST FEATURES AND SAVE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: MERGE SELECTED FEATURES")
print("=" * 80)

# Prepare dataframes with only selected features
df_graph_selected = df_graph[['sender', 'date', 'label'] + best_graph_features].copy()
df_ts_selected = df_ts[['sender', 'date'] + best_ts_features].copy()
df_text_selected = df_text[['sender', 'date'] + best_text_features].copy()

# Merge all selected features
print("\n🔗 Merging selected features...")
df_final = df_graph_selected.merge(df_ts_selected, on=['sender', 'date'], how='inner')
df_final = df_final.merge(df_text_selected, on=['sender', 'date'], how='inner')

print(f"\n✅ Final dataset shape: {df_final.shape}")
print(f"   Total features: {len(best_graph_features) + len(best_ts_features) + len(best_text_features)}")
print(f"   - Graph features: {len(best_graph_features)} ({best_graph_name})")
print(f"   - Timeseries features: {len(best_ts_features)} ({best_ts_name})")
print(f"   - Text features: {len(best_text_features)} ({best_text_name})")

# Save final engineered features
output_path = 'data/processed/engineered_features_selected.csv'
df_final.to_csv(output_path, index=False)
print(f"\n💾 Final features saved to: {output_path}")

# Save feature selection summary
feature_summary = pd.DataFrame({
    'Feature Type': ['Graph', 'Timeseries', 'Text'],
    'Selected Group': [best_graph_name, best_ts_name, best_text_name],
    'Num Features': [len(best_graph_features), len(best_ts_features), len(best_text_features)],
    'F1-Score': [best_graph_metrics['F1-Score'], best_ts_metrics['F1-Score'], best_text_metrics['F1-Score']],
    'Accuracy': [best_graph_metrics['Accuracy'], best_ts_metrics['Accuracy'], best_text_metrics['Accuracy']],
    'Precision': [best_graph_metrics['Precision'], best_ts_metrics['Precision'], best_text_metrics['Precision']],
    'Recall': [best_graph_metrics['Recall'], best_ts_metrics['Recall'], best_text_metrics['Recall']]
})
feature_summary.to_csv('data/processed/feature_selection_summary.csv', index=False)
print(f"💾 Feature selection summary saved")

# Print feature list
print("\n📋 SELECTED FEATURES:")
print("\n🔹 Graph Features:")
for i, feat in enumerate(best_graph_features, 1):
    print(f"   {i}. {feat}")

print("\n🔹 Timeseries Features:")
for i, feat in enumerate(best_ts_features, 1):
    print(f"   {i}. {feat}")

print("\n🔹 Text Features:")
for i, feat in enumerate(best_text_features, 1):
    print(f"   {i}. {feat}")

print("\n" + "=" * 80)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("=" * 80)
print(f"\n⏱️  Total time:")
print(f"   Graph: {graph_time:.2f}s")
print(f"   Timeseries: {ts_time:.2f}s")
print(f"   Text: {text_time:.2f}s")
print(f"   Total: {graph_time + ts_time + text_time:.2f}s")

print("\n📁 Output files:")
print("   • data/processed/cleaned_date_merge.csv (cleaned data)")
print("   • data/processed/graph_ablation_results.csv")
print("   • data/processed/timeseries_ablation_results.csv")
print("   • data/processed/text_ablation_results.csv")
print("   • data/processed/engineered_features_selected.csv (final features)")
print("   • data/processed/feature_selection_summary.csv")
print("\n" + "=" * 80 + "\n")
# %%
# ============================================================================
# STEP 8: MODEL TRAINING AND EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: MODEL TRAINING AND EVALUATION")
print("=" * 80)

# Define feature columns (exclude metadata)
exclude_cols = ['date', 'label', 'sender', 'receiver', 'subject', 'body', 
                'full_text', 'cleaned_text', 'sender_domain', 'timezone_region']
feature_cols = [col for col in df_final.columns if col not in exclude_cols]

print(f"\nDataset ready for modeling: {len(df_final):,} samples, {len(feature_cols)} features")

# Initialize model trainer
trainer = ModelTrainer(verbose=True, random_state=42)

# Prepare data (random 80/20 split)
print("\n📊 Preparing data with random 80/20 split...")
X_train, X_test, y_train, y_test, train_dates, test_dates = trainer.prepare_data(
    df_final, feature_cols, label_col='label', test_size=0.2
)

# Train baseline models
print("\n🔧 Training baseline models...")
baseline_df = trainer.train_baseline_models(X_train, y_train, X_test, y_test)

# Tune hyperparameters (all models, random 3-fold CV)
print("\n⚙️  Tuning hyperparameters...")
tuned_df = trainer.tune_hyperparameters(
    X_train, y_train, X_test, y_test,
    models_to_tune=None,  # Tune all baseline models
    n_iter=20,
    cv_splits=3
)

# Train stacking ensemble (logistic regression meta-learner)
print("\n🔗 Training stacking ensemble...")
ensemble_metrics, ensemble_pred = trainer.train_stacking_ensemble(
    X_train, y_train, X_test, y_test,
    use_tuned=True,
    val_split=0.2
)

# Compare all models and save results
print("\n📊 Comparing all models...")
final_results_df = trainer.compare_all_models(
    include_ensemble=True,
    ensemble_metrics=ensemble_metrics
)
trainer.save_results(final_results_df, filename="model_results_finalized.csv")

# Generate visualizations
print("\n📊 Generating visualizations...")

# Collect all predictions for confusion matrices
predictions_dict = {}

# Add tuned models (prefer tuned over baseline)
for model_name, y_pred in trainer.tuned_predictions.items():
    predictions_dict[f"{model_name} (Tuned)"] = y_pred

# Add baseline models that weren't tuned
for model_name, y_pred in trainer.baseline_predictions.items():
    if model_name not in trainer.tuned_predictions:
        predictions_dict[f"{model_name} (Baseline)"] = y_pred

# Add ensemble
predictions_dict["Logistic Stacking (Ensemble)"] = ensemble_pred

# Generate plots
plot_confusion_matrices(y_test, predictions_dict, save_dir="reports/figures")
plot_metrics_comparison(final_results_df, save_dir="reports/figures")

# Save trained models
# trainer.save_models(
#     models_dir="models",
#     save_baseline=False,
#     save_tuned=True
# )

print("\n✅ Model training and evaluation complete!")
print(f"\n💾 Model results saved to: data/processed/model_results_finalized.csv")

print("\n📁 All output files:")
print("   • data/processed/cleaned_date_merge.csv (cleaned data)")
print("   • data/processed/graph_ablation_results.csv")
print("   • data/processed/timeseries_ablation_results.csv")
print("   • data/processed/text_ablation_results.csv")
print("   • data/processed/engineered_features_selected.csv (final features)")
print("   • data/processed/feature_selection_summary.csv")
print("   • data/processed/model_results_finalized.csv (model comparison)")
print("   • reports/figures/confusion_matrices.png")
print("   • reports/figures/metrics_comparison.png")
print("   • models/tuned/ (trained models)")
print("   • models/scaler.pkl (feature scaler)")
print("\n" + "=" * 80)
print("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
print("=" * 80 + "\n")
# %%
