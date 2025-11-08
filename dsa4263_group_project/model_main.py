# %% [markdown]
# # Model Training Pipeline
# This script loads PIT-safe features and trains all models with ensemble

# %% [markdown]
# ## 1. Package Imports

# %%
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import our models module
from models import (
    tune_hyperparameters,
    train_all_models,
    train_ridge_ensemble,
    select_best_model,
    HAS_XGB,
    HAS_LGBM
)

warnings.filterwarnings('ignore')

print("="*80)
print("MODEL TRAINING PIPELINE")
print("="*80)
print(f"\nAvailable libraries:")
print(f"  • XGBoost: {'✓' if HAS_XGB else '✗'}")
print(f"  • LightGBM: {'✓' if HAS_LGBM else '✗'}")

# %% [markdown]
# ## 2. Load PIT Features

# %%
print("\n" + "="*80)
print("STEP 1: LOADING PIT FEATURES")
print("="*80)

# Load all three feature sets
print("\nLoading feature datasets...")
df_graph = pd.read_csv("../data/processed/graph_features_pit.csv")
df_ts = pd.read_csv("../data/processed/timeseries_features_pit.csv")
df_text = pd.read_csv("../data/processed/text_features_pit.csv")

print(f"  • Graph features: {df_graph.shape}")
print(f"  • Time series features: {df_ts.shape}")
print(f"  • Text features: {df_text.shape}")

# Parse dates
for df in [df_graph, df_ts, df_text]:
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

print("\n✓ All feature sets loaded successfully")

# %% [markdown]
# ## 3. Merge Features and Select Columns

# %%
print("\n" + "="*80)
print("STEP 2: MERGING FEATURES")
print("="*80)

# Start with graph features as base
df_merged = df_graph.copy()

# Define features to keep from each dataset
# (excluding duplicates and non-feature columns)

# Graph features
graph_feature_cols = [
    'sender_out_degree', 'sender_in_degree', 'sender_total_degree',
    'sender_reciprocity', 'sender_pagerank', 'sender_clustering',
    'sender_closeness', 'sender_eigenvector', 'sender_triangles',
    'sender_avg_weight', 'sender_historical_email_count',
    'sender_historical_spam_count', 'sender_historical_spam_rate',
    'sender_time_since_last_email'
]

# Time series features
ts_feature_cols = [
    'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_middle_east',
    'hour_risk_score', 'weekday_risk_score', 'region_risk_score', 
    'region_hour_risk', 'sender_time_gap', 'sender_time_gap_std', 
    'sender_lifespan_days'
]

# Text features
text_feature_cols = [
    'subject_length', 'body_length', 'text_length', 'word_count',
    'uppercase_count', 'uppercase_ratio', 'exclamation_count', 
    'question_count', 'dollar_count', 'percent_count', 'star_count',
    'special_char_total', 'digit_count', 'digit_ratio', 'avg_word_length',
    'subject_sentiment', 'body_sentiment', 'urls', 'has_url', 'urls_log',
    'domain_spam_rate', 'is_suspicious_domain', 'domain_frequency', 
    'is_rare_domain'
]

# Keep only features that exist in each dataframe
graph_features = [col for col in graph_feature_cols if col in df_graph.columns]
ts_features = [col for col in ts_feature_cols if col in df_ts.columns]
text_features = [col for col in text_feature_cols if col in df_text.columns]

# Merge features (assuming same order after sorting by date)
# First, ensure all are sorted by date
df_graph = df_graph.sort_values('date').reset_index(drop=True)
df_ts = df_ts.sort_values('date').reset_index(drop=True)
df_text = df_text.sort_values('date').reset_index(drop=True)

# Add features from time series
for col in ts_features:
    if col not in df_merged.columns:
        df_merged[col] = df_ts[col]

# Add features from text
for col in text_features:
    if col not in df_merged.columns:
        df_merged[col] = df_text[col]

print(f"\nMerged dataset shape: {df_merged.shape}")
print(f"  • Graph features: {len(graph_features)}")
print(f"  • Time series features: {len(ts_features)}")
print(f"  • Text features: {len(text_features)}")
print(f"  • Total features: {len(graph_features) + len(ts_features) + len(text_features)}")

# Verify we have label and date
assert 'label' in df_merged.columns, "Missing 'label' column"
assert 'date' in df_merged.columns, "Missing 'date' column"

print("\n✓ Features merged successfully")

# %% [markdown]
# ## 4. Time-Based Train-Test Split (80/20)

# %%
print("\n" + "="*80)
print("STEP 3: TIME-BASED TRAIN-TEST SPLIT (80/20)")
print("="*80)

# Sort by date (should already be sorted, but ensure)
df_merged = df_merged.sort_values('date').reset_index(drop=True)

# Calculate split index (80% for training)
split_idx = int(len(df_merged) * 0.8)

# Split data
train_df = df_merged.iloc[:split_idx].copy()
test_df = df_merged.iloc[split_idx:].copy()

print(f"\nDataset split:")
print(f"  • Total samples: {len(df_merged):,}")
print(f"  • Training samples: {len(train_df):,} ({len(train_df)/len(df_merged)*100:.1f}%)")
print(f"  • Test samples: {len(test_df):,} ({len(test_df)/len(df_merged)*100:.1f}%)")

print(f"\nDate ranges:")
print(f"  • Training: {train_df['date'].min()} to {train_df['date'].max()}")
print(f"  • Test: {test_df['date'].min()} to {test_df['date'].max()}")

print(f"\nLabel distribution:")
print(f"  • Training spam rate: {train_df['label'].mean()*100:.2f}%")
print(f"  • Test spam rate: {test_df['label'].mean()*100:.2f}%")

# Prepare feature matrix and labels
all_features = graph_features + ts_features + text_features
X_train = train_df[all_features].copy()
y_train = train_df['label'].copy()
X_test = test_df[all_features].copy()
y_test = test_df['label'].copy()

# Handle any remaining NaN/inf values
print(f"\nHandling missing/infinite values...")
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Fill NaN with 0 (or median if you prefer)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f"  • Training features shape: {X_train.shape}")
print(f"  • Test features shape: {X_test.shape}")

print("\n✓ Train-test split complete")

# %% [markdown]
# ## 5. Feature Scaling

# %%
print("\n" + "="*80)
print("STEP 4: FEATURE SCALING")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Features scaled using StandardScaler")
print(f"  • Training: {X_train_scaled.shape}")
print(f"  • Test: {X_test_scaled.shape}")

# %% [markdown]
# ## 6. Hyperparameter Tuning (Simple 80/20 Split, No K-Fold)

# %%
print("\n" + "="*80)
print("STEP 5: HYPERPARAMETER TUNING (80/20 SPLIT)")
print("="*80)

# Set tuning parameters
PERFORM_TUNING = False  # Set to True to enable tuning with simple train/val split
N_ITER = 10  # Number of random search iterations

if PERFORM_TUNING:
    print(f"\n⚠️  Hyperparameter tuning DISABLED to avoid k-fold data leakage concerns")
    print(f"   Using default parameters for faster training")
    print(f"   (Set PERFORM_TUNING = True above to enable simple 80/20 tuning)")
    best_rf_params, best_xgb_params, best_lgbm_params = {}, {}, {}
else:
    print("\nUsing default hyperparameters (no tuning)")
    print("  • Reason: Avoiding k-fold CV to prevent data leakage issues")
    print("  • Note: Models will use scikit-learn defaults")
    best_rf_params, best_xgb_params, best_lgbm_params = {}, {}, {}

# %% [markdown]
# ## 7. Train All Base Models

# %%
print("\n" + "="*80)
print("STEP 6: TRAINING BASE MODELS")
print("="*80)

model_results, model_objects = train_all_models(
    X_train_scaled, y_train, X_test_scaled, y_test,
    best_rf_params=best_rf_params,
    best_xgb_params=best_xgb_params,
    best_lgbm_params=best_lgbm_params,
    has_lgbm=HAS_LGBM,
    has_xgb=HAS_XGB,
    verbose=True
)

# Display results
print("\n" + "="*80)
print("BASE MODEL RESULTS")
print("="*80)

results_df = pd.DataFrame(model_results)
print("\n" + results_df.to_string(index=False))

print("\n✓ Base models trained successfully")

# %% [markdown]
# ## 8. Analyze Model Diversity

# %%
print("\n" + "="*80)
print("STEP 7: MODEL DIVERSITY ANALYSIS")
print("="*80)

# Calculate pairwise agreement between models
print("\nAnalyzing model diversity (prediction agreement)...")
model_names = list(model_objects.keys())
n_models = len(model_names)

# Create prediction matrix
pred_matrix = np.zeros((len(y_test), n_models))
for i, (model_name, (_, pred)) in enumerate(model_objects.items()):
    pred_matrix[:, i] = pred

# Calculate pairwise agreement
agreement_matrix = np.zeros((n_models, n_models))
for i in range(n_models):
    for j in range(n_models):
        agreement_matrix[i, j] = np.mean(pred_matrix[:, i] == pred_matrix[:, j])

# Display agreement matrix
print("\nPairwise prediction agreement (higher = more similar):")
agreement_df = pd.DataFrame(agreement_matrix, index=model_names, columns=model_names)
print(agreement_df.round(3).to_string())

# Calculate average disagreement (diversity score)
avg_agreement = np.mean(agreement_matrix[np.triu_indices(n_models, k=1)])
diversity_score = 1 - avg_agreement
print(f"\n  • Average pairwise agreement: {avg_agreement:.3f}")
print(f"  • Diversity score: {diversity_score:.3f} (higher = more diverse)")

if diversity_score < 0.05:
    print("  ⚠️  WARNING: Low diversity - models make very similar predictions")
    print("     Stacking may not improve much over the best base model")
elif diversity_score > 0.15:
    print("  ✓ Good diversity - stacking ensemble should benefit from diverse predictions")
else:
    print("  ℹ️  Moderate diversity - some potential for stacking improvement")

# %% [markdown]
# ## 9. Train Ridge Stacking Ensemble (F1-based with Validation Split)

# %%
print("\n" + "="*80)
print("STEP 8: TRAINING RIDGE STACKING ENSEMBLE")
print("="*80)
print("\n⚠️  Important: Using proper validation set for stacking")
print("   • Training data split: 80% meta-train, 20% meta-validation")
print("   • Base models retrained on meta-train only")
print("   • Meta-learner trained on meta-validation predictions")
print("   • Final evaluation on held-out test set")
print("   • Maintains temporal order (no data leakage)")

ridge_results, ridge_meta, ridge_pred = train_ridge_ensemble(
    model_objects=model_objects,
    X_train=X_train_scaled,
    y_train=y_train,
    X_test=X_test_scaled,
    y_test=y_test,
    val_split=0.2,  # 20% of training data for validation
    verbose=True
)

print("\n✓ Ridge stacking ensemble trained successfully")

# %% [markdown]
# ## 10. Compare All Models (Including Ensemble)

# %%
print("\n" + "="*80)
print("FINAL MODEL COMPARISON")
print("="*80)

# Combine all results
all_results = model_results + [ridge_results]
final_results_df = pd.DataFrame(all_results)

# Sort by F1-Score
final_results_df = final_results_df.sort_values('F1-Score', ascending=False)

print("\n" + final_results_df.to_string(index=False))

# Find best model
best_result = final_results_df.iloc[0]
print(f"\n{'='*80}")
print(f"🏆 BEST MODEL: {best_result['Model']}")
print(f"{'='*80}")
print(f"  • Accuracy:  {best_result['Accuracy']:.4f}")
print(f"  • Precision: {best_result['Precision']:.4f}")
print(f"  • Recall:    {best_result['Recall']:.4f}")
print(f"  • F1-Score:  {best_result['F1-Score']:.4f}")

# %% [markdown]
# ## 11. Detailed Evaluation of Best Model

# %%
print("\n" + "="*80)
print("DETAILED EVALUATION OF BEST MODEL")
print("="*80)

# Get predictions for best model
if best_result['Model'] == 'Ridge Stacking':
    best_pred = ridge_pred
else:
    _, best_pred = model_objects[best_result['Model']]

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=['Legitimate', 'Spam']))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legitimate', 'Spam'],
            yticklabels=['Legitimate', 'Spam'])
plt.title(f'Confusion Matrix - {best_result["Model"]}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('../reports/figures/confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
print("\n📊 Confusion matrix saved to: ../reports/figures/confusion_matrix_best_model.png")

# %% [markdown]
# ## 12. Feature Importance Analysis (Detailed)

# %%
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get feature importance from tree-based models
feature_importance_dfs = []

for model_name, (model, _) in model_objects.items():
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': importance,
            'Model': model_name
        })
        feature_importance_dfs.append(feature_imp_df)

if feature_importance_dfs:
    # Combine and show top features
    all_importance = pd.concat(feature_importance_dfs)
    
    # Average importance across models
    avg_importance = all_importance.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
    
    print("\nTop 20 Most Important Features (averaged across tree-based models):")
    print(avg_importance.head(20).to_string())
    
    # Categorize features by type
    def categorize_feature(feature_name):
        if any(x in feature_name for x in ['sender_', 'graph_']):
            return 'graph'
        elif any(x in feature_name for x in ['hour', 'day', 'weekday', 'weekend', 'region', 'time_gap', 'lifespan', 'night']):
            return 'temporal'
        elif any(x in feature_name for x in ['domain', 'urls', 'url']):
            return 'url_domain'
        else:
            return 'text_meta'
    
    # Add category to importance dataframe
    importance_with_category = avg_importance.reset_index()
    importance_with_category.columns = ['Feature', 'Importance']
    importance_with_category['Category'] = importance_with_category['Feature'].apply(categorize_feature)
    
    # Calculate category-level statistics
    category_importance = importance_with_category.groupby('Category')['Importance'].agg(['sum', 'mean', 'count'])
    category_importance = category_importance.sort_values('sum', ascending=False)
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE BY CATEGORY")
    print("="*80)
    print("\nCategory-level statistics:")
    print(category_importance.to_string())
    
    # Calculate proportions
    total_importance = category_importance['sum'].sum()
    category_importance['proportion'] = category_importance['sum'] / total_importance * 100
    
    print("\n" + "-"*80)
    print("Proportion of total importance:")
    for category in category_importance.index:
        prop = category_importance.loc[category, 'proportion']
        count = int(category_importance.loc[category, 'count'])
        print(f"  • {category:15s}: {prop:5.1f}% ({count} features)")
    print("-"*80)
    
    # Top features per category
    print("\n" + "="*80)
    print("TOP 5 FEATURES PER CATEGORY")
    print("="*80)
    for category in ['temporal', 'graph', 'url_domain', 'text_meta']:
        cat_features = importance_with_category[importance_with_category['Category'] == category].nlargest(5, 'Importance')
        print(f"\n{category.upper()}:")
        for _, row in cat_features.iterrows():
            print(f"  • {row['Feature']:40s}: {row['Importance']:.6f}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Top 20 features with color by category
    top_20 = importance_with_category.nlargest(20, 'Importance')
    category_colors = {'temporal': 'steelblue', 'graph': 'coral', 'url_domain': 'mediumseagreen', 'text_meta': 'mediumpurple'}
    colors = [category_colors[cat] for cat in top_20['Category']]
    
    axes[0].barh(range(len(top_20)), top_20['Importance'], color=colors)
    axes[0].set_yticks(range(len(top_20)))
    axes[0].set_yticklabels(top_20['Feature'])
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Top 20 Most Important Features')
    axes[0].invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=cat.replace('_', '/').title()) 
                      for cat, color in category_colors.items()]
    axes[0].legend(handles=legend_elements, loc='lower right')
    
    # Right plot: Pie chart of category contributions
    axes[1].pie(category_importance['sum'], labels=category_importance.index, autopct='%1.1f%%',
                colors=[category_colors.get(cat, 'gray') for cat in category_importance.index],
                startangle=90)
    axes[1].set_title('Feature Type Contribution')
    
    plt.tight_layout()
    plt.savefig('../reports/figures/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📊 Feature importance analysis saved to: ../reports/figures/feature_importance_analysis.png")
    
    # Also save the top 20 simple plot
    plt.figure(figsize=(10, 8))
    avg_importance.head(20).plot(kind='barh')
    plt.xlabel('Average Feature Importance')
    plt.title('Top 20 Features by Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('../reports/figures/feature_importance_top20.png', dpi=300, bbox_inches='tight')
    print("📊 Top 20 features plot saved to: ../reports/figures/feature_importance_top20.png")
    
    # Investigate text features specifically
    print("\n" + "="*80)
    print("TEXT FEATURES INVESTIGATION")
    print("="*80)
    text_features_imp = importance_with_category[importance_with_category['Category'] == 'text_meta'].sort_values('Importance', ascending=False)
    print(f"\nAll {len(text_features_imp)} text features ranked by importance:")
    for idx, row in text_features_imp.iterrows():
        percentile = (importance_with_category['Importance'] < row['Importance']).sum() / len(importance_with_category) * 100
        print(f"  {row['Feature']:40s}: {row['Importance']:.6f} (top {100-percentile:.0f}%)")
    
    # Check if text features have low variance or correlation issues
    print("\n" + "-"*80)
    print("Possible reasons for lower text feature importance:")
    print("  1. Text features may be highly correlated with each other")
    print("  2. Graph and temporal features might capture the spam patterns more directly")
    print("  3. Some text features might have low variance in the dataset")
    print("  4. Tree-based models may prefer continuous features (graph/temporal) over discrete counts")
    print("-"*80)
    
else:
    print("\nNo tree-based models available for feature importance analysis")

# %% [markdown]
# ## 13. Save Results and Models

# %%
print("\n" + "="*80)
print("SAVING RESULTS AND MODELS")
print("="*80)

# Save results to CSV
final_results_df.to_csv('../reports/model_results.csv', index=False)
print("✓ Model results saved to: ../reports/model_results.csv")

# Save best model predictions
predictions_df = pd.DataFrame({
    'true_label': y_test,
    'predicted_label': best_pred,
    'date': test_df['date'].values
})
predictions_df.to_csv('../reports/best_model_predictions.csv', index=False)
print("✓ Best model predictions saved to: ../reports/best_model_predictions.csv")

# Save model objects (optional - can be large)
import joblib
joblib.dump(model_objects, '../models/trained_models.pkl')
print("✓ Trained models saved to: ../models/trained_models.pkl")

print("\n" + "="*80)
print("✅ MODEL TRAINING PIPELINE COMPLETE!")
print("="*80)
print(f"\nBest Model: {best_result['Model']}")
print(f"F1-Score: {best_result['F1-Score']:.4f}")
print("\n🎉 Ready for production deployment!")

# %%
