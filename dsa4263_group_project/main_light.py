# %%
"""
EMAIL SPAM DETECTION - LIGHTWEIGHT PIPELINE

This is a refactored version of main.py with better modularity and readability.
All heavy lifting is done by specialized modules:
- data_cleaning: Data preprocessing and text cleaning
- feature_engineering: Feature extraction (temporal, URL, text, graph)
- models: Model training and hyperparameter tuning
- visualization: EDA and result visualizations
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import our custom modules
from dsa4263_group_project.data_cleaning import clean_email_data, clean_text
from dsa4263_group_project.feature_engineering import (
    engineer_temporal_features,
    engineer_url_domain_features,
    engineer_text_meta_features,
    prepare_features_with_graph
)
from dsa4263_group_project.models import (
    tune_hyperparameters,
    train_all_models,
    select_best_model
)
from dsa4263_group_project.visualization import (
    create_temporal_eda_plots,
    create_feature_importance_plots,
    create_model_comparison_plots
)

# Check for optional dependencies
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("⚠️  LightGBM not available")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not available")

try:
    from sklearn.model_selection import RandomizedSearchCV
    HAS_TUNING = True
except ImportError:
    HAS_TUNING = False
    print("⚠️  RandomizedSearchCV not available")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer
    
    # Download NLTK data
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    
    STOP_WORDS = set(stopwords.words('english'))
    STOP_WORDS.update(['re', 'fwd', 'subject', 'email', 'com', 'www'])
    LEMMATIZER = WordNetLemmatizer()
    SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()
    HAS_NLP = True
except Exception as e:
    HAS_NLP = False
    STOP_WORDS = None
    LEMMATIZER = None
    SENTIMENT_ANALYZER = None
    print(f"⚠️  NLP tools not available: {e}")

warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_PATH = os.path.join(PROJECT_ROOT, "reports", "figures")

os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

# Configuration
ENABLE_TUNING = True  # Set to False to skip hyperparameter tuning


def main():
    """Main pipeline execution."""
    
    print("="*80)
    print(" EMAIL SPAM DETECTION - LIGHTWEIGHT PIPELINE")
    print("="*80)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Path: {DATA_PROCESSED_PATH}")
    print(f"Reports Path: {REPORTS_PATH}")
    
    # ========================================================================
    # SECTION 1: DATA LOADING & CLEANING
    # ========================================================================
    print("\n" + "="*80)
    print(" SECTION 1: DATA LOADING & CLEANING")
    print("="*80)
    
    # Load data
    data_file = os.path.join(DATA_PROCESSED_PATH, "date_merge.csv")
    if not os.path.exists(data_file):
        print(f"❌ ERROR: Data file not found: {data_file}")
        sys.exit(1)
    
    print(f"\n[1.1] Loading data from {os.path.basename(data_file)}...")
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce', utc=True)
    df = df[df['date'].notna()].copy()
    df['date'] = df['date'].dt.tz_localize(None)
    
    print(f"  ✓ Loaded {len(df):,} emails")
    print(f"  ✓ Spam rate: {df['label'].mean()*100:.1f}%")
    
    # Add basic temporal features if not present
    if 'hour' not in df.columns:
        print(f"\n[1.2] Adding basic temporal features...")
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_name'] = df['date'].dt.day_name()
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['timezone_region'] = 'Americas'  # Simplified
        print(f"  ✓ Added temporal features")
    
    # Text cleaning
    if HAS_NLP:
        print(f"\n[1.3] Cleaning text with NLTK...")
        df['subject_clean'] = df['subject'].fillna('').apply(
            lambda x: clean_text(x, STOP_WORDS, LEMMATIZER)
        )
        df['body_clean'] = df['body'].fillna('').apply(
            lambda x: clean_text(x, STOP_WORDS, LEMMATIZER)
        )
        df['text_combined'] = df['subject_clean'] + ' ' + df['body_clean']
        print(f"  ✓ Text cleaning complete")
    else:
        print(f"\n[1.3] Skipping text cleaning (NLTK not available)")
        df['text_combined'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    
    print(f"\n✅ Data cleaning complete: {len(df):,} emails ready")
    
    # ========================================================================
    # SECTION 2: EXPLORATORY DATA ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print(" SECTION 2: EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Create temporal EDA plots
    eda_file = create_temporal_eda_plots(df, REPORTS_PATH, verbose=True)
    
    print(f"\n✅ EDA complete!")
    
    # ========================================================================
    # SECTION 3: FEATURE ENGINEERING
    # ========================================================================
    print("\n" + "="*80)
    print(" SECTION 3: FEATURE ENGINEERING")
    print("="*80)
    
    # 3.1: Temporal features
    df_with_features = engineer_temporal_features(df, verbose=True)
    
    # 3.2: URL and domain features
    df_with_features = engineer_url_domain_features(df_with_features, verbose=True)
    
    # 3.3: Text meta-features
    df_with_features = engineer_text_meta_features(
        df_with_features,
        sentiment_analyzer=SENTIMENT_ANALYZER if HAS_NLP else None,
        verbose=True
    )
    
    # Define feature lists
    temporal_features = [
        'hour', 'day_of_week', 'is_weekend',
        'hour_risk_score', 'weekday_risk_score', 'region_risk_score',
        'is_high_risk_region', 'sender_historical_spam_rate',
        'sender_email_count', 'time_since_last_email'
    ]
    
    url_domain_features = [
        'urls', 'has_url', 'urls_log',
        'domain_spam_rate', 'is_suspicious_domain', 'domain_frequency', 'is_rare_domain',
        'is_night'
    ]
    
    text_meta_features = [
        'subject_length', 'body_length', 'text_length', 'word_count',
        'uppercase_ratio', 'exclamation_count', 'dollar_count',
        'special_char_total', 'digit_ratio', 'avg_word_length',
        'subject_sentiment', 'body_sentiment'
    ]
    
    graph_feature_names = [
        'graph_out_degree', 'graph_in_degree', 'graph_reciprocity', 'graph_total_sent',
        'graph_pagerank', 'graph_clustering', 'graph_degree_centrality', 'graph_avg_weight'
    ]
    
    print(f"\n✅ Non-graph feature engineering complete!")
    print(f"  • Temporal: {len(temporal_features)}")
    print(f"  • URL/Domain: {len(url_domain_features)}")
    print(f"  • Text Meta: {len(text_meta_features)}")
    print(f"  • Graph: Will be added after train/test split")
    
    # ========================================================================
    # SECTION 4: MODEL TRAINING & COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print(" SECTION 4: MODEL TRAINING & COMPARISON")
    print("="*80)
    
    # First split WITHOUT graph features (to prevent leakage)
    print(f"\n[4.1] Splitting data into train/test sets...")
    non_graph_features = temporal_features + url_domain_features + text_meta_features
    X_temp = df_with_features[non_graph_features].fillna(0)
    y = df_with_features['label']
    
    _, _, _, _, train_idx, test_idx = train_test_split(
        X_temp, y, df_with_features.index,
        test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"  ✓ Train set: {len(train_idx):,} samples")
    print(f"  ✓ Test set:  {len(test_idx):,} samples")
    
    # Build graph features on training data only
    X_train, X_test, y_train, y_test = prepare_features_with_graph(
        df_with_features,
        train_idx,
        test_idx,
        temporal_features,
        url_domain_features,
        text_meta_features,
        graph_feature_names,
        verbose=True
    )
    
    # Hyperparameter tuning
    best_rf_params, best_xgb_params, best_lgbm_params = tune_hyperparameters(
        X_train, y_train,
        has_lgbm=HAS_LGBM,
        has_xgb=HAS_XGB,
        has_tuning=ENABLE_TUNING and HAS_TUNING,
        verbose=True
    )
    
    # Train all models
    model_results, model_objects = train_all_models(
        X_train, y_train, X_test, y_test,
        best_rf_params, best_xgb_params, best_lgbm_params,
        has_lgbm=HAS_LGBM,
        has_xgb=HAS_XGB,
        verbose=True
    )
    
    # Display results
    print(f"\n{'='*80}")
    print(" MODEL COMPARISON RESULTS")
    print("="*80)
    results_df = pd.DataFrame(model_results)
    print(results_df.to_string(index=False))
    
    # Select best model
    best_model, best_model_obj, best_predictions = select_best_model(
        model_results, model_objects, verbose=True
    )
    
    # Detailed classification report
    print(f"\n{'='*80}")
    print(" DETAILED CLASSIFICATION REPORT (BEST MODEL)")
    print("="*80)
    print(classification_report(y_test, best_predictions, target_names=['Legitimate', 'Spam']))
    
    # ========================================================================
    # SECTION 5: FEATURE IMPORTANCE & VISUALIZATION
    # ========================================================================
    print("\n" + "="*80)
    print(" SECTION 5: FEATURE IMPORTANCE & VISUALIZATION")
    print("="*80)
    
    # Get feature importance
    all_ml_features = temporal_features + url_domain_features + text_meta_features + graph_feature_names
    feature_types = (
        ['temporal']*len(temporal_features) +
        ['url_domain']*len(url_domain_features) +
        ['text_meta']*len(text_meta_features) +
        ['graph']*len(graph_feature_names)
    )
    
    if hasattr(best_model_obj, 'feature_importances_'):
        feature_importances = best_model_obj.feature_importances_
    elif hasattr(best_model_obj, 'coef_'):
        feature_importances = np.abs(best_model_obj.coef_[0])
    else:
        # Fallback to Random Forest
        rf_obj, _ = model_objects['Random Forest']
        feature_importances = rf_obj.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': all_ml_features,
        'importance': feature_importances,
        'type': feature_types
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 20 Features:")
    print(importance_df.head(20).to_string(index=False))
    
    # Create visualizations
    importance_file = create_feature_importance_plots(importance_df, REPORTS_PATH, verbose=True)
    model_comparison_file = create_model_comparison_plots(model_results, REPORTS_PATH, verbose=True)
    
    # ========================================================================
    # SECTION 6: FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print(" PIPELINE COMPLETE - SUMMARY")
    print("="*80)
    
    print(f"\n📊 DATA STATISTICS:")
    print(f"  • Input emails:           {len(df):,}")
    print(f"  • Training set:           {len(X_train):,}")
    print(f"  • Test set:               {len(X_test):,}")
    print(f"  • Unique senders:         {df['sender'].nunique():,}")
    print(f"  • Spam rate:              {df['label'].mean()*100:.1f}%")
    
    print(f"\n🎯 FEATURES GENERATED:")
    print(f"  • Temporal features:      {len(temporal_features)}")
    print(f"  • URL/Domain features:    {len(url_domain_features)}")
    print(f"  • Text meta features:     {len(text_meta_features)}")
    print(f"  • Graph features:         {len(graph_feature_names)}")
    print(f"  • Total ML features:      {len(all_ml_features)}")
    
    print(f"\n🏆 MODEL PERFORMANCE:")
    print(f"  • Best Model:             {best_model['Model']}")
    print(f"  • Accuracy:               {best_model['Accuracy']*100:.2f}%")
    print(f"  • F1-Score:               {best_model['F1-Score']:.4f}")
    print(f"  • Models trained:         {len(model_results)}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  • Source data:            {data_file}")
    print(f"  • EDA visualization:      {eda_file}")
    print(f"  • Feature importance:     {importance_file}")
    print(f"  • Model comparison:       {model_comparison_file}")
    
    print(f"\n💡 KEY INSIGHTS:")
    type_importance = importance_df.groupby('type')['importance'].sum()
    print(f"  • Top feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['type']})")
    print(f"    Importance: {importance_df.iloc[0]['importance']:.4f}")
    print(f"\n  • Feature type contributions:")
    for ftype in sorted(type_importance.index):
        print(f"    - {ftype.replace('_', ' ').title()}: {type_importance[ftype]/type_importance.sum()*100:.1f}%")
    
    print(f"\n  • Model rankings (by F1-Score):")
    sorted_models = sorted(model_results, key=lambda x: x['F1-Score'], reverse=True)
    for i, model in enumerate(sorted_models, 1):
        marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"    {marker} {i}. {model['Model']:<20s} F1: {model['F1-Score']:.4f} | Acc: {model['Accuracy']:.4f}")
    
    print("\n" + "="*80)
    print("✅ PIPELINE EXECUTION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
