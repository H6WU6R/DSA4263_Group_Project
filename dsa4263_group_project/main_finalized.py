"""
Complete End-to-End Pipeline for Spam Detection
================================================
This script runs the complete pipeline:
1. Data Cleaning (DataCleaner)
2. Feature Engineering (FeatureEngineer)
3. Model Training (ModelTrainer) - baseline + tuning + ensemble

Usage:
    python main_finalized.py [--skip-cleaning] [--skip-features] [--skip-tuning]
    
Options:
    --skip-cleaning: Skip data cleaning step (use existing cleaned data)
    --skip-features: Skip feature engineering step (use existing features)
    --skip-tuning: Skip hyperparameter tuning (baseline + ensemble only)
"""

import sys
import os
import time
import argparse
import pandas as pd
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Import our modules
from data_cleaning_finalized import DataCleaner
from feature_engineering_finalized import FeatureEngineer
from modeling_finalized import ModelTrainer

warnings.filterwarnings('ignore')


def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    print("\n" + char*80)
    print(title.center(80))
    print(char*80)


def step_1_data_cleaning(
    input_path: str = "../data/processed/date_merge.csv",
    output_path: str = "../data/processed/cleaned_date_merge.csv",
    skip: bool = False
) -> pd.DataFrame:
    """
    Step 1: Data Cleaning
    Clean dates, handle missing values, clean text
    """
    print_header("STEP 1: DATA CLEANING")
    
    text_columns = [
        ('subject', 'subject_clean'),
        ('body', 'body_clean')
    ]
    
    def _setup_text_resources():
        print("\n📦 Setting up NLTK...")
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            for resource in ['punkt', 'stopwords', 'wordnet']:
                try:
                    nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
                except LookupError:
                    print(f"  • Downloading {resource}...")
                    nltk.download(resource, quiet=True)
            
            stop_words = set(stopwords.words('english'))
            stop_words.update(['re', 'fwd', 'subject'])  # Email-specific stopwords
            lemmatizer = WordNetLemmatizer()
            
            print("  ✓ NLTK resources ready")
        except Exception as e:
            print(f"  ⚠️  NLTK setup failed: {e}")
            print("     Continuing without NLTK features...")
            stop_words = None
            lemmatizer = None
        return stop_words, lemmatizer
    
    def _apply_text_cleaning(df_to_clean: pd.DataFrame, cleaner: DataCleaner, only_missing: bool = False):
        for source_col, cleaned_col in text_columns:
            if source_col not in df_to_clean.columns:
                print(f"  • Skipping '{source_col}' (column not found)")
                continue
            if only_missing and cleaned_col in df_to_clean.columns:
                continue
            action = "Updating" if not only_missing else "Filling missing"
            print(f"  • {action} '{source_col}' text -> '{cleaned_col}'")
            source_series = df_to_clean[source_col].fillna("")
            df_to_clean[cleaned_col] = cleaner.clean_text_column(source_series, show_progress=True)
    
    if skip and os.path.exists(output_path):
        print(f"\n⏭️  Skipping data cleaning (using existing file)")
        print(f"   Loading: {output_path}")
        df = pd.read_csv(output_path)
        df['date'] = pd.to_datetime(df['date'])
        if 'email_id' not in df.columns:
            df['email_id'] = df.index
            print("  • Added missing 'email_id' column")
        missing_clean_cols = [col for _, col in text_columns if col not in df.columns]
        if missing_clean_cols:
            print("  • Detected missing cleaned text columns. Updating existing file...")
            stop_words, lemmatizer = _setup_text_resources()
            cleaner = DataCleaner(stop_words=stop_words, lemmatizer=lemmatizer)
            _apply_text_cleaning(df, cleaner, only_missing=True)
            df.to_csv(output_path, index=False)
        print(f"   ✓ Loaded {len(df):,} emails")
        return df
    
    print(f"\nLoading raw data from: {input_path}")
    df = pd.read_csv(input_path)
    df['email_id'] = df.index
    print(f"  • Initial rows: {len(df):,}")
    
    stop_words, lemmatizer = _setup_text_resources()
    
    # Initialize cleaner
    cleaner = DataCleaner(stop_words=stop_words, lemmatizer=lemmatizer)
    
    # Clean data
    print("\n🧹 Cleaning data...")
    df = cleaner.clean_and_merge(df)
    print(f"  ✓ Handled missing values: {len(df):,} rows")
    
    df = cleaner.clean_dates(df)
    print(f"  ✓ Cleaned dates: {len(df):,} rows")

    # Clean subject/body text for downstream reuse
    _apply_text_cleaning(df, cleaner)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"\n💾 Cleaned data saved to: {output_path}")
    print(f"   Final rows: {len(df):,}")
    
    return df


def step_2_feature_engineering(
    df: pd.DataFrame,
    output_dir: str = "../data/processed",
    skip: bool = False
) -> pd.DataFrame:
    """
    Step 2: Feature Engineering
    Generate graph, time series, and text features
    """
    print_header("STEP 2: FEATURE ENGINEERING")
    
    # Check if features already exist
    graph_path = os.path.join(output_dir, "graph_features_pit.csv")
    ts_path = os.path.join(output_dir, "timeseries_features_pit.csv")
    text_path = os.path.join(output_dir, "text_features_pit.csv")
    merged_path = os.path.join(output_dir, "engineered_features.csv")
    
    if skip and all(os.path.exists(p) for p in [graph_path, ts_path, text_path]):
        print(f"\n⏭️  Skipping feature engineering (using existing files)")
        print(f"   Loading pre-computed features...")
        
        df_graph = pd.read_csv(graph_path)
        df_ts = pd.read_csv(ts_path)
        df_text = pd.read_csv(text_path)
        
        # Merge features
        df_merged = merge_features(df_graph, df_ts, df_text, output_dir)
        print(f"   ✓ Loaded {df_merged.shape[1]} total features")
        
        return df_merged
    
    print(f"\nPreparing data for feature engineering...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['sender', 'receiver', 'date', 'label'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  • Dataset: {len(df):,} emails")
    print(f"  • Date range: {df['date'].min()} to {df['date'].max()}")
    
    print(f"\n⏳ Running sequential feature engineering...")
    start_time = time.time()
    
    engineer = FeatureEngineer(verbose=True)
    
    df_graph = engineer.compute_graph_features(df.copy())
    df_ts = engineer.compute_timeseries_features(df.copy())
    df_text = engineer.compute_text_features(df.copy())
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Feature engineering time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    
    # Save individual feature sets
    engineer = FeatureEngineer(verbose=False)
    engineer.save_features(df_graph, graph_path)
    engineer.save_features(df_ts, ts_path)
    engineer.save_features(df_text, text_path)
    
    # Merge all features
    df_merged = merge_features(df_graph, df_ts, df_text, output_dir)
    
    return df_merged


def merge_features(
    df_graph: pd.DataFrame,
    df_ts: pd.DataFrame,
    df_text: pd.DataFrame,
    output_dir: str
) -> pd.DataFrame:
    """
    Merge all feature sets into one DataFrame.
    """
    print(f"\n🔗 Merging feature sets...")
    
    key_col = 'email_id'
    dataframes = [
        ('graph', df_graph),
        ('timeseries', df_ts),
        ('text', df_text)
    ]
    
    for name, df in dataframes:
        if key_col not in df.columns:
            raise KeyError(f"'{key_col}' missing in {name} features. "
                           "Please rerun data cleaning to generate row identifiers.")
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    # Start with graph features as base
    df_merged = df_graph.copy()
    
    # Metadata columns should not be duplicated during merge
    metadata_cols = [
        'email_id', 'date', 'label', 'sender', 'receiver', 'subject', 'body',
        'subject_clean', 'body_clean', 'sender_domain', 'timezone_region',
        'full_text', 'cleaned_text', 'urls'
    ]
    metadata_cols = list(dict.fromkeys(metadata_cols))
    
    def _feature_count(frame: pd.DataFrame) -> int:
        return sum(1 for col in frame.columns if col not in metadata_cols)
    
    current_cols = set(df_merged.columns)
    
    # Merge helper
    def _merge_features(base_df: pd.DataFrame, new_df: pd.DataFrame):
        feature_cols = [
            col for col in new_df.columns
            if col not in metadata_cols and col not in current_cols
        ]
        if not feature_cols:
            return base_df, feature_cols
        subset = new_df[[key_col] + feature_cols]
        merged = base_df.merge(
            subset,
            on=key_col,
            how='inner',
            validate='one_to_one'
        )
        current_cols.update(feature_cols)
        return merged, feature_cols
    
    df_merged, ts_feature_cols = _merge_features(df_merged, df_ts)
    df_merged, text_feature_cols = _merge_features(df_merged, df_text)
    
    print(f"  • Graph features: {_feature_count(df_graph)}")
    print(f"  • Time series features: {len(ts_feature_cols)}")
    print(f"  • Text features: {len(text_feature_cols)}")
    print(f"  • Total features: {_feature_count(df_merged)}")
    
    # Ensure chronological order after merging
    df_merged = df_merged.sort_values('date').reset_index(drop=True)
    
    # Save merged features
    merged_path = os.path.join(output_dir, "engineered_features.csv")
    df_merged.to_csv(merged_path, index=False)
    print(f"\n💾 Merged features saved to: {merged_path}")
    
    return df_merged


def step_3_model_training(
    df: pd.DataFrame,
    perform_tuning: bool = True,
    n_iter: int = 20
) -> pd.DataFrame:
    """
    Step 3: Model Training
    Train baseline models, tune hyperparameters, train ensemble
    """
    print_header("STEP 3: MODEL TRAINING")
    
    # Define feature columns
    metadata_cols = ['email_id', 'date', 'label', 'sender', 'receiver', 'subject', 'body',
                     'subject_clean', 'body_clean',
                     'full_text', 'cleaned_text', 'sender_domain', 'timezone_region']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    print(f"\n📊 Dataset info:")
    print(f"  • Total samples: {len(df):,}")
    print(f"  • Total features: {len(feature_cols)}")
    print(f"  • Spam rate: {df['label'].mean()*100:.2f}%")
    
    # Initialize trainer
    trainer = ModelTrainer(verbose=True, random_state=42)
    
    # Prepare data
    X_train, X_test, y_train, y_test, _, _ = trainer.prepare_data(
        df, feature_cols, test_size=0.2
    )
    
    # Train baseline models
    baseline_df = trainer.train_baseline_models(X_train, y_train, X_test, y_test)
    
    # Hyperparameter tuning
    if perform_tuning:
        # Select top 3 baseline models to tune
        top_models = baseline_df.head(3)['Model'].tolist()
        print(f"\n📈 Tuning top {len(top_models)} models: {top_models}")
        
        tuned_df = trainer.tune_hyperparameters(
            X_train, y_train, X_test, y_test,
            models_to_tune=top_models,
            n_iter=n_iter,
            cv_splits=3
        )
    else:
        print(f"\n⏭️  Skipping hyperparameter tuning")
        trainer.tuned_models = trainer.baseline_models
        trainer.tuned_results = trainer.baseline_results.copy()
    
    # Train stacking ensemble
    ensemble_metrics, ensemble_pred = trainer.train_stacking_ensemble(
        X_train, y_train, X_test, y_test,
        use_tuned=perform_tuning,
        val_split=0.2
    )
    
    # Compare all models
    final_df = trainer.compare_all_models(
        include_ensemble=True,
        ensemble_metrics=ensemble_metrics
    )
    
    # Save results
    trainer.save_results(final_df)
    
    return final_df


def main():
    """Main pipeline execution."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run complete spam detection pipeline')
    parser.add_argument('--skip-cleaning', action='store_true', 
                       help='Skip data cleaning step')
    parser.add_argument('--skip-features', action='store_true',
                       help='Skip feature engineering step')
    parser.add_argument('--skip-tuning', action='store_true',
                       help='Skip hyperparameter tuning')
    parser.add_argument('--tune-iter', type=int, default=20,
                       help='Number of random search iterations for tuning')
    
    args = parser.parse_args()
    
    # Print pipeline configuration
    print_header("SPAM DETECTION - COMPLETE PIPELINE", "=")
    print("\n📋 Pipeline Configuration:")
    print(f"  • Data Cleaning: {'Skipped' if args.skip_cleaning else 'Enabled'}")
    print(f"  • Feature Engineering: {'Skipped' if args.skip_features else 'Enabled'}")
    print(f"  • Hyperparameter Tuning: {'Disabled' if args.skip_tuning else f'Enabled ({args.tune_iter} iterations)'}")
    
    pipeline_start = time.time()
    
    try:
        # Step 1: Data Cleaning
        df_cleaned = step_1_data_cleaning(
            skip=args.skip_cleaning
        )
        
        # Step 2: Feature Engineering
        df_features = step_2_feature_engineering(
            df_cleaned,
            skip=args.skip_features
        )
        
        # Step 3: Model Training
        results_df = step_3_model_training(
            df_features,
            perform_tuning=not args.skip_tuning,
            n_iter=args.tune_iter
        )
        
        # Final summary
        pipeline_elapsed = time.time() - pipeline_start
        
        print_header("PIPELINE COMPLETE!", "=")
        print(f"\n⏱️  Total pipeline time: {pipeline_elapsed:.2f} seconds ({pipeline_elapsed/60:.2f} minutes)")
        print(f"\n📊 Final Results:")
        print(results_df[['Model', 'Type', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].head(5).to_string(index=False))
        
        print(f"\n📁 Output Files:")
        print(f"  • Cleaned data: data/processed/cleaned_date_merge.csv")
        print(f"  • Graph features: data/processed/graph_features_pit.csv")
        print(f"  • Time series features: data/processed/timeseries_features_pit.csv")
        print(f"  • Text features: data/processed/text_features_pit.csv")
        print(f"  • Merged features: data/processed/engineered_features.csv")
        print(f"  • Model results: reports/model_results_finalized.csv")
        
        print("\n✅ Pipeline executed successfully!")
        print("🎉 Ready for deployment!")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Pipeline failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
