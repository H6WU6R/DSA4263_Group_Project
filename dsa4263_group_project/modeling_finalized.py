# Model Training Pipeline with Baseline Tracking and Hyperparameter Tuning
from dsa4263_group_project.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import warnings
from typing import Dict, Tuple, List, Optional
import time

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Template for model training with baseline tracking and hyperparameter tuning.
    Includes:
      - Baseline model training (default parameters)
      - Hyperparameter tuning with time-based validation
      - Multiple model types (Logistic, RF, XGB, LightGBM, etc.)
      - Ridge stacking ensemble
      - Performance tracking and comparison
    """
    
    def __init__(self, verbose: bool = True, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            verbose: Whether to print progress messages
            random_state: Random seed for reproducibility
        """
        self.verbose = verbose
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Track baseline and tuned results
        self.baseline_results = []
        self.tuned_results = []
        self.baseline_models = {}
        self.tuned_models = {}
        
        # Check optional libraries
        self.has_xgb = self._check_library('xgboost')
        self.has_lgbm = self._check_library('lightgbm')
    
    def _check_library(self, library_name: str) -> bool:
        """Check if optional library is available."""
        try:
            __import__(library_name)
            return True
        except ImportError:
            return False
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    # ============================================================================
    # DATA PREPARATION
    # ============================================================================
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = 'label',
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data with random train-test split and scaling.
        
        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names
            label_col: Name of label column
            test_size: Proportion of data for testing
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self._log("\n" + "="*80)
        self._log("DATA PREPARATION (Random Split)")
        self._log("="*80)
        
        # Extract features and labels
        X = df[feature_cols].values
        y = df[label_col].values
        
        # Random train/test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Handle NaN/inf values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        self._log(f"\nDataset split:")
        self._log(f"  • Total samples: {len(X):,}")
        self._log(f"  • Training samples: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
        self._log(f"  • Test samples: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
        self._log(f"\nLabel distribution:")
        self._log(f"  • Training spam rate: {y_train.mean()*100:.2f}%")
        self._log(f"  • Test spam rate: {y_test.mean()*100:.2f}%")
        self._log(f"\n✓ Data prepared and scaled")
        
        return X_train, X_test, y_train, y_test
    
    # ============================================================================
    # MODEL EVALUATION
    # ============================================================================
    
    def _evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        is_baseline: bool = True
    ) -> Dict:
        """
        Evaluate model and return metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of model
            is_baseline: Whether this is baseline or tuned model
        
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'Model': model_name,
            'Type': 'Baseline' if is_baseline else 'Tuned',
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Add AUC if model has predict_proba
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics['AUC'] = roc_auc_score(y_test, y_proba)
            except:
                metrics['AUC'] = None
        else:
            metrics['AUC'] = None
        
        return metrics
    
    # ============================================================================
    # BASELINE MODELS
    # ============================================================================
    
    def train_baseline_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Train baseline models with default parameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
        
        Returns:
            DataFrame with baseline results
        """
        self._log("\n" + "="*80)
        self._log("BASELINE MODEL TRAINING (DEFAULT PARAMETERS)")
        self._log("="*80)
        
        # Define baseline models
        baseline_models_dict = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(
                random_state=self.random_state,
                probability=True
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            # 'Gradient Boosting': GradientBoostingClassifier(
            #     random_state=self.random_state
            # )
        }
        
        # Add XGBoost if available
        if self.has_xgb:
            import xgboost as xgb
            baseline_models_dict['XGBoost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
        
        # Add LightGBM if available
        if self.has_lgbm:
            import lightgbm as lgb
            baseline_models_dict['LightGBM'] = lgb.LGBMClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        
        # Train and evaluate each baseline model
        self.baseline_results = []
        self.baseline_models = {}
        
        for model_name, model in baseline_models_dict.items():
            self._log(f"\n  • Training {model_name}...")
            start_time = time.time()
            
            model.fit(X_train, y_train)
            
            elapsed = time.time() - start_time
            self._log(f"    ✓ Trained in {elapsed:.2f}s")
            
            # Evaluate
            metrics = self._evaluate_model(model, X_test, y_test, model_name, is_baseline=True)
            metrics['Training_Time'] = elapsed
            
            self.baseline_results.append(metrics)
            self.baseline_models[model_name] = model
            
            self._log(f"    F1-Score: {metrics['F1-Score']:.4f}, Accuracy: {metrics['Accuracy']:.4f}")
        
        results_df = pd.DataFrame(self.baseline_results)
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        self._log("\n" + "="*80)
        self._log("BASELINE RESULTS SUMMARY")
        self._log("="*80)
        self._log("\n" + results_df.to_string(index=False))
        
        return results_df
    
    # ============================================================================
    # HYPERPARAMETER TUNING
    # ============================================================================
    
    def _get_param_grid(self, model_name: str) -> Dict:
        """
        Get parameter grid for hyperparameter tuning.
        
        Args:
            model_name: Name of model
        
        Returns:
            Dictionary of parameter distributions
        """
        param_grids = {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'saga']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            # 'Gradient Boosting': {
            #     'n_estimators': [50, 100, 200],
            #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
            #     'max_depth': [3, 5, 7],
            #     'min_samples_split': [2, 5, 10],
            #     'subsample': [0.8, 0.9, 1.0]
            # },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            },
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
        
        if self.has_xgb:
            param_grids['XGBoost'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        if self.has_lgbm:
            param_grids['LightGBM'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, -1],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        return param_grids.get(model_name, {})
    
    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        models_to_tune: Optional[List[str]] = None,
        n_iter: int = 20,
        cv_splits: int = 3
    ) -> pd.DataFrame:
        """
        Tune hyperparameters using RandomizedSearchCV with random cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            models_to_tune: List of model names to tune (None = tune all)
            n_iter: Number of random search iterations
            cv_splits: Number of CV splits (default: 3)
        
        Returns:
            DataFrame with tuned model results
        """
        self._log("\n" + "="*80)
        self._log("HYPERPARAMETER TUNING (RANDOMIZED SEARCH)")
        self._log("="*80)
        self._log(f"\n  • Using {cv_splits}-fold random CV")
        self._log(f"  • Random search iterations: {n_iter}")
        self._log(f"  • Scoring metric: F1-score")
        
        if models_to_tune is None:
            models_to_tune = list(self.baseline_models.keys())
        
        self.tuned_results = []
        self.tuned_models = {}
        
        for model_name in models_to_tune:
            if model_name not in self.baseline_models:
                self._log(f"\n  ⚠️  Skipping {model_name} (not in baseline models)")
                continue
            
            self._log(f"\n  • Tuning {model_name}...")
            start_time = time.time()
            
            # Get base model and param grid
            base_model = self.baseline_models[model_name]
            param_grid = self._get_param_grid(model_name)
            
            if not param_grid:
                self._log(f"    ⚠️  No parameter grid defined, using baseline model")
                self.tuned_models[model_name] = base_model
                # Copy baseline results
                baseline_result = [r for r in self.baseline_results if r['Model'] == model_name][0]
                tuned_result = baseline_result.copy()
                tuned_result['Type'] = 'Tuned'
                self.tuned_results.append(tuned_result)
                continue
            
            # Perform random search
            try:
                random_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    scoring='f1',
                    cv=cv_splits,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=0
                )
                
                random_search.fit(X_train, y_train)
                
                elapsed = time.time() - start_time
                
                # Get best model
                best_model = random_search.best_estimator_
                self.tuned_models[model_name] = best_model
                
                self._log(f"    ✓ Tuned in {elapsed:.2f}s")
                self._log(f"    Best parameters: {random_search.best_params_}")
                
                # Evaluate tuned model
                metrics = self._evaluate_model(best_model, X_test, y_test, model_name, is_baseline=False)
                metrics['Training_Time'] = elapsed
                
                self.tuned_results.append(metrics)
                
                # Compare with baseline
                baseline_f1 = [r['F1-Score'] for r in self.baseline_results if r['Model'] == model_name][0]
                improvement = metrics['F1-Score'] - baseline_f1
                self._log(f"    F1-Score: {metrics['F1-Score']:.4f} (baseline: {baseline_f1:.4f}, "
                         f"improvement: {improvement:+.4f})")
            
            except Exception as e:
                self._log(f"    ✗ Tuning failed: {e}")
                self._log(f"    Using baseline model instead")
                self.tuned_models[model_name] = base_model
                baseline_result = [r for r in self.baseline_results if r['Model'] == model_name][0]
                tuned_result = baseline_result.copy()
                tuned_result['Type'] = 'Tuned'
                self.tuned_results.append(tuned_result)
        
        results_df = pd.DataFrame(self.tuned_results)
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        self._log("\n" + "="*80)
        self._log("TUNED RESULTS SUMMARY")
        self._log("="*80)
        self._log("\n" + results_df.to_string(index=False))
        
        return results_df
    
    # ============================================================================
    # STACKING ENSEMBLE
    # ============================================================================
    
    def train_stacking_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        use_tuned: bool = True,
        val_split: float = 0.2
    ) -> Tuple[Dict, np.ndarray]:
        """
        Train Ridge stacking ensemble with proper validation split.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            use_tuned: Whether to use tuned models (True) or baseline (False)
            val_split: Proportion of training data for meta-validation
        
        Returns:
            Tuple of (metrics dict, predictions)
        """
        self._log("\n" + "="*80)
        self._log("STACKING ENSEMBLE (LOGISTIC REGRESSION META-LEARNER)")
        self._log("="*80)
        
        models_dict = self.tuned_models if use_tuned else self.baseline_models
        model_type = "tuned" if use_tuned else "baseline"
        
        self._log(f"\n  • Using {model_type} models as base learners")
        self._log(f"  • Meta-learner: Logistic Regression")
        self._log(f"  • Validation split: {val_split*100:.0f}% for meta-training")
        
        # Split training data for meta-learning (maintain temporal order)
        n_train = len(X_train)
        meta_train_size = int(n_train * (1 - val_split))
        
        X_meta_train = X_train[:meta_train_size]
        y_meta_train = y_train[:meta_train_size]
        X_meta_val = X_train[meta_train_size:]
        y_meta_val = y_train[meta_train_size:]
        
        self._log(f"\n  Split details:")
        self._log(f"    • Meta-train: {len(X_meta_train):,} samples")
        self._log(f"    • Meta-validation: {len(X_meta_val):,} samples")
        self._log(f"    • Final test: {len(X_test):,} samples")
        
        # Retrain base models on meta-train set only
        self._log("\n  • Retraining base models on meta-train set...")
        meta_train_models = {}
        
        for model_name, original_model in models_dict.items():
            # Clone model with same parameters
            from sklearn.base import clone
            model = clone(original_model)
            model.fit(X_meta_train, y_meta_train)
            meta_train_models[model_name] = model
        
        # Generate meta-features for validation set
        self._log("  • Generating meta-features for validation set...")
        meta_val_features = np.column_stack([
            model.predict_proba(X_meta_val)[:, 1] if hasattr(model, 'predict_proba')
            else model.predict(X_meta_val)
            for model in meta_train_models.values()
        ])
        
        # Train meta-learner on validation predictions
        self._log("  • Training Logistic Regression meta-learner...")
        meta_learner = LogisticRegression(max_iter=1000, random_state=self.random_state)
        meta_learner.fit(meta_val_features, y_meta_val)
        
        # Show meta-learner coefficients
        self._log("\n  Meta-learner coefficients:")
        for model_name, coef in zip(meta_train_models.keys(), meta_learner.coef_):
            self._log(f"    • {model_name:25s}: {coef:+.6f}")
        
        coef_std = np.std(meta_learner.coef_)
        self._log(f"\n  Coefficient statistics:")
        self._log(f"    • Std dev: {coef_std:.6f}")
        self._log(f"    • Diversity score: {coef_std:.6f}")
        
        # Generate final test predictions
        self._log("\n  • Generating final test predictions...")
        test_meta_features = np.column_stack([
            model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba')
            else model.predict(X_test)
            for model in meta_train_models.values()
        ])
        
        test_pred_proba = meta_learner.predict(test_meta_features)
        test_pred = (test_pred_proba >= 0.5).astype(int)
        
        # Evaluate
        metrics = {
            'Model': 'Logistic Stacking',
            'Type': f'Ensemble ({model_type})',
            'Accuracy': accuracy_score(y_test, test_pred),
            'Precision': precision_score(y_test, test_pred, zero_division=0),
            'Recall': recall_score(y_test, test_pred, zero_division=0),
            'F1-Score': f1_score(y_test, test_pred, zero_division=0),
            'AUC': roc_auc_score(y_test, test_pred_proba)
        }
        
        self._log("\n  ✓ Stacking ensemble trained successfully")
        self._log(f"    F1-Score: {metrics['F1-Score']:.4f}")
        
        return metrics, test_pred
    
    # ============================================================================
    # COMPARISON AND REPORTING
    # ============================================================================
    
    def compare_all_models(
        self,
        include_ensemble: bool = True,
        ensemble_metrics: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Compare baseline, tuned, and ensemble models.
        
        Args:
            include_ensemble: Whether to include ensemble results
            ensemble_metrics: Ensemble metrics dictionary
        
        Returns:
            DataFrame with all model comparisons
        """
        self._log("\n" + "="*80)
        self._log("FINAL MODEL COMPARISON")
        self._log("="*80)
        
        all_results = self.baseline_results + self.tuned_results
        
        if include_ensemble and ensemble_metrics:
            all_results.append(ensemble_metrics)
        
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        self._log("\n" + results_df.to_string(index=False))
        
        # Find best model
        best_model = results_df.iloc[0]
        self._log(f"\n{'='*80}")
        self._log(f"🏆 BEST MODEL: {best_model['Model']} ({best_model['Type']})")
        self._log(f"{'='*80}")
        self._log(f"  • Accuracy:  {best_model['Accuracy']:.4f}")
        self._log(f"  • Precision: {best_model['Precision']:.4f}")
        self._log(f"  • Recall:    {best_model['Recall']:.4f}")
        self._log(f"  • F1-Score:  {best_model['F1-Score']:.4f}")
        if best_model.get('AUC'):
            self._log(f"  • AUC:       {best_model['AUC']:.4f}")
        
        return results_df
    
    def save_results(
        self,
        results_df: pd.DataFrame,
        filename: str = "model_results_finalized.csv"
    ):
        """
        Save model results to processed directory using config path.
        Args:
            results_df: DataFrame with all results
            filename: Name of output file
        """
        results_df.to_csv(PROCESSED_DATA_DIR / filename, index=False)
        self._log(f"\n💾 Results saved to: {PROCESSED_DATA_DIR / filename}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../dsa4263_group_project')
    
    print("\n" + "="*80)
    print("MODEL TRAINER DEMO")
    print("="*80)
    
    # Load merged features (assuming already created)
    df = pd.read_csv("../data/processed/engineered_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # Define feature columns (exclude metadata)
    exclude_cols = ['date', 'label', 'sender', 'receiver', 'subject', 'body', 
                   'full_text', 'cleaned_text', 'sender_domain', 'timezone_region']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\nDataset loaded: {len(df):,} samples, {len(feature_cols)} features")
    
    # Initialize trainer
    trainer = ModelTrainer(verbose=True, random_state=42)
    
    try:
            df = pd.read_csv(PROCESSED_DATA_DIR / "engineered_features.csv")
            df['date'] = pd.to_datetime(df['date'])
            # Define feature columns (exclude metadata)
            exclude_cols = ['date', 'label', 'sender', 'receiver', 'subject', 'body', 
                           'full_text', 'cleaned_text', 'sender_domain', 'timezone_region']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            print(f"\nDataset loaded: {len(df):,} samples, {len(feature_cols)} features")
            # Initialize trainer
            trainer = ModelTrainer(verbose=True, random_state=42)
            # Prepare data
            X_train, X_test, y_train, y_test = trainer.prepare_data(
                df, feature_cols, test_size=0.2
            )
            # Train baseline models
            baseline_df = trainer.train_baseline_models(X_train, y_train, X_test, y_test)
            # Tune hyperparameters (on subset of models for demo)
            tuned_df = trainer.tune_hyperparameters(
                X_train, y_train, X_test, y_test,
                models_to_tune=['Logistic Regression', 'Random Forest'],
                n_iter=10,
                cv_splits=3
            )
            # Train stacking ensemble
            ensemble_metrics, ensemble_pred = trainer.train_stacking_ensemble(
                X_train, y_train, X_test, y_test,
                use_tuned=True,
                val_split=0.2
            )
            # Compare all models
            final_df = trainer.compare_all_models(
                include_ensemble=True,
                ensemble_metrics=ensemble_metrics
            )
            # Save results
            trainer.save_results(final_df)
            print("\n✅ Model training demo complete!")
    except FileNotFoundError:
        print("\n⚠️  Feature file not found. Please run feature engineering first.")
        print(f"   Expected: {PROCESSED_DATA_DIR / 'engineered_features.csv'}")
        print("\n⚠️  Feature file not found. Please run feature engineering first.")
