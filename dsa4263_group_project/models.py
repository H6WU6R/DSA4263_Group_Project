"""
Model Training Module

This module contains functions for training machine learning models,
including hyperparameter tuning and model comparison.
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Optional


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    has_lgbm: bool = False,
    has_xgb: bool = False,
    has_tuning: bool = True,
    verbose: bool = True
) -> Tuple[Dict, Optional[Dict], Optional[Dict]]:
    """
    Perform hyperparameter tuning for multiple models using RandomizedSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training labels
        has_lgbm: Whether LightGBM is available
        has_xgb: Whether XGBoost is available
        has_tuning: Whether to perform tuning
        verbose: Print progress messages
        
    Returns:
        Tuple of (best_rf_params, best_xgb_params, best_lgbm_params)
    """
    if not has_tuning:
        if verbose:
            print(f"\n[Training] Skipping hyperparameter tuning (using default parameters)...")
        
        best_rf_params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced'
        }
        best_xgb_params = {
            'n_estimators': 100,
            'max_depth': 7,
            'learning_rate': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0
        } if has_xgb else None
        best_lgbm_params = {
            'n_estimators': 100,
            'max_depth': 7,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 1.0
        } if has_lgbm else None
        
        return best_rf_params, best_xgb_params, best_lgbm_params
    
    from sklearn.model_selection import RandomizedSearchCV
    
    if verbose:
        print(f"\n[Training] Hyperparameter Tuning (RandomizedSearchCV)...")
        print(f"  ⏳ This may take a few minutes...")
    
    # Define parameter distributions
    rf_param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    # Tune Random Forest
    if verbose:
        print(f"\n  [Tuning] Random Forest...")
    rf_random = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=rf_param_dist,
        n_iter=10,
        cv=3,
        scoring='f1',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf_random.fit(X_train, y_train)
    best_rf_params = rf_random.best_params_
    if verbose:
        print(f"    ✓ Best params: {best_rf_params}")
        print(f"    ✓ Best CV F1-Score: {rf_random.best_score_:.4f}")
    
    # Tune XGBoost (if available)
    best_xgb_params = None
    if has_xgb:
        from xgboost import XGBClassifier
        
        xgb_param_dist = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [5, 7, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        
        if verbose:
            print(f"\n  [Tuning] XGBoost...")
        xgb_random = RandomizedSearchCV(
            XGBClassifier(random_state=42, tree_method='hist', n_jobs=-1),
            param_distributions=xgb_param_dist,
            n_iter=10,
            cv=3,
            scoring='f1',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        xgb_random.fit(X_train, y_train)
        best_xgb_params = xgb_random.best_params_
        if verbose:
            print(f"    ✓ Best params: {best_xgb_params}")
            print(f"    ✓ Best CV F1-Score: {xgb_random.best_score_:.4f}")
    
    # Tune LightGBM (if available)
    best_lgbm_params = None
    if has_lgbm:
        from lightgbm import LGBMClassifier
        
        lgbm_param_dist = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [5, 7, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [20, 31, 50],
            'subsample': [0.6, 0.8, 1.0]
        }
        
        if verbose:
            print(f"\n  [Tuning] LightGBM...")
        lgbm_random = RandomizedSearchCV(
            LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
            param_distributions=lgbm_param_dist,
            n_iter=10,
            cv=3,
            scoring='f1',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        lgbm_random.fit(X_train, y_train)
        best_lgbm_params = lgbm_random.best_params_
        if verbose:
            print(f"    ✓ Best params: {best_lgbm_params}")
            print(f"    ✓ Best CV F1-Score: {lgbm_random.best_score_:.4f}")
    
    if verbose:
        print(f"\n  ✅ Hyperparameter tuning complete!")
    
    return best_rf_params, best_xgb_params, best_lgbm_params


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict,
    verbose: bool = True
) -> Tuple[object, np.ndarray, Dict]:
    """Train Random Forest model."""
    if verbose:
        print(f"\n[Training] Model 1: Random Forest...")
    
    start_time = time.time()
    
    rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    train_time = time.time() - start_time
    
    results = {
        'Model': 'Random Forest',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Training Time (s)': train_time
    }
    
    if verbose:
        print(f"  ✓ Accuracy:  {results['Accuracy']:.4f}")
        print(f"  ✓ F1-Score:  {results['F1-Score']:.4f}")
        print(f"  ✓ Time:      {train_time:.2f}s")
    
    return rf, y_pred, results


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True
) -> Tuple[object, np.ndarray, Dict]:
    """Train Logistic Regression model."""
    if verbose:
        print(f"\n[Training] Model 2: Logistic Regression...")
    
    start_time = time.time()
    
    lr = LogisticRegression(
        max_iter=1000,
        solver='saga',
        penalty='l2',
        C=1.0,
        random_state=42,
        n_jobs=-1
    )
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    train_time = time.time() - start_time
    
    results = {
        'Model': 'Logistic Regression',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Training Time (s)': train_time
    }
    
    if verbose:
        print(f"  ✓ Accuracy:  {results['Accuracy']:.4f}")
        print(f"  ✓ F1-Score:  {results['F1-Score']:.4f}")
        print(f"  ✓ Time:      {train_time:.2f}s")
    
    return lr, y_pred, results


def train_naive_bayes(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True
) -> Tuple[object, np.ndarray, Dict]:
    """Train Naive Bayes model (with feature shifting for non-negative values)."""
    if verbose:
        print(f"\n[Training] Model 3: Naive Bayes...")
    
    start_time = time.time()
    
    # Naive Bayes requires non-negative features - shift if needed
    X_train_nb = X_train.copy()
    X_test_nb = X_test.copy()
    
    min_vals = X_train_nb.min()
    for col in X_train_nb.columns:
        if min_vals[col] < 0:
            shift = abs(min_vals[col]) + 0.01
            X_train_nb[col] += shift
            X_test_nb[col] += shift
    
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_nb, y_train)
    y_pred = nb.predict(X_test_nb)
    
    train_time = time.time() - start_time
    
    results = {
        'Model': 'Naive Bayes',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Training Time (s)': train_time
    }
    
    if verbose:
        print(f"  ✓ Accuracy:  {results['Accuracy']:.4f}")
        print(f"  ✓ F1-Score:  {results['F1-Score']:.4f}")
        print(f"  ✓ Time:      {train_time:.2f}s")
    
    return nb, y_pred, results


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict,
    verbose: bool = True
) -> Tuple[object, np.ndarray, Dict]:
    """Train LightGBM model."""
    from lightgbm import LGBMClassifier
    
    if verbose:
        print(f"\n[Training] Model 4: LightGBM...")
    
    start_time = time.time()
    
    lgbm = LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    
    train_time = time.time() - start_time
    
    results = {
        'Model': 'LightGBM',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Training Time (s)': train_time
    }
    
    if verbose:
        print(f"  ✓ Accuracy:  {results['Accuracy']:.4f}")
        print(f"  ✓ F1-Score:  {results['F1-Score']:.4f}")
        print(f"  ✓ Time:      {train_time:.2f}s")
    
    return lgbm, y_pred, results


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict,
    verbose: bool = True
) -> Tuple[object, np.ndarray, Dict]:
    """Train XGBoost model."""
    from xgboost import XGBClassifier
    
    if verbose:
        print(f"\n[Training] Model 5: XGBoost...")
    
    start_time = time.time()
    
    xgb = XGBClassifier(**params, tree_method='hist', random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    
    train_time = time.time() - start_time
    
    results = {
        'Model': 'XGBoost',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Training Time (s)': train_time
    }
    
    if verbose:
        print(f"  ✓ Accuracy:  {results['Accuracy']:.4f}")
        print(f"  ✓ F1-Score:  {results['F1-Score']:.4f}")
        print(f"  ✓ Time:      {train_time:.2f}s")
    
    return xgb, y_pred, results


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_rf_params: Dict,
    best_xgb_params: Optional[Dict],
    best_lgbm_params: Optional[Dict],
    has_lgbm: bool = False,
    has_xgb: bool = False,
    verbose: bool = True
) -> Tuple[List[Dict], Dict[str, Tuple]]:
    """
    Train all available models and return results.
    
    Args:
        X_train, y_train, X_test, y_test: Train/test data
        best_rf_params, best_xgb_params, best_lgbm_params: Tuned parameters
        has_lgbm, has_xgb: Model availability flags
        verbose: Print progress
        
    Returns:
        Tuple of (model_results list, model_objects dict)
    """
    model_results = []
    model_objects = {}
    
    # Model 1: Random Forest
    rf, y_pred_rf, rf_results = train_random_forest(
        X_train, y_train, X_test, y_test, best_rf_params, verbose
    )
    model_results.append(rf_results)
    model_objects['Random Forest'] = (rf, y_pred_rf)
    
    # Model 2: Logistic Regression
    lr, y_pred_lr, lr_results = train_logistic_regression(
        X_train, y_train, X_test, y_test, verbose
    )
    model_results.append(lr_results)
    model_objects['Logistic Regression'] = (lr, y_pred_lr)
    
    # Model 3: Naive Bayes
    nb, y_pred_nb, nb_results = train_naive_bayes(
        X_train, y_train, X_test, y_test, verbose
    )
    model_results.append(nb_results)
    model_objects['Naive Bayes'] = (nb, y_pred_nb)
    
    # Model 4: LightGBM (if available)
    if has_lgbm and best_lgbm_params is not None:
        lgbm, y_pred_lgbm, lgbm_results = train_lightgbm(
            X_train, y_train, X_test, y_test, best_lgbm_params, verbose
        )
        model_results.append(lgbm_results)
        model_objects['LightGBM'] = (lgbm, y_pred_lgbm)
    elif verbose:
        print(f"\n[Training] Skipping LightGBM (not available)")
    
    # Model 5: XGBoost (if available)
    if has_xgb and best_xgb_params is not None:
        xgb, y_pred_xgb, xgb_results = train_xgboost(
            X_train, y_train, X_test, y_test, best_xgb_params, verbose
        )
        model_results.append(xgb_results)
        model_objects['XGBoost'] = (xgb, y_pred_xgb)
    elif verbose:
        print(f"\n[Training] Skipping XGBoost (not available)")
    
    return model_results, model_objects


def select_best_model(
    model_results: List[Dict],
    model_objects: Dict[str, Tuple],
    verbose: bool = True
) -> Tuple[Dict, object, np.ndarray]:
    """
    Select the best model based on F1-score.
    
    Args:
        model_results: List of model result dictionaries
        model_objects: Dictionary of {model_name: (model_obj, predictions)}
        verbose: Print progress
        
    Returns:
        Tuple of (best_model_dict, best_model_obj, best_predictions)
    """
    results_df = pd.DataFrame(model_results)
    best_idx = results_df['F1-Score'].idxmax()
    best_model = results_df.iloc[best_idx]
    
    if verbose:
        print(f"\n{'='*80}")
        print(" BEST MODEL")
        print("="*80)
        print(f"  Model:      {best_model['Model']}")
        print(f"  Accuracy:   {best_model['Accuracy']:.4f}")
        print(f"  Precision:  {best_model['Precision']:.4f}")
        print(f"  Recall:     {best_model['Recall']:.4f}")
        print(f"  F1-Score:   {best_model['F1-Score']:.4f}")
        print(f"  Time:       {best_model['Training Time (s)']:.2f}s")
    
    # Get best model object and predictions
    best_model_obj, best_predictions = model_objects[best_model['Model']]
    
    return best_model.to_dict(), best_model_obj, best_predictions
