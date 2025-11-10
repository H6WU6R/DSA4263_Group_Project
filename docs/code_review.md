# Code Review Findings

## Critical

1. **RandomizedSearchCV uses non-temporal CV, leaking future data**
   - **What/Why:** `RandomizedSearchCV` receives an integer `cv` argument, so it defaults to KFold/StratifiedKFold and each split trains on data from the future folds. This violates the PIT constraint the project documents. Models tuned this way are optimistically biased and will not generalise.
   - **Where:** `ModelTrainer.tune_hyperparameters` (`dsa4263_group_project/modeling_finalized.py`, lines 410-421).
   - **How to fix:** Replace the integer `cv` argument with an explicit `TimeSeriesSplit` (or custom splitter) so every validation fold only sees future examples and training folds never include future labels. Initialise the splitter once per tune call to keep determinism.
   - **Patch:**

````python
// filepath: dsa4263_group_project/modeling_finalized.py
from sklearn.model_selection import train_test_split, RandomizedSearchCV
+from sklearn.model_selection import TimeSeriesSplit
// ...existing code...
            # Perform random search
            try:
+                tscv = TimeSeriesSplit(n_splits=cv_splits)
                random_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    scoring='f1',
-                    cv=cv_splits,
+                    cv=tscv,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=0
                )
````

## Major

1. **Global fallback means leak future labels in temporal risk scores**
   - **What/Why:** `_compute_pit_risk_score` fills the first observation for each group with `df[value_col].mean()`, i.e. the overall label rate computed on the entire dataset (including future events). This injects look-ahead bias into every group’s first row.
   - **Where:** `FeatureEngineer._compute_pit_risk_score` (`dsa4263_group_project/feature_engineering_finalized.py`, lines 262-264).
   - **How to fix:** Use an expanding global prior that is shifted by one step so the fallback is derived only from already-seen rows (and falls back to a neutral constant for the very first row).
   - **Patch:**

````python
// filepath: dsa4263_group_project/feature_engineering_finalized.py
-                # Fill NaN (first occurrence of each group) with global mean
-                global_mean = df[value_col].mean()
-                risk_scores = risk_scores.fillna(global_mean)
+                # Fill NaN (first occurrence of each group) with a past-only global prior
+                global_prior = df[value_col].expanding().mean().shift(1).fillna(0.0)
+                risk_scores = risk_scores.fillna(global_prior)
````

2. **Sender temporal features use overall mean with future leakage**
   - **What/Why:** `_compute_sender_temporal_features` falls back to `df['label'].mean()` when a sender has no history. As above, that mean includes future samples, leaking the future spam rate into early rows.
   - **Where:** `FeatureEngineer._compute_sender_temporal_features` (`dsa4263_group_project/feature_engineering_finalized.py`, lines 290-296).
   - **How to fix:** Reuse a shifted expanding global prior (or an externally supplied training prior) so the fallback depends only on previously observed labels.
   - **Patch:**

````python
// filepath: dsa4263_group_project/feature_engineering_finalized.py
-                # Historical phishing rate
-                global_mean = df['label'].mean()
-                df['sender_historical_phishing_rate'] = np.where(
-                        df['sender_historical_count'] > 0,
-                        df['sender_historical_spam_count'] / df['sender_historical_count'],
-                        global_mean
-                )
+                # Historical phishing rate
+                global_prior = df['label'].expanding().mean().shift(1).fillna(0.0)
+                df['sender_historical_phishing_rate'] = np.where(
+                        df['sender_historical_count'] > 0,
+                        df['sender_historical_spam_count'] / df['sender_historical_count'],
+                        global_prior
+                )
````

3. **Domain features also leak through dataset-wide mean fallback**
   - **What/Why:** `_extract_domain_features` uses `df['label'].mean()` for domains with no history, again importing the dataset’s future-positive rate into the first appearance of every domain.
   - **Where:** `FeatureEngineer._extract_domain_features` (`dsa4263_group_project/feature_engineering_finalized.py`, lines 549-555).
   - **How to fix:** Swap in a shifted expanding global prior so “cold start” domains receive a neutral value derived only from past observations (or a hard-coded prior if preferred).
   - **Patch:**

````python
// filepath: dsa4263_group_project/feature_engineering_finalized.py
-                # Domain spam rate
-                global_spam_rate = df['label'].mean()
-                df['domain_spam_rate'] = np.where(
-                        df['domain_email_count'] > 0,
-                        df['domain_spam_cumsum'] / df['domain_email_count'],
-                        global_spam_rate
-                )
+                # Domain spam rate
+                global_prior = df['label'].expanding().mean().shift(1).fillna(0.0)
+                df['domain_spam_rate'] = np.where(
+                        df['domain_email_count'] > 0,
+                        df['domain_spam_cumsum'] / df['domain_email_count'],
+                        global_prior
+                )
````

## Minor

1. **Unknown timezones default to “Europe/Africa”**
   - **What/Why:** `_parse_timezone_offset` returns `0.0` when the offset is missing or malformed. `_get_simple_region` treats zero as “Europe/Africa”, so any email without a recognised offset is silently classified as European, skewing downstream regional features.
   - **Where:** `DataCleaner._parse_timezone_offset` (`dsa4263_group_project/data_cleaning_finalized.py`, lines 33-41).
   - **How to fix:** Return `pd.NA` (or `None`) for missing/invalid offsets so `_get_simple_region` can mark them as `Unknown` and downstream code can handle them explicitly.
   - **Patch:**

````python
// filepath: dsa4263_group_project/data_cleaning_finalized.py
-                if pd.isna(tz_str):
-                        return 0.0
+                if pd.isna(tz_str):
+                        return pd.NA
                 try:
                        sign = 1 if tz_str[0] == '+' else -1
                        hours = int(tz_str[1:3])
                        minutes = int(tz_str[3:5])
                        return sign * (hours + minutes / 60.0)
                 except (ValueError, IndexError):
-                        return 0.0
+                        return pd.NA
````

