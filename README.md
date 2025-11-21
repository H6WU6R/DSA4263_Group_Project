# DSA4263 Group Project: Email Spam Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning pipeline for email spam detection using graph-based, temporal, and text features with automated feature selection and ensemble modeling.

## Overview

This project implements an end-to-end machine learning pipeline for detecting spam emails using multi-modal features from email communication networks. The system combines:

- **Graph-based features**: Network topology analysis (degree centrality, clustering, eigenvector centrality)
- **Temporal features**: Time-series patterns and behavioral statistics
- **Text features**: NLP-based content analysis (sentiment, readability, linguistic patterns)

The pipeline includes automated feature selection through ablation studies and trains multiple machine learning models with hyperparameter tuning and stacking ensemble methods.

## Key Features

### Automated Pipeline
- **Point-in-Time (PIT) Safe**: All features respect temporal ordering to prevent data leakage
- **Parallel Processing**: Multiprocessing support for fast feature engineering
- **Ablation Studies**: Automatic feature group selection based on F1-Score
- **End-to-End**: From raw CSV files to trained models and visualizations

### Machine Learning
- **Multiple Models**: Logistic Regression, Random Forest, SVM, Naive Bayes, KNN, XGBoost, LightGBM
- **Hyperparameter Tuning**: RandomizedSearchCV with 3-fold cross-validation
- **Stacking Ensemble**: Logistic Regression meta-learner combining best models
- **Model Persistence**: Save/load trained models as pickle files

### Visualization & Analysis
- **Confusion Matrices**: Per-model performance visualization
- **Metrics Comparison**: Grouped bar charts for F1, Precision, Recall, Accuracy
- **Feature Statistics**: Comprehensive ablation study reports

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW DATA (CSV Files)                        │
│   CEAS_08, Nazario, Nigerian_Fraud, SpamAssasin                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1-3: DATA CLEANING                            │
│  • Merge datasets    • Clean dates    • Clean text (NLTK)       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         STEP 4-6: FEATURE ENGINEERING + ABLATION                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Graph        │  │ Timeseries   │  │ Text         │           │
│  │ Features     │  │ Features     │  │ Features     │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Ablation     │  │ Ablation     │  │ Ablation     │           │
│  │ Study        │  │ Study        │  │ Study        │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                  │                  │                 │
│         └──────────────────┴──────────────────┘                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 7: MERGE BEST FEATURES                        │
│      Select top-performing feature groups by F1-Score           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           STEP 8: MODEL TRAINING & EVALUATION                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  1. Baseline Training (5-7 models, default params)      │    │
│  │  2. Hyperparameter Tuning (RandomizedSearchCV)          │    │
│  │  3. Stacking Ensemble (Logistic meta-learner)           │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OUTPUTS & VISUALIZATIONS                       │
│  • Model results CSV   • Confusion matrices                     │
│  • Trained models      • Metrics comparison charts              │
│  • Feature summaries   • Ablation study reports                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip or [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/H6WU6R/DSA4263_Group_Project.git
cd DSA4263_Group_Project
```

2. **Install dependencies**

**Option A: Using uv (Recommended - Faster)**
```bash
# Install uv if you haven't already
pip install uv

# Sync dependencies from uv.lock
uv sync
```

**Option B: Using pip**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (automatic on first run, or manual)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

4. **Prepare data**
Place your raw email CSV files in `data/raw/`:
- `CEAS_08.csv`
- `Nazario.csv`
- `Nigerian_Fraud.csv`
- `SpamAssasin.csv`

Required columns: `sender`, `receiver`, `date`, `subject`, `body`, `label`

## 🎮 Quick Start

### Run Complete Pipeline

**Using uv:**
```bash
cd dsa4263_group_project
uv run python main_finalized.py
```

**Using standard Python:**
```bash
cd dsa4263_group_project
python main_finalized.py
```

This will execute all 8 steps:
1. Load and merge datasets
2. Clean dates
3. Clean text
4. Extract graph features + ablation
5. Extract timeseries features + ablation
6. Extract text features + ablation
7. Merge best feature groups
8. Train models and generate visualizations

### Expected Runtime
- Feature engineering: ~55-60 minutes (with parallelization)
- Model training: ~10-20 minutes (with hyperparameter tuning)
- Total: ~70-80 minutes (depending on dataset size)

### Using Individual Modules

```python
from data_cleaning_finalized import DataCleaner
from feature_engineering_finalized import FeatureEngineer
from modeling_finalized import ModelTrainer

# Data cleaning
cleaner = DataCleaner(stop_words=stop_words, lemmatizer=lemmatizer, stemmer=stemmer)
cleaned_df = cleaner.clean_and_merge(raw_datasets)

# Feature engineering
engineer = FeatureEngineer(verbose=True)
df_features = engineer.compute_graph_features_parallel(cleaned_df)

# Model training
trainer = ModelTrainer(verbose=True, random_state=42)
X_train, X_test, y_train, y_test = trainer.prepare_data(df_features, feature_cols)
baseline_df = trainer.train_baseline_models(X_train, y_train, X_test, y_test)
```

## 🗂️ Project Structure

```
DSA4263_Group_Project/
├── data/
│   ├── raw/                          # Raw email CSV files
│   ├── processed/                    # Cleaned and engineered features
│   │   ├── cleaned_date_merge.csv
│   │   ├── engineered_features_selected.csv
│   │   ├── graph_ablation_results.csv
│   │   ├── timeseries_ablation_results.csv
│   │   ├── text_ablation_results.csv
│   │   ├── feature_selection_summary.csv
│   │   └── model_results_finalized.csv
│   ├── interim/                      # Intermediate processing files
│   └── external/                     # External reference data
│
├── dsa4263_group_project/           # Main package
│   ├── __init__.py
│   ├── config.py                     # Path configurations
│   ├── data_cleaning_finalized.py    # Data cleaning module
│   ├── feature_engineering_finalized.py  # Feature engineering module
│   ├── modeling_finalized.py         # Model training module
│   └── main_finalized.py            # End-to-end pipeline script
│
├── models/                           # Saved trained models
│   ├── tuned/                       # Tuned model pickle files
│   │   ├── logistic_regression_tuned.pkl
│   │   ├── random_forest_tuned.pkl
│   │   └── ...
│   └── scaler.pkl                   # Feature scaler
│
├── reports/
│   └── figures/                     # Visualizations
│       ├── confusion_matrices.png
│       └── metrics_comparison.png
│
├── notebooks/                       # Jupyter notebooks for EDA
│   ├── graph_clean.ipynb
│   ├── nlp_eda_ablation.ipynb
│   └── sender_time_series_analysis.ipynb
│
├── requirements.txt                 # Python dependencies (pip)
├── pyproject.toml                   # Project metadata & uv config
├── uv.lock                          # Locked dependencies (uv)
├── LICENSE                          # MIT License
└── README.md                        # This file
```

## Feature Engineering

### Graph Features (Network Topology)
- **out_degree**: Number of unique recipients
- **in_degree**: Number of unique senders
- **total_sent**: Total emails sent
- **reciprocity**: Proportion of recipients who replied
- **clustering**: Clustering coefficient
- **eigenvector**: Eigenvector centrality
- **closeness**: Closeness centrality
- **avg_weight**: Average emails per recipient

### Timeseries Features (Temporal Patterns)
- **emails_last_7d**: Email count in last 7 days
- **emails_last_30d**: Email count in last 30 days
- **mean_interval**: Mean time between emails
- **std_interval**: Standard deviation of intervals
- **hour_of_day**: Most common sending hour
- **day_of_week**: Most common sending day
- **burstiness**: Email sending pattern burstiness

### Text Features (NLP Analysis)
- **body_length**: Character count
- **word_count**: Word count
- **avg_word_length**: Average word length
- **capital_ratio**: Uppercase letter ratio
- **special_char_ratio**: Special character ratio
- **url_count**: Number of URLs
- **sentiment**: VADER sentiment score
- **flesch_reading_ease**: Readability score
- **contains_urgent**: Urgency indicator words

### Ablation Studies
Each feature group is evaluated using Logistic Regression baseline:
- Train on individual feature groups
- Measure F1-Score, Precision, Recall, Accuracy
- Automatically select best-performing group
- Results saved to CSV for analysis

## 🤖 Model Training

### Models Supported
1. **Logistic Regression** (baseline + tuned)
2. **Random Forest** (baseline + tuned)
3. **Support Vector Machine (SVM)** (baseline + tuned)
4. **Naive Bayes** (baseline + tuned)
5. **K-Nearest Neighbors** (baseline + tuned)
6. **XGBoost** (optional, if installed)
7. **LightGBM** (optional, if installed)
8. **Stacking Ensemble** (combines tuned models)

### Training Process
1. **Data Preparation**: Random 80/20 train-test split with stratification
2. **Feature Scaling**: StandardScaler applied to all features
3. **Baseline Training**: Default hyperparameters for quick comparison
4. **Hyperparameter Tuning**: RandomizedSearchCV with 3-fold CV, 20 iterations
5. **Ensemble Stacking**: Logistic Regression meta-learner on validation predictions

## Results and Outputs

### CSV Files
- `model_results_finalized.csv`: Performance metrics for all models
- `feature_selection_summary.csv`: Best feature groups selected
- `graph_ablation_results.csv`: Graph feature ablation scores
- `timeseries_ablation_results.csv`: Timeseries feature ablation scores
- `text_ablation_results.csv`: Text feature ablation scores

### Visualizations
- `confusion_matrices.png`: Grid of confusion matrices for all models
- `metrics_comparison.png`: Grouped bar chart comparing F1/Precision/Recall/Accuracy

### Saved Models
- `models/tuned/*.pkl`: Trained model objects
- `models/scaler.pkl`: Feature scaler for inference

### Loading Saved Models
```python
import pickle

# Load model
with open('models/tuned/random_forest_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Make predictions
X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
```

## Package Management

This project supports both **uv** and **pip** for dependency management:

### Using uv (Recommended)
[uv](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver written in Rust.

**Benefits:**
- ⚡ 10-100x faster than pip
- 🔒 Deterministic builds with `uv.lock`
- 🎯 Better dependency resolution
- 📦 Built-in virtual environment management

**Common commands:**
```bash
# Sync dependencies
uv sync

# Add a new package
uv add package-name

# Run Python script with uv
uv run python script.py

# Update dependencies
uv sync --upgrade
```

### Using pip
Traditional pip workflow is fully supported:
```bash
pip install -r requirements.txt
```

## Configuration

### Path Configuration (`config.py`)
```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
```

### Model Configuration
Edit `modeling_finalized.py` to:
- Add/remove models from training
- Adjust hyperparameter grids
- Change CV splits or random search iterations
- Modify train-test split ratio

### Feature Configuration
Edit `feature_engineering_finalized.py` to:
- Enable/disable feature groups
- Add custom features
- Adjust parallelization workers
- Modify feature ablation settings

## Contributing

This project is part of DSA4263 course work. For contributions:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This project implements Point-in-Time (PIT) safe features to prevent data leakage and ensure realistic model evaluation. All temporal features respect the chronological ordering of emails.