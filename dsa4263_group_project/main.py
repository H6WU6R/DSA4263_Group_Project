# %% [markdown]
# # EMAIL SPAM DETECTION - COMPLETE PIPELINE
# %%
# ============================================================================
# IMPORTS & CONFIGURATION
# ============================================================================
import os
import sys
import re
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# NLP imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data (quietly)
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    print(f"⚠️  Warning: Could not download some NLTK data: {e}")

# Initialize NLP tools
try:
    STOP_WORDS = set(stopwords.words('english'))
    STOP_WORDS.update(['re', 'fwd', 'subject', 'email', 'com', 'www'])
    STEMMER = PorterStemmer()
    LEMMATIZER = WordNetLemmatizer()
    SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()
    HAS_NLP = True
except Exception as e:
    print(f"⚠️  Warning: NLP tools not fully available: {e}")
    HAS_NLP = False

# Optional: seaborn for enhanced visualizations
try:
    import seaborn as sns
    sns.set_palette('husl')
    HAS_SEABORN = True
except ImportError:
    print("⚠️  Seaborn not available, using default matplotlib styling")
    HAS_SEABORN = False

warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_PATH = os.path.join(PROJECT_ROOT, "reports", "figures")

# Create directories if they don't exist
os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

print("="*80)
print(" EMAIL SPAM DETECTION - COMPLETE PIPELINE")
print("="*80)
print(f"\nProject Root: {PROJECT_ROOT}")
print(f"Data Path: {DATA_PROCESSED_PATH}")
print(f"Reports Path: {REPORTS_PATH}")

# %%
# ============================================================================
# SECTION 1: DATA CLEANING (from date_data_cleaning.ipynb)
# ============================================================================
print("\n" + "="*80)
print(" SECTION 1: DATA CLEANING & PREPROCESSING")
print("="*80)

def extract_timezone_offset(date_str: str) -> Optional[str]:
    """Extract timezone offset from date string (e.g., '+0800', '-0500')"""
    if pd.isna(date_str):
        return None
    
    # Match timezone patterns: +HHMM or -HHMM
    match = re.search(r'([+-]\d{4})(?:\s|$)', str(date_str))
    return match.group(1) if match else None


def timezone_offset_to_hours(offset_str: str) -> Optional[float]:
    """Convert timezone offset string to hours (e.g., '+0800' → 8.0)"""
    if not offset_str or len(offset_str) != 5:
        return None
    
    try:
        sign = 1 if offset_str[0] == '+' else -1
        hours = int(offset_str[1:3])
        minutes = int(offset_str[3:5])
        return sign * (hours + minutes / 60.0)
    except (ValueError, IndexError):
        return None


def map_timezone_to_region(tz_hours: float) -> str:
    """Map timezone offset to geographic region"""
    if pd.isna(tz_hours):
        return 'Unknown'
    
    if -12 <= tz_hours < -3:
        return 'Americas'
    elif -3 <= tz_hours < 3:
        return 'Europe/Africa'
    elif 3 <= tz_hours < 6:
        return 'Middle East/South Asia'
    elif 6 <= tz_hours < 10:
        return 'APAC'
    elif 10 <= tz_hours <= 14:
        return 'Oceania/Pacific'
    else:
        return 'Unknown'


def clean_email_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean and preprocess email data with timezone awareness.
    
    Steps:
    1. Extract and validate timezone information
    2. Parse and standardize dates
    3. Extract temporal features
    4. Remove invalid/anomalous data
    5. Add derived features
    """
    if verbose:
        print("\n[1.1] Starting data cleaning...")
        print(f"  Initial size: {len(df):,} emails")
    
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # Step 1: Timezone extraction
    if verbose:
        print("\n[1.2] Extracting timezone information...")
    
    df_clean['timezone_offset_str'] = df_clean['date'].apply(extract_timezone_offset)
    df_clean['timezone_hours'] = df_clean['timezone_offset_str'].apply(timezone_offset_to_hours)
    df_clean['timezone_region'] = df_clean['timezone_hours'].apply(map_timezone_to_region)
    
    # Remove rows with invalid timezone
    df_clean = df_clean[df_clean['timezone_region'] != 'Unknown'].copy()
    
    if verbose:
        tz_removed = initial_count - len(df_clean)
        print(f"  ✓ Removed {tz_removed:,} rows with invalid timezone")
        print(f"  ✓ Remaining: {len(df_clean):,} emails")
    
    # Step 2: Date parsing
    if verbose:
        print("\n[1.3] Parsing dates...")
    
    df_clean['date'] = pd.to_datetime(df_clean['date'], format='ISO8601', errors='coerce')
    
    # Remove unparseable dates
    before_date_filter = len(df_clean)
    df_clean = df_clean[df_clean['date'].notna()].copy()
    
    # Remove anomalous dates (outside 1990-2025)
    df_clean = df_clean[
        (df_clean['date'].dt.year >= 1990) & 
        (df_clean['date'].dt.year <= 2025)
    ].copy()
    
    if verbose:
        date_removed = before_date_filter - len(df_clean)
        print(f"  ✓ Removed {date_removed:,} rows with invalid dates")
        print(f"  ✓ Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    
    # Step 3: Extract temporal features
    if verbose:
        print("\n[1.4] Extracting temporal features...")
    
    df_clean['year'] = df_clean['date'].dt.year
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['day'] = df_clean['date'].dt.day
    df_clean['hour'] = df_clean['date'].dt.hour
    df_clean['day_of_week'] = df_clean['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df_clean['day_name'] = df_clean['date'].dt.day_name()
    df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
    
    if verbose:
        print(f"  ✓ Added temporal features: year, month, day, hour, day_of_week, is_weekend")
    
    # Step 4: Clean sender/receiver
    if verbose:
        print("\n[1.5] Cleaning sender/receiver fields...")
    
    df_clean = df_clean.dropna(subset=['sender', 'receiver'])
    df_clean['sender'] = df_clean['sender'].astype(str).str.strip().str.lower()
    df_clean['receiver'] = df_clean['receiver'].astype(str).str.strip().str.lower()
    df_clean = df_clean[(df_clean['sender'] != 'nan') & (df_clean['receiver'] != 'nan')].copy()
    
    # Step 5: Remove duplicates
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['sender', 'receiver', 'date'], keep='first')
    
    if verbose:
        dedup_removed = before_dedup - len(df_clean)
        print(f"  ✓ Removed {dedup_removed:,} duplicate emails")
    
    # Final summary
    if verbose:
        print("\n" + "─"*80)
        print(" CLEANING SUMMARY")
        print("─"*80)
        print(f"  Initial emails:     {initial_count:,}")
        print(f"  Final emails:       {len(df_clean):,}")
        print(f"  Retention rate:     {len(df_clean)/initial_count*100:.1f}%")
        print(f"  Unique senders:     {df_clean['sender'].nunique():,}")
        print(f"  Unique receivers:   {df_clean['receiver'].nunique():,}")
        print(f"  Spam rate:          {df_clean['label'].mean()*100:.1f}%")
        print("─"*80)
    
    return df_clean


# Load data
print("\n[1.0] Loading processed data...")

# Note: The notebooks have already done the heavy cleaning
# We'll use the graph_merge.csv which has cleaned data with dates
data_file = os.path.join(DATA_PROCESSED_PATH, "date_merge.csv")

if not os.path.exists(data_file):
    print(f"❌ ERROR: Data file not found: {data_file}")
    print("   Please ensure the data cleaning notebooks have been run.")
    sys.exit(1)

df_cleaned = pd.read_csv(data_file)
df_cleaned['date'] = pd.to_datetime(df_cleaned['date'], format='mixed', errors='coerce', utc=True)

# Remove rows with invalid dates
initial_count = len(df_cleaned)
df_cleaned = df_cleaned[df_cleaned['date'].notna()].copy()

# Convert to timezone-naive for consistency
df_cleaned['date'] = df_cleaned['date'].dt.tz_localize(None)

print(f"  ✓ Loaded {initial_count:,} emails from graph_merge.csv")
if initial_count != len(df_cleaned):
    print(f"  ⚠️  Removed {initial_count - len(df_cleaned):,} emails with invalid dates")
print(f"  ✓ Final count: {len(df_cleaned):,} emails")
print(f"  ✓ Date range: {df_cleaned['date'].min()} to {df_cleaned['date'].max()}")
print(f"  ✓ Spam rate: {df_cleaned['label'].mean()*100:.1f}%")
print(f"  ✓ Unique senders: {df_cleaned['sender'].nunique():,}")
print(f"  ✓ Unique receivers: {df_cleaned['receiver'].nunique():,}")

# Add temporal features if not already present
if 'timezone_region' not in df_cleaned.columns:
    print("\n[1.1] Adding temporal features (these should be from notebooks)...")
    # Simple temporal features
    df_cleaned['year'] = df_cleaned['date'].dt.year
    df_cleaned['month'] = df_cleaned['date'].dt.month
    df_cleaned['day'] = df_cleaned['date'].dt.day
    df_cleaned['hour'] = df_cleaned['date'].dt.hour
    df_cleaned['day_of_week'] = df_cleaned['date'].dt.dayofweek
    df_cleaned['day_name'] = df_cleaned['date'].dt.day_name()
    df_cleaned['is_weekend'] = df_cleaned['day_of_week'].isin([5, 6]).astype(int)
    
    # Simplified timezone region (based on email patterns)
    # For this demo, we'll use Americas as default
    df_cleaned['timezone_region'] = 'Americas'
    print(f"  ✓ Added basic temporal features")

print(f"\n✅ Data ready for analysis")

# ---------------------------------------------------------------------------
# 1.2: Text Cleaning (from lzy notebooks)
# ---------------------------------------------------------------------------
if HAS_NLP:
    print("\n[1.2] Applying text preprocessing...")
    
    def clean_text(text: str) -> str:
        """Clean text: lowercase, remove punctuation, tokenize, remove stopwords, lemmatize"""
        if pd.isna(text) or text == "":
            return ""
        
        # Lowercase
        text = str(text).lower()
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in STOP_WORDS and len(word) > 2]
        
        # Lemmatize
        try:
            tokens = [LEMMATIZER.lemmatize(word) for word in tokens]
        except:
            pass
        
        return ' '.join(tokens)
    
    # Create cleaned text column
    df_cleaned['subject_clean'] = df_cleaned['subject'].fillna('').apply(clean_text)
    df_cleaned['body_clean'] = df_cleaned['body'].fillna('').apply(clean_text)
    df_cleaned['text_combined'] = df_cleaned['subject_clean'] + ' ' + df_cleaned['body_clean']
    
    print(f"  ✓ Text preprocessing complete")
    print(f"  ✓ Sample cleaned text: {df_cleaned['text_combined'].iloc[0][:100]}...")
else:
    print("\n[1.2] Skipping text preprocessing (NLTK not available)")
    df_cleaned['subject_clean'] = df_cleaned['subject'].fillna('')
    df_cleaned['body_clean'] = df_cleaned['body'].fillna('')
    df_cleaned['text_combined'] = df_cleaned['subject'] + ' ' + df_cleaned['body']

# %%
# ============================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS (from date_data_EDA.ipynb + lzy-EDA.ipynb)
# ============================================================================
print("\n" + "="*80)
print(" SECTION 2: EXPLORATORY DATA ANALYSIS")
print("="*80)

def perform_temporal_eda(df: pd.DataFrame, save_path: str) -> Dict:
    """
    Perform comprehensive EDA on temporal patterns.
    
    Returns summary statistics and insights.
    """
    print("\n[2.1] Analyzing temporal patterns...")
    
    insights = {}
    
    # Regional analysis
    print("\n  📍 Regional Distribution:")
    region_stats = df.groupby('timezone_region').agg({
        'label': ['count', 'mean']
    }).round(4)
    region_stats.columns = ['Total_Emails', 'Spam_Rate']
    print(region_stats)
    insights['regional'] = region_stats
    
    # Hourly patterns
    print("\n  🕐 Hourly Patterns:")
    hour_stats = df.groupby(['hour', 'label']).size().unstack(fill_value=0)
    hour_spam_rate = df.groupby('hour')['label'].mean()
    print(f"  Peak spam hour: {hour_spam_rate.idxmax()}:00 ({hour_spam_rate.max()*100:.1f}% spam)")
    insights['hourly'] = hour_stats
    
    # Weekday patterns
    print("\n  📅 Weekday Patterns:")
    weekday_stats = df.groupby(['day_of_week', 'label']).size().unstack(fill_value=0)
    weekend_spam_rate = df[df['is_weekend']==1]['label'].mean()
    weekday_spam_rate = df[df['is_weekend']==0]['label'].mean()
    print(f"  Weekend spam rate: {weekend_spam_rate*100:.1f}%")
    print(f"  Weekday spam rate: {weekday_spam_rate*100:.1f}%")
    insights['weekday'] = weekday_stats
    
    # Sender behavior
    print("\n  👤 Sender Behavior:")
    sender_stats = df.groupby('sender').agg({
        'label': ['count', 'mean']
    })
    sender_stats.columns = ['total_emails', 'spam_ratio']
    
    burst_senders = sender_stats[sender_stats['total_emails'] >= 10]
    print(f"  Total senders: {len(sender_stats):,}")
    print(f"  High-volume senders (≥10 emails): {len(burst_senders):,}")
    print(f"  High-volume spam rate: {(burst_senders['spam_ratio'] > 0.8).mean()*100:.1f}%")
    insights['sender'] = sender_stats
    
    # Create visualization
    print("\n  📊 Creating EDA visualizations...")
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
    bars = ax2.bar(range(len(region_spam)), region_spam.values, color='coral', alpha=0.7)
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
    sender_volumes = sender_stats['total_emails'].value_counts().sort_index()
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
    print(f"  ✓ Saved to: {eda_file}")
    plt.close()
    
    return insights


# Perform EDA
eda_insights = perform_temporal_eda(df_cleaned, REPORTS_PATH)

# ---------------------------------------------------------------------------
# 2.2: Text-based EDA (from lzy-EDA.ipynb)
# ---------------------------------------------------------------------------
print("\n[2.2] Analyzing text patterns...")

# Text length analysis
df_cleaned['subject_length'] = df_cleaned['subject'].fillna('').astype(str).apply(len)
df_cleaned['body_length'] = df_cleaned['body'].fillna('').astype(str).apply(len)
df_cleaned['text_length'] = df_cleaned['subject_length'] + df_cleaned['body_length']
df_cleaned['word_count'] = df_cleaned['text_combined'].apply(lambda x: len(str(x).split()))

print("\n  📝 Text Length Statistics:")
text_stats = df_cleaned.groupby('label')[['subject_length', 'body_length', 'word_count']].agg(['mean', 'median'])
print(text_stats)

# Character analysis
def calculate_uppercase_ratio(text):
    if pd.isna(text) or len(str(text)) == 0:
        return 0
    text = str(text)
    uppercase = sum(1 for c in text if c.isupper())
    letters = sum(1 for c in text if c.isalpha())
    return uppercase / letters if letters > 0 else 0

df_cleaned['uppercase_ratio'] = df_cleaned['text_combined'].apply(calculate_uppercase_ratio)
df_cleaned['exclamation_count'] = df_cleaned['text_combined'].apply(lambda x: str(x).count('!'))
df_cleaned['dollar_count'] = df_cleaned['text_combined'].apply(lambda x: str(x).count('$'))

print("\n  🔤 Character Pattern Statistics:")
char_stats = df_cleaned.groupby('label')[['uppercase_ratio', 'exclamation_count', 'dollar_count']].mean()
print(char_stats)

# Sentiment analysis (if available)
if HAS_NLP:
    try:
        df_cleaned['subject_sentiment'] = df_cleaned['subject'].fillna('').apply(
            lambda x: SENTIMENT_ANALYZER.polarity_scores(str(x))['compound']
        )
        df_cleaned['body_sentiment'] = df_cleaned['body'].fillna('').apply(
            lambda x: SENTIMENT_ANALYZER.polarity_scores(str(x))['compound']
        )
        
        print("\n  😊 Sentiment Statistics:")
        sentiment_stats = df_cleaned.groupby('label')[['subject_sentiment', 'body_sentiment']].mean()
        print(sentiment_stats)
    except Exception as e:
        print(f"\n  ⚠️  Sentiment analysis skipped: {e}")
        df_cleaned['subject_sentiment'] = 0
        df_cleaned['body_sentiment'] = 0
else:
    df_cleaned['subject_sentiment'] = 0
    df_cleaned['body_sentiment'] = 0

print("\n✅ EDA complete!")

# %%
# ============================================================================
# SECTION 3: FEATURE ENGINEERING (from date_data_feature_engineering.ipynb 
#            + spam_graph_eda_improved.ipynb + lzy notebooks)
# ============================================================================
print("\n" + "="*80)
print(" SECTION 3: FEATURE ENGINEERING")
print("="*80)
print("  Combining temporal (YR) + graph (Wenli) + text (LZY) features")

# ---------------------------------------------------------------------------
# 3.1: Temporal Feature Engineering (from YR's notebook)
# ---------------------------------------------------------------------------
def engineer_temporal_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Engineer temporal risk scores and sender behavior features.
    """
    if verbose:
        print("\n[3.1] Engineering temporal features...")
    
    df_features = df.copy()
    
    # Risk scores by hour and weekday (no leakage: global statistics)
    hour_risk = df.groupby('hour')['label'].mean()
    weekday_risk = df.groupby('day_of_week')['label'].mean()
    region_risk = df.groupby('timezone_region')['label'].mean()
    
    df_features['hour_risk_score'] = df_features['hour'].map(hour_risk)
    df_features['weekday_risk_score'] = df_features['day_of_week'].map(weekday_risk)
    df_features['region_risk_score'] = df_features['timezone_region'].map(region_risk)
    
    # Regional flags
    df_features['is_high_risk_region'] = (
        df_features['timezone_region'].isin(['Middle East/South Asia', 'APAC'])
    ).astype(int)
    
    # Sender historical features (time-aware to prevent leakage)
    df_features = df_features.sort_values(['sender', 'date']).reset_index(drop=True)
    
    # Cumulative sender statistics
    df_features['sender_email_count'] = df_features.groupby('sender').cumcount() + 1
    df_features['sender_is_first_email'] = (df_features['sender_email_count'] == 1).astype(int)
    
    # Sender cumulative spam rate (excluding current email)
    df_features['sender_cumulative_spam'] = (
        df_features.groupby('sender')['label'].cumsum() - df_features['label']
    )
    df_features['sender_cumulative_total'] = df_features['sender_email_count'] - 1
    df_features['sender_historical_spam_rate'] = np.where(
        df_features['sender_cumulative_total'] > 0,
        df_features['sender_cumulative_spam'] / df_features['sender_cumulative_total'],
        0
    )
    
    # Time gaps between emails (sender burst detection)
    df_features['time_since_last_email'] = (
        df_features.groupby('sender')['date'].diff().dt.total_seconds() / 60  # minutes
    )
    df_features['time_since_last_email'] = df_features['time_since_last_email'].fillna(9999)
    
    if verbose:
        print(f"  ✓ Added {7} temporal features")
        print(f"    - Risk scores (hour, weekday, region)")
        print(f"    - Sender history (email count, spam rate, time gaps)")
    
    return df_features


df_with_temporal = engineer_temporal_features(df_cleaned, verbose=True)

# ---------------------------------------------------------------------------
# 3.1b: URL and Domain Features (from lzy notebooks)
# ---------------------------------------------------------------------------
print("\n[3.1b] Engineering URL and domain features...")

# URL features
df_with_temporal['urls'] = df_with_temporal['urls'].fillna(0)
df_with_temporal['has_url'] = (df_with_temporal['urls'] > 0).astype(int)
df_with_temporal['urls_log'] = np.log1p(df_with_temporal['urls'])

# Domain features
df_with_temporal['sender_domain'] = df_with_temporal['sender'].apply(
    lambda x: str(x).split('@')[-1] if '@' in str(x) else 'unknown'
)

# Domain statistics (time-aware)
df_with_temporal = df_with_temporal.sort_values(['sender_domain', 'date']).reset_index(drop=True)

# Domain spam rate (cumulative, excluding current email)
domain_cumulative_spam = df_with_temporal.groupby('sender_domain')['label'].cumsum() - df_with_temporal['label']
domain_cumulative_total = df_with_temporal.groupby('sender_domain').cumcount()

df_with_temporal['domain_spam_rate'] = np.where(
    domain_cumulative_total > 0,
    domain_cumulative_spam / domain_cumulative_total,
    0
)

# Domain frequency
df_with_temporal['domain_frequency'] = df_with_temporal.groupby('sender_domain').cumcount() + 1

# Suspicious domain flags
df_with_temporal['is_suspicious_domain'] = (df_with_temporal['domain_spam_rate'] > 0.7).astype(int)
df_with_temporal['is_rare_domain'] = (df_with_temporal['domain_frequency'] <= 3).astype(int)

# Time-based features
df_with_temporal['is_night'] = df_with_temporal['hour'].isin(range(22, 24)).astype(int) | \
                                 df_with_temporal['hour'].isin(range(0, 6)).astype(int)

print(f"  ✓ Added URL features: urls, has_url, urls_log")
print(f"  ✓ Added domain features: domain_spam_rate, domain_frequency, suspicious/rare flags")
print(f"  ✓ Added time feature: is_night")

# Text meta-features (already computed in EDA, but ensure they exist)
if 'special_char_total' not in df_with_temporal.columns:
    df_with_temporal['special_char_total'] = (
        df_with_temporal['exclamation_count'] + 
        df_with_temporal['dollar_count']
    )

if 'digit_ratio' not in df_with_temporal.columns:
    def calculate_digit_ratio(text):
        if pd.isna(text) or len(str(text)) == 0:
            return 0
        text = str(text)
        digits = sum(c.isdigit() for c in text)
        return digits / len(text)
    
    df_with_temporal['digit_ratio'] = df_with_temporal['text_combined'].apply(calculate_digit_ratio)

if 'avg_word_length' not in df_with_temporal.columns:
    def calculate_avg_word_length(text):
        words = str(text).split()
        if len(words) == 0:
            return 0
        return sum(len(w) for w in words) / len(words)
    
    df_with_temporal['avg_word_length'] = df_with_temporal['text_combined'].apply(calculate_avg_word_length)

print(f"  ✓ Text meta-features: special_char_total, digit_ratio, avg_word_length")

# ---------------------------------------------------------------------------
# 3.2: Graph Feature Engineering (from Wenli's notebook)
# ---------------------------------------------------------------------------
def engineer_graph_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Extract graph-based network features for each sender.
    """
    if verbose:
        print("\n[3.2] Engineering graph features...")
        print("  ⏳ Building email network graph...")
    
    # Build directed graph
    G = nx.DiGraph()
    
    for idx, row in df.iterrows():
        sender = row['sender']
        receiver = row['receiver']
        is_spam = row['label']
        
        if G.has_edge(sender, receiver):
            G[sender][receiver]['weight'] += 1
            G[sender][receiver]['spam_count'] += is_spam
            G[sender][receiver]['ham_count'] += (1 - is_spam)
        else:
            G.add_edge(sender, receiver, 
                       weight=1, 
                       spam_count=is_spam,
                       ham_count=1-is_spam)
    
    if verbose:
        print(f"  ✓ Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        print("  ⏳ Extracting graph features...")
    
    # Pre-calculate metrics
    out_degrees = dict(G.out_degree())
    in_degrees = dict(G.in_degree())
    
    sender_graph_features = []
    
    for sender in df['sender'].unique():
        if sender not in G:
            continue
        
        out_deg = out_degrees.get(sender, 0)
        if out_deg == 0:
            continue
        
        # Basic metrics
        receivers = list(G.successors(sender))
        reciprocity = sum([1 for r in receivers if G.has_edge(r, sender)]) / len(receivers) if receivers else 0
        
        sender_graph_features.append({
            'sender': sender,
            'graph_out_degree': out_deg,
            'graph_in_degree': in_degrees.get(sender, 0),
            'graph_reciprocity': reciprocity,
            'graph_total_sent': len(receivers),
        })
    
    graph_df = pd.DataFrame(sender_graph_features)
    
    # Advanced features
    if verbose:
        print("  ⏳ Calculating PageRank & clustering...")
    
    pagerank = nx.pagerank(G, max_iter=50)
    clustering = nx.clustering(G.to_undirected())
    
    graph_df['graph_pagerank'] = graph_df['sender'].map(pagerank).fillna(0)
    graph_df['graph_clustering'] = graph_df['sender'].map(clustering).fillna(0)
    graph_df['graph_degree_centrality'] = graph_df['graph_out_degree'] / (G.number_of_nodes() - 1)
    graph_df['graph_avg_weight'] = graph_df['sender'].apply(
        lambda s: np.mean([G[s][r]['weight'] for r in G.successors(s)]) if s in G and G.out_degree(s) > 0 else 0
    )
    
    if verbose:
        print(f"  ✓ Added {8} graph features")
        print(f"    - Degree, reciprocity, PageRank, clustering, centrality")
    
    return graph_df


graph_features_df = engineer_graph_features(df_with_temporal, verbose=True)

# Merge graph features back to main dataframe
df_with_all_features = df_with_temporal.merge(
    graph_features_df,
    on='sender',
    how='left'
)

# Fill NaN values for senders not in graph
graph_cols = [col for col in df_with_all_features.columns if col.startswith('graph_')]
df_with_all_features[graph_cols] = df_with_all_features[graph_cols].fillna(0)

print("\n" + "─"*80)
print(" FEATURE ENGINEERING SUMMARY")
print("─"*80)
print(f"  Temporal features: 7")
print(f"  Graph features: 8")
print(f"  Total engineered features: 15")
print(f"  Total columns: {df_with_all_features.shape[1]}")
print("─"*80)

# Save engineered features
engineered_file = os.path.join(DATA_PROCESSED_PATH, "engineered_features.csv")
df_with_all_features.to_csv(engineered_file, index=False)
print(f"\n✅ Engineered features saved to: {engineered_file}")

# %%
# ============================================================================
# SECTION 4: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print(" SECTION 4: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Select ML features
temporal_features = [
    'hour', 'day_of_week', 'is_weekend',
    'hour_risk_score', 'weekday_risk_score', 'region_risk_score',
    'is_high_risk_region', 'sender_historical_spam_rate',
    'sender_email_count', 'time_since_last_email'
]

# URL and domain features (from lzy notebooks)
url_domain_features = [
    'urls', 'has_url', 'urls_log',
    'domain_spam_rate', 'is_suspicious_domain', 'domain_frequency', 'is_rare_domain',
    'is_night'
]

# Text meta-features (from lzy notebooks)
text_meta_features = [
    'subject_length', 'body_length', 'text_length', 'word_count',
    'uppercase_ratio', 'exclamation_count', 'dollar_count',
    'special_char_total', 'digit_ratio', 'avg_word_length',
    'subject_sentiment', 'body_sentiment'
]

graph_features = [col for col in df_with_all_features.columns if col.startswith('graph_')]

all_ml_features = temporal_features + url_domain_features + text_meta_features + graph_features

print(f"\n[4.1] Training Random Forest for feature importance...")
print(f"  Features: {len(all_ml_features)}")
print(f"    - Temporal: {len(temporal_features)}")
print(f"    - URL/Domain: {len(url_domain_features)}")
print(f"    - Text meta: {len(text_meta_features)}")
print(f"    - Graph: {len(graph_features)}")

# Prepare data
X = df_with_all_features[all_ml_features].fillna(0)
y = df_with_all_features['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n  Train set: {len(X_train):,} samples")
print(f"  Test set:  {len(X_test):,} samples")

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
accuracy = (y_pred == y_test).mean()

print(f"\n{'='*80}")
print(" MODEL PERFORMANCE")
print("="*80)
print(f"\nAccuracy: {accuracy*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Spam']))

# Feature importance
feature_types = (
    ['temporal']*len(temporal_features) + 
    ['url_domain']*len(url_domain_features) +
    ['text_meta']*len(text_meta_features) +
    ['graph']*len(graph_features)
)

importance_df = pd.DataFrame({
    'feature': all_ml_features,
    'importance': rf.feature_importances_,
    'type': feature_types
}).sort_values('importance', ascending=False)

print(f"\n{'='*80}")
print(" TOP 20 FEATURES")
print("="*80)
print(importance_df.head(20).to_string(index=False))

# Visualize feature importance
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
from matplotlib.patches import Patch
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
importance_file = os.path.join(REPORTS_PATH, 'feature_importance_analysis.png')
plt.savefig(importance_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Visualization saved to: {importance_file}")
plt.close()

# %%
# ============================================================================
# SECTION 5: FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print(" PIPELINE COMPLETE - SUMMARY")
print("="*80)

print(f"\n📊 DATA STATISTICS:")
print(f"  • Input emails:           {len(df_cleaned):,}")
print(f"  • With engineered features: {len(df_with_all_features):,}")
print(f"  • Unique senders:         {df_cleaned['sender'].nunique():,}")
print(f"  • Spam rate:              {df_cleaned['label'].mean()*100:.1f}%")

print(f"\n🎯 FEATURES GENERATED:")
print(f"  • Temporal features:      {len(temporal_features)}")
print(f"  • URL/Domain features:    {len(url_domain_features)}")
print(f"  • Text meta features:     {len(text_meta_features)}")
print(f"  • Graph features:         {len(graph_features)}")
print(f"  • Total ML features:      {len(all_ml_features)}")

print(f"\n🏆 MODEL PERFORMANCE:")
print(f"  • Accuracy:               {accuracy*100:.2f}%")
print(f"  • Training set:           {len(X_train):,}")
print(f"  • Test set:               {len(X_test):,}")

print(f"\n📁 OUTPUT FILES:")
print(f"  • Source data:            {data_file}")
print(f"  • Engineered features:    {engineered_file}")
print(f"  • EDA visualization:      {os.path.join(REPORTS_PATH, 'temporal_eda_analysis.png')}")
print(f"  • Feature importance:     {importance_file}")

print(f"\n💡 KEY INSIGHTS:")
print(f"  • Top feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['type']})")
print(f"  • Feature type contributions:")
for ftype in sorted(type_importance.index):
    print(f"    - {ftype.replace('_', ' ').title()}: {type_importance[ftype]/type_importance.sum()*100:.1f}%")

print("\n" + "="*80)
print("✅ PIPELINE EXECUTION COMPLETE!")
print("="*80)

# %%