# %% [markdown]
# # Haoran Feature Engineering Pipeline (Parallel Execution)
# This script runs three feature engineering modules in parallel:
# 1. Graph features (PIT-safe)
# 2. Time series features (PIT-safe)
# 3. Text/NLP features (PIT-safe)

# %% [markdown]
# ## Package Import
# %%
import os
import sys
import warnings
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
import time

warnings.filterwarnings('ignore')

# %% [markdown]
# ## Load and Basic Cleaning (Shared)

# %%
def load_and_clean_data(input_path="../data/processed/cleaned_date_merge.csv"):
    """
    Load and perform basic cleaning on the dataset.
    This is shared across all feature engineering modules.
    """
    print(f"\n{'='*80}")
    print("LOADING AND CLEANING DATA")
    print(f"{'='*80}")
    
    df = pd.read_csv(input_path)
    print(f"Dataset loaded: {len(df):,} emails")
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Clean sender/receiver
    df = df.dropna(subset=['sender', 'receiver', 'date', 'label'])
    df['sender'] = df['sender'].astype(str).str.strip().str.lower()
    df['receiver'] = df['receiver'].astype(str).str.strip().str.lower()
    df = df[(df['sender'] != 'nan') & (df['receiver'] != 'nan')]
    
    # Sort by date for point-in-time processing
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"After cleaning: {len(df):,} emails")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df

# %% [markdown]
# ## 1. Graph Feature Engineering Function

# %%
def compute_graph_features(df_input):
    """
    Compute graph features using point-in-time methods.
    Run as separate process.
    """
    print(f"\n{'='*80}")
    print("GRAPH FEATURE ENGINEERING (PIT)")
    print(f"{'='*80}")
    
    df = df_input.copy()
    
    # Initialize feature columns
    df['sender_out_degree'] = 0
    df['sender_in_degree'] = 0
    df['sender_total_degree'] = 0
    df['sender_reciprocity'] = 0.0
    df['sender_pagerank'] = 0.0
    df['sender_clustering'] = 0.0
    df['sender_closeness'] = 0.0
    df['sender_eigenvector'] = 0.0
    df['sender_triangles'] = 0
    df['sender_avg_weight'] = 0.0
    
    # Track historical stats per sender
    sender_history = {}
    
    # Process emails in chronological order (simplified for speed)
    G = nx.DiGraph()  # Build incrementally
    
    for idx in tqdm(range(len(df)), desc="Graph features"):
        current_sender = df.loc[idx, 'sender']
        current_receiver = df.loc[idx, 'receiver']
        current_label = df.loc[idx, 'label']
        
        # Compute features from current G (contains only past edges)
        if current_sender in G:
            out_deg = G.out_degree(current_sender)
            in_deg = G.in_degree(current_sender)
            
            df.loc[idx, 'sender_out_degree'] = out_deg
            df.loc[idx, 'sender_in_degree'] = in_deg
            df.loc[idx, 'sender_total_degree'] = out_deg + in_deg
            
            receivers = list(G.successors(current_sender))
            if receivers:
                reciprocity = sum([1 for r in receivers if G.has_edge(r, current_sender)]) / len(receivers)
                df.loc[idx, 'sender_reciprocity'] = reciprocity
                avg_weight = np.mean([G[current_sender][r]['weight'] for r in receivers])
                df.loc[idx, 'sender_avg_weight'] = avg_weight
        
        # Add current email to G for next iterations
        if G.has_edge(current_sender, current_receiver):
            G[current_sender][current_receiver]['weight'] += 1
        else:
            G.add_edge(current_sender, current_receiver, weight=1)
    
    # Add historical sender statistics
    df = df.sort_values(['sender', 'date']).reset_index(drop=True)
    df['sender_historical_email_count'] = df.groupby('sender').cumcount()
    df['sender_historical_spam_count'] = (
        df.groupby('sender')['label']
        .apply(lambda x: x.shift().cumsum().fillna(0))
        .reset_index(level=0, drop=True)
    )
    df['sender_historical_spam_rate'] = np.where(
        df['sender_historical_email_count'] > 0,
        df['sender_historical_spam_count'] / df['sender_historical_email_count'],
        0.0
    )
    df['sender_time_since_last_email'] = (
        df.groupby('sender')['date']
        .diff()
        .dt.total_seconds()
        .fillna(-1)
    )
    
    # Save
    output_path = "../data/processed/graph_features_pit.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Graph features saved to: {output_path}")
    
    return "graph_features_pit.csv"

# %% [markdown]
# ## 2. Time Series Feature Engineering Function

# %%
def compute_timeseries_features(df_input):
    """
    Compute time series features using point-in-time methods.
    Run as separate process.
    """
    print(f"\n{'='*80}")
    print("TIME SERIES FEATURE ENGINEERING (PIT)")
    print(f"{'='*80}")
    
    df = df_input.copy()
    
    # Basic temporal features
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    if 'timezone_region' in df.columns:
        df['is_middle_east'] = (df['timezone_region'] == 'Middle East/South Asia').astype(int)
    else:
        df['is_middle_east'] = 0
    
    # PIT-safe risk scores
    def compute_pit_risk_score(df, group_col, value_col='label'):
        df = df.sort_values('date').reset_index(drop=True)
        risk_scores = (
            df.groupby(group_col)[value_col]
            .apply(lambda x: x.shift().expanding().mean())
            .reset_index(level=0, drop=True)
        )
        global_mean = df[value_col].mean()
        risk_scores = risk_scores.fillna(global_mean)
        return risk_scores
    
    df['hour_risk_score'] = compute_pit_risk_score(df, 'hour', 'label')
    df['weekday_risk_score'] = compute_pit_risk_score(df, 'day_of_week', 'label')
    
    if 'timezone_region' in df.columns:
        df['region_risk_score'] = compute_pit_risk_score(df, 'timezone_region', 'label')
        df['region_hour'] = df['timezone_region'].astype(str) + '_' + df['hour'].astype(str)
        df['region_hour_risk'] = compute_pit_risk_score(df, 'region_hour', 'label')
    else:
        df['region_risk_score'] = df['label'].mean()
        df['region_hour_risk'] = df['label'].mean()
    
    # Sender historical features
    df = df.sort_values(['sender', 'date']).reset_index(drop=True)
    df['sender_historical_count'] = df.groupby('sender').cumcount()
    df['sender_historical_spam_count'] = (
        df.groupby('sender')['label']
        .apply(lambda x: x.shift().cumsum().fillna(0))
        .reset_index(level=0, drop=True)
    )
    df['sender_historical_phishing_rate'] = np.where(
        df['sender_historical_count'] > 0,
        df['sender_historical_spam_count'] / df['sender_historical_count'],
        df['label'].mean()
    )
    
    # Sender temporal features
    df['sender_time_gap'] = (
        df.groupby('sender')['date']
        .diff()
        .dt.total_seconds()
        .fillna(-1)
    )
    df['sender_time_gap_std'] = (
        df.groupby('sender')['sender_time_gap']
        .apply(lambda x: x.shift().expanding().std())
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    df['sender_first_date'] = df.groupby('sender')['date'].transform('first')
    df['sender_lifespan_days'] = (
        (df['date'] - df['sender_first_date']).dt.total_seconds() / 86400
    )
    df = df.drop(columns=['sender_first_date'])
    
    # Re-sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Save
    output_path = "../data/processed/timeseries_features_pit.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Time series features saved to: {output_path}")
    
    return "timeseries_features_pit.csv"

# %% [markdown]
# ## 3. Text/NLP Feature Engineering Function

# %%
def compute_text_features(df_input):
    """
    Compute text/NLP features.
    Run as separate process.
    """
    print(f"\n{'='*80}")
    print("TEXT/NLP FEATURE ENGINEERING")
    print(f"{'='*80}")
    
    df = df_input.copy()
    
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    
    # Download NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    # Fill missing text
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')
    df['full_text'] = df['subject'] + ' ' + df['body']
    
    # Text cleaning
    def clean_text_pipeline(text):
        if pd.isna(text) or text == "":
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        try:
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            tokens = [stemmer.stem(word) for word in tokens]
        except:
            pass
        return ' '.join(tokens)
    
    df['cleaned_text'] = df['full_text'].apply(clean_text_pipeline)
    
    # Text meta-features
    df['subject_length'] = df['subject'].str.len()
    df['body_length'] = df['body'].str.len()
    df['text_length'] = df['full_text'].str.len()
    df['word_count'] = df['full_text'].str.split().str.len()
    df['uppercase_count'] = df['full_text'].str.count(r'[A-Z]')
    df['uppercase_ratio'] = df['uppercase_count'] / (df['text_length'] + 1)
    df['exclamation_count'] = df['full_text'].str.count(r'!')
    df['question_count'] = df['full_text'].str.count(r'\?')
    df['dollar_count'] = df['full_text'].str.count(r'\$')
    df['percent_count'] = df['full_text'].str.count(r'%')
    df['star_count'] = df['full_text'].str.count(r'\*')
    df['special_char_total'] = (
        df['exclamation_count'] + df['question_count'] + 
        df['dollar_count'] + df['percent_count'] + df['star_count']
    )
    df['digit_count'] = df['full_text'].str.count(r'\d')
    df['digit_ratio'] = df['digit_count'] / (df['text_length'] + 1)
    df['avg_word_length'] = np.where(
        df['word_count'] > 0,
        df['text_length'] / df['word_count'],
        0
    )
    
    # Sentiment
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        df['subject_sentiment'] = df['subject'].apply(
            lambda x: analyzer.polarity_scores(str(x) if pd.notna(x) else "")['compound']
        )
        df['body_sentiment'] = df['body'].apply(
            lambda x: analyzer.polarity_scores(str(x) if pd.notna(x) else "")['compound']
        )
    except:
        df['subject_sentiment'] = 0.0
        df['body_sentiment'] = 0.0
    
    # URL features
    df['urls'] = df['urls'].fillna(0)
    df['has_url'] = (df['urls'] > 0).astype(int)
    df['urls_log'] = np.log1p(df['urls'])
    
    # Domain features (PIT-safe)
    df['sender_domain'] = df['sender'].str.split('@').str[-1]
    df = df.sort_values('date').reset_index(drop=True)
    df['domain_email_count'] = df.groupby('sender_domain').cumcount()
    df['domain_spam_cumsum'] = (
        df.groupby('sender_domain')['label']
        .apply(lambda x: x.shift().cumsum().fillna(0))
        .reset_index(level=0, drop=True)
    )
    global_spam_rate = df['label'].mean()
    df['domain_spam_rate'] = np.where(
        df['domain_email_count'] > 0,
        df['domain_spam_cumsum'] / df['domain_email_count'],
        global_spam_rate
    )
    df['is_suspicious_domain'] = (df['domain_spam_rate'] > 0.7).astype(int)
    df['domain_frequency'] = df['domain_email_count']
    df['is_rare_domain'] = (df['domain_frequency'] <= 5).astype(int)
    
    # Fill NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(0)
    
    # Save
    output_path = "../data/processed/text_features_pit.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Text/NLP features saved to: {output_path}")
    
    return "text_features_pit.csv"

# %% [markdown]
# ## Main Execution - Parallel Processing

# %%
if __name__ == '__main__':
    print("\n" + "="*80)
    print("PARALLEL FEATURE ENGINEERING PIPELINE")
    print("="*80)
    
    start_time = time.time()
    
    # Load data once
    df_original = load_and_clean_data()
    
    # Run three feature engineering tasks in parallel
    print("\n🚀 Starting parallel execution of 3 feature engineering modules...")
    print("  1. Graph features")
    print("  2. Time series features")
    print("  3. Text/NLP features")
    
    with Pool(processes=3) as pool:
        results = pool.map(
            lambda func: func(df_original),
            [compute_graph_features, compute_timeseries_features, compute_text_features]
        )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ ALL FEATURE ENGINEERING COMPLETE!")
    print("="*80)
    print(f"\n⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"\n📁 Output files created:")
    for result in results:
        print(f"  • {result}")
    print("\n🎉 Ready for downstream modeling!")

# %%
