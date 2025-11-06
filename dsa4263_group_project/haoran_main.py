# %% [marokdown]
# Haoran test file to udnerstand the structure of project
# %% [makrkdown]
# Packages Import
# %%
import os
import sys
import warnings
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
# %%
# TODO: To check how to add Yongrui's data cleaning method here instead of loading directly
df = pd.read_csv("../data/processed/cleaned_date_merge.csv")
print(f"Dataset loaded: {len(df):,} emails")
print(f"Columns: {df.columns.tolist()}")

# %% [markdown]
# ## Point-in-Time Graph Feature Engineering
# This section computes graph features using only past information for each email,
# preventing data leakage.

# %%
# Ensure date is datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Clean sender/receiver
df = df.dropna(subset=['sender', 'receiver', 'date'])
df['sender'] = df['sender'].astype(str).str.strip().str.lower()
df['receiver'] = df['receiver'].astype(str).str.strip().str.lower()
df = df[(df['sender'] != 'nan') & (df['receiver'] != 'nan')]

# Sort by date for point-in-time processing
df = df.sort_values('date').reset_index(drop=True)

print(f"\nAfter cleaning: {len(df):,} emails")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# %% [markdown]
# ### Step 1: Point-in-Time Graph Construction
# For each email, we build a graph using only emails sent BEFORE that timestamp.

# %%
# TODO: Need to check if this PIT method is working properly
def compute_point_in_time_graph_features(df):
    """
    Compute graph features for each email using only past data.
    
    For each row i, we:
    1. Build a graph from rows 0 to i-1 (emails before current email)
    2. Compute sender's graph features from that historical graph
    3. Assign those features to row i
    
    This ensures no data leakage.
    """
    
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
    sender_history = {}  # sender -> {email_count, spam_count}
    
    print("\n" + "="*80)
    print("COMPUTING POINT-IN-TIME GRAPH FEATURES")
    print("="*80)
    print("This may take a while for large datasets...\n")
    
    # Process emails in chronological order
    for idx in tqdm(range(len(df)), desc="Processing emails"):
        current_sender = df.loc[idx, 'sender']
        current_label = df.loc[idx, 'label']
        
        # Only compute if we have historical data
        if idx > 0:
            # Build graph from all emails BEFORE this one
            historical_df = df.iloc[:idx]
            
            # Build directed graph
            G = nx.DiGraph()
            
            for _, row in historical_df.iterrows():
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
            
            # Compute features for current sender if they exist in historical graph
            if current_sender in G:
                # Basic degree features
                out_deg = G.out_degree(current_sender)
                in_deg = G.in_degree(current_sender)
                
                df.loc[idx, 'sender_out_degree'] = out_deg
                df.loc[idx, 'sender_in_degree'] = in_deg
                df.loc[idx, 'sender_total_degree'] = out_deg + in_deg
                
                # Reciprocity
                receivers = list(G.successors(current_sender))
                if receivers:
                    reciprocity = sum([1 for r in receivers if G.has_edge(r, current_sender)]) / len(receivers)
                    df.loc[idx, 'sender_reciprocity'] = reciprocity
                    
                    # Average weight
                    avg_weight = np.mean([G[current_sender][r]['weight'] for r in receivers])
                    df.loc[idx, 'sender_avg_weight'] = avg_weight
                
                # Advanced features (only compute periodically to save time)
                # For demonstration, we'll compute every 100 emails or for first occurrence
                if idx % 100 == 0 or current_sender not in sender_history:
                    try:
                        # PageRank
                        pagerank = nx.pagerank(G, max_iter=50)
                        df.loc[idx, 'sender_pagerank'] = pagerank.get(current_sender, 0)
                        
                        # Clustering coefficient
                        G_undirected = G.to_undirected()
                        clustering = nx.clustering(G_undirected)
                        df.loc[idx, 'sender_clustering'] = clustering.get(current_sender, 0)
                        
                        # Closeness centrality
                        closeness = nx.closeness_centrality(G)
                        df.loc[idx, 'sender_closeness'] = closeness.get(current_sender, 0)
                        
                        # Eigenvector centrality
                        try:
                            eigenvector = nx.eigenvector_centrality(G, max_iter=100)
                            df.loc[idx, 'sender_eigenvector'] = eigenvector.get(current_sender, 0)
                        except:
                            df.loc[idx, 'sender_eigenvector'] = 0
                        
                        # Triangles
                        G_temp = G.copy()
                        G_temp.remove_edges_from(nx.selfloop_edges(G_temp))
                        triangles = nx.triangles(G_temp.to_undirected())
                        df.loc[idx, 'sender_triangles'] = triangles.get(current_sender, 0)
                        
                    except Exception as e:
                        # If computation fails, keep zeros
                        pass
        
        # Update sender history (for future reference)
        if current_sender not in sender_history:
            sender_history[current_sender] = {'email_count': 0, 'spam_count': 0}
        
        sender_history[current_sender]['email_count'] += 1
        sender_history[current_sender]['spam_count'] += current_label
    
    print("\n✓ Point-in-time graph features computed!")
    return df

# Compute features
df = compute_point_in_time_graph_features(df)

# %% [markdown]
# ### Step 2: Add Historical Sender Statistics

# %%
# Add point-in-time sender statistics
df = df.sort_values(['sender', 'date']).reset_index(drop=True)

# Historical email count for this sender (up to but not including current email)
df['sender_historical_email_count'] = df.groupby('sender').cumcount()

# Historical spam count for this sender (shifted to exclude current email)
df['sender_historical_spam_count'] = (
    df.groupby('sender')['label']
    .apply(lambda x: x.shift().cumsum().fillna(0))
    .reset_index(level=0, drop=True)
)

# Historical spam rate
df['sender_historical_spam_rate'] = np.where(
    df['sender_historical_email_count'] > 0,
    df['sender_historical_spam_count'] / df['sender_historical_email_count'],
    0.0
)

# Time since last email from this sender
df['sender_time_since_last_email'] = (
    df.groupby('sender')['date']
    .diff()
    .dt.total_seconds()
    .fillna(-1)
)

# %% [markdown]
# ### Step 3: Summary Statistics

# %%
print("\n" + "="*80)
print("POINT-IN-TIME GRAPH FEATURES SUMMARY")
print("="*80)

graph_features = [
    'sender_out_degree', 'sender_in_degree', 'sender_total_degree',
    'sender_reciprocity', 'sender_pagerank', 'sender_clustering',
    'sender_closeness', 'sender_eigenvector', 'sender_triangles',
    'sender_avg_weight', 'sender_historical_email_count',
    'sender_historical_spam_count', 'sender_historical_spam_rate',
    'sender_time_since_last_email'
]

print(f"\nTotal graph features added: {len(graph_features)}")
print(f"\nFeature list:")
for feature in graph_features:
    print(f"  • {feature}")

print(f"\n{'='*80}")
print("FEATURE STATISTICS (non-zero values)")
print(f"{'='*80}\n")

for feature in graph_features[:10]:  # Show first 10
    non_zero = (df[feature] != 0).sum()
    if non_zero > 0:
        print(f"{feature:40s}: {non_zero:6,} / {len(df):,} ({non_zero/len(df)*100:5.1f}%)")
        print(f"  Mean: {df[feature].mean():.4f}, Median: {df[feature].median():.4f}, "
              f"Max: {df[feature].max():.4f}")

# %%
print("\n✅ Point-in-time graph feature engineering complete!")
print(f"Final dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Save graph features to CSV
output_graph_path = "../data/processed/graph_features_pit.csv"
df.to_csv(output_graph_path, index=False)
print(f"\n💾 Graph features saved to: {output_graph_path}")

# %% [markdown]
# ## Point-in-Time Time Series Feature Engineering
# This section computes temporal risk scores and sender temporal features using only past data.
# Works on a fresh copy of the original cleaned data.

# %%
print("\n" + "="*80)
print("POINT-IN-TIME TIME SERIES FEATURE ENGINEERING")
print("="*80)

# Reload original data to work on a clean copy
df_ts = pd.read_csv("../data/processed/cleaned_date_merge.csv")
df_ts['date'] = pd.to_datetime(df_ts['date'], errors='coerce')

# Clean and prepare
df_ts = df_ts.dropna(subset=['sender', 'receiver', 'date', 'label'])
df_ts['sender'] = df_ts['sender'].astype(str).str.strip().str.lower()
df_ts['receiver'] = df_ts['receiver'].astype(str).str.strip().str.lower()
df_ts = df_ts[(df_ts['sender'] != 'nan') & (df_ts['receiver'] != 'nan')]

# Sort by date for PIT processing
df_ts = df_ts.sort_values('date').reset_index(drop=True)

print(f"\nTime series df loaded: {len(df_ts):,} emails")
print(f"Date range: {df_ts['date'].min()} to {df_ts['date'].max()}")

# %% [markdown]
# ### Step 1: Basic Temporal Features (Static, PIT-safe)

# %%
print("\n[1/6] Extracting basic temporal features...")

# Extract basic temporal components (already PIT-safe - these are static properties)
df_ts['hour'] = df_ts['date'].dt.hour
df_ts['day_of_week'] = df_ts['date'].dt.dayofweek  # 0=Monday, 6=Sunday
df_ts['is_weekend'] = df_ts['day_of_week'].isin([5, 6]).astype(int)
df_ts['is_night'] = ((df_ts['hour'] >= 22) | (df_ts['hour'] <= 6)).astype(int)

# Timezone region (if available)
if 'timezone_region' in df_ts.columns:
    df_ts['is_middle_east'] = (df_ts['timezone_region'] == 'Middle East/South Asia').astype(int)
else:
    df_ts['is_middle_east'] = 0

print("✓ Basic temporal features extracted")

# %% [markdown]
# ### Step 2: PIT-Safe Temporal Risk Scores
# For each email, compute risk scores using only emails sent BEFORE that timestamp.

# %%
print("\n[2/6] Computing PIT-safe temporal risk scores...")

def compute_pit_risk_score(df, group_col, value_col='label'):
    """
    Compute point-in-time risk score (mean of value_col) for each group.
    For row i, uses only rows 0 to i-1 in that group.
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    # Use expanding window with shift to exclude current row
    risk_scores = (
        df.groupby(group_col)[value_col]
        .apply(lambda x: x.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )
    
    # Fill NaN with global mean (for first occurrence of each group)
    global_mean = df[value_col].mean()
    risk_scores = risk_scores.fillna(global_mean)
    
    return risk_scores

# Hour risk score (PIT-safe)
df_ts['hour_risk_score'] = compute_pit_risk_score(df_ts, 'hour', 'label')

# Weekday risk score (PIT-safe)
df_ts['weekday_risk_score'] = compute_pit_risk_score(df_ts, 'day_of_week', 'label')

# Region risk score (PIT-safe) - if timezone_region exists
if 'timezone_region' in df_ts.columns:
    df_ts['region_risk_score'] = compute_pit_risk_score(df_ts, 'timezone_region', 'label')
else:
    df_ts['region_risk_score'] = df_ts['label'].mean()

print("✓ PIT-safe temporal risk scores computed")

# %% [markdown]
# ### Step 3: PIT-Safe Interaction Risk Scores

# %%
print("\n[3/6] Computing PIT-safe interaction risk scores...")

# Create interaction column
if 'timezone_region' in df_ts.columns:
    df_ts['region_hour'] = df_ts['timezone_region'].astype(str) + '_' + df_ts['hour'].astype(str)
    df_ts['region_hour_risk'] = compute_pit_risk_score(df_ts, 'region_hour', 'label')
else:
    df_ts['region_hour_risk'] = df_ts['label'].mean()

print("✓ PIT-safe interaction risk scores computed")

# %% [markdown]
# ### Step 4: Sender Historical Features (PIT-safe)

# %%
print("\n[4/6] Computing sender historical features...")

# Sort by sender and date
df_ts = df_ts.sort_values(['sender', 'date']).reset_index(drop=True)

# Sender historical email count (up to but not including current email)
df_ts['sender_historical_count'] = df_ts.groupby('sender').cumcount()

# Sender historical spam count (shifted to exclude current email)
df_ts['sender_historical_spam_count'] = (
    df_ts.groupby('sender')['label']
    .apply(lambda x: x.shift().cumsum().fillna(0))
    .reset_index(level=0, drop=True)
)

# Sender historical phishing rate
df_ts['sender_historical_phishing_rate'] = np.where(
    df_ts['sender_historical_count'] > 0,
    df_ts['sender_historical_spam_count'] / df_ts['sender_historical_count'],
    df_ts['label'].mean()  # Use global mean for first email
)

print("✓ Sender historical features computed")

# %% [markdown]
# ### Step 5: Sender Temporal Features (PIT-safe)

# %%
print("\n[5/6] Computing sender temporal features...")

# Time gap since last email from sender (in seconds)
df_ts['sender_time_gap'] = (
    df_ts.groupby('sender')['date']
    .diff()
    .dt.total_seconds()
    .fillna(-1)  # -1 for first email from sender
)

# Standard deviation of time gaps (expanding window, excluding current)
df_ts['sender_time_gap_std'] = (
    df_ts.groupby('sender')['sender_time_gap']
    .apply(lambda x: x.shift().expanding().std())
    .reset_index(level=0, drop=True)
    .fillna(0)
)

# Sender lifespan in days (from first email to current)
df_ts['sender_first_date'] = df_ts.groupby('sender')['date'].transform('first')
df_ts['sender_lifespan_days'] = (
    (df_ts['date'] - df_ts['sender_first_date']).dt.total_seconds() / 86400
)
df_ts = df_ts.drop(columns=['sender_first_date'])

print("✓ Sender temporal features computed")

# %% [markdown]
# ### Step 6: Summary and Save

# %%
print("\n[6/6] Summary and saving...")

# Re-sort by date for consistency
df_ts = df_ts.sort_values('date').reset_index(drop=True)

# List all time series features
ts_features = [
    'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_middle_east',
    'hour_risk_score', 'weekday_risk_score', 'region_risk_score', 'region_hour_risk',
    'sender_historical_count', 'sender_historical_spam_count', 'sender_historical_phishing_rate',
    'sender_time_gap', 'sender_time_gap_std', 'sender_lifespan_days'
]

print("\n" + "="*80)
print("TIME SERIES FEATURES SUMMARY")
print("="*80)
print(f"\nTotal time series features: {len(ts_features)}")
print(f"\nFeature list:")
for feature in ts_features:
    if feature in df_ts.columns:
        print(f"  • {feature}")

print(f"\n{'='*80}")
print("FEATURE STATISTICS")
print(f"{'='*80}\n")

for feature in ts_features[:10]:  # Show first 10
    if feature in df_ts.columns:
        non_null = df_ts[feature].notna().sum()
        print(f"{feature:40s}: {non_null:6,} / {len(df_ts):,} ({non_null/len(df_ts)*100:5.1f}%)")
        if df_ts[feature].dtype in ['float64', 'int64']:
            print(f"  Mean: {df_ts[feature].mean():.4f}, Median: {df_ts[feature].median():.4f}, "
                  f"Max: {df_ts[feature].max():.4f}")

# Save time series features to CSV
output_ts_path = "../data/processed/timeseries_features_pit.csv"
df_ts.to_csv(output_ts_path, index=False)
print(f"\n💾 Time series features saved to: {output_ts_path}")

print("\n✅ Point-in-time time series feature engineering complete!")
print(f"Final dataset shape: {df_ts.shape}")

# %% [markdown]
# ## Text/NLP Feature Engineering
# This section computes text-based features including cleaning, sentiment, and domain features.
# Works on a fresh copy of the original cleaned data.
# Most features are PIT-safe by design (computed per email).
# Domain-based features will use PIT methods.

# %%
print("\n" + "="*80)
print("TEXT/NLP FEATURE ENGINEERING")
print("="*80)

# Reload original data to work on a clean copy
df_text = pd.read_csv("../data/processed/cleaned_date_merge.csv")
df_text['date'] = pd.to_datetime(df_text['date'], errors='coerce')

# Clean and prepare
df_text = df_text.dropna(subset=['sender', 'receiver', 'date', 'label'])
df_text['sender'] = df_text['sender'].astype(str).str.strip().str.lower()
df_text['receiver'] = df_text['receiver'].astype(str).str.strip().str.lower()
df_text = df_text[(df_text['sender'] != 'nan') & (df_text['receiver'] != 'nan')]

# Sort by date for PIT processing
df_text = df_text.sort_values('date').reset_index(drop=True)

print(f"\nText df loaded: {len(df_text):,} emails")
print(f"Date range: {df_text['date'].min()} to {df_text['date'].max()}")

# %% [markdown]
# ### Step 1: Text Cleaning Pipeline

# %%
print("\n[1/7] Text cleaning pipeline...")

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download required NLTK data (run once)
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

# Initialize
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Fill missing text
df_text['subject'] = df_text['subject'].fillna('')
df_text['body'] = df_text['body'].fillna('')

# Create combined text
df_text['full_text'] = df_text['subject'] + ' ' + df_text['body']

def clean_text_pipeline(text):
    """
    Multi-step text cleaning pipeline.
    PIT-safe: operates on each email independently.
    """
    if pd.isna(text) or text == "":
        return ""
    
    # Step 1: Lowercase
    text = str(text).lower()
    
    # Step 2: Remove punctuation and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Step 3: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 4: Tokenization
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Step 5: Remove stopwords
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Step 6: Lemmatization
    try:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    except:
        pass
    
    # Step 7: Stemming (optional - can be final step)
    try:
        tokens = [stemmer.stem(word) for word in tokens]
    except:
        pass
    
    return ' '.join(tokens)

# Apply cleaning
print("  Cleaning text (this may take a few minutes)...")
df_text['cleaned_text'] = df_text['full_text'].apply(clean_text_pipeline)

print("✓ Text cleaning complete")

# %% [markdown]
# ### Step 2: Text Meta-Features (PIT-safe by design)

# %%
print("\n[2/7] Computing text meta-features...")

# Length features (computed from raw text, PIT-safe)
df_text['subject_length'] = df_text['subject'].str.len()
df_text['body_length'] = df_text['body'].str.len()
df_text['text_length'] = df_text['full_text'].str.len()
df_text['word_count'] = df_text['full_text'].str.split().str.len()

# Character-based features (PIT-safe)
df_text['uppercase_count'] = df_text['full_text'].str.count(r'[A-Z]')
df_text['uppercase_ratio'] = df_text['uppercase_count'] / (df_text['text_length'] + 1)

df_text['exclamation_count'] = df_text['full_text'].str.count(r'!')
df_text['question_count'] = df_text['full_text'].str.count(r'\?')
df_text['dollar_count'] = df_text['full_text'].str.count(r'\$')
df_text['percent_count'] = df_text['full_text'].str.count(r'%')
df_text['star_count'] = df_text['full_text'].str.count(r'\*')

# Total special characters
df_text['special_char_total'] = (
    df_text['exclamation_count'] + df_text['question_count'] + 
    df_text['dollar_count'] + df_text['percent_count'] + df_text['star_count']
)

# Digit features (PIT-safe)
df_text['digit_count'] = df_text['full_text'].str.count(r'\d')
df_text['digit_ratio'] = df_text['digit_count'] / (df_text['text_length'] + 1)

# Average word length (PIT-safe)
df_text['avg_word_length'] = np.where(
    df_text['word_count'] > 0,
    df_text['text_length'] / df_text['word_count'],
    0
)

print("✓ Text meta-features computed")

# %% [markdown]
# ### Step 3: Sentiment Analysis (PIT-safe by design)

# %%
print("\n[3/7] Computing sentiment features...")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        """Get VADER compound sentiment score. PIT-safe: per-email only."""
        if pd.isna(text) or text == "":
            return 0.0
        try:
            return analyzer.polarity_scores(str(text))['compound']
        except:
            return 0.0
    
    df_text['subject_sentiment'] = df_text['subject'].apply(get_sentiment)
    df_text['body_sentiment'] = df_text['body'].apply(get_sentiment)
    
    print("✓ Sentiment features computed")
except ImportError:
    print("⚠️  vaderSentiment not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "vaderSentiment", "-q"])
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    analyzer = SentimentIntensityAnalyzer()
    df_text['subject_sentiment'] = df_text['subject'].apply(
        lambda x: analyzer.polarity_scores(str(x) if pd.notna(x) else "")['compound']
    )
    df_text['body_sentiment'] = df_text['body'].apply(
        lambda x: analyzer.polarity_scores(str(x) if pd.notna(x) else "")['compound']
    )
    print("✓ Sentiment features computed")

# %% [markdown]
# ### Step 4: URL Features (PIT-safe by design)

# %%
print("\n[4/7] Computing URL features...")

# Handle missing URLs
df_text['urls'] = df_text['urls'].fillna(0)

# URL features (PIT-safe - per email only)
df_text['has_url'] = (df_text['urls'] > 0).astype(int)
df_text['urls_log'] = np.log1p(df_text['urls'])

print("✓ URL features computed")

# %% [markdown]
# ### Step 5: Domain Features (PIT-safe with historical data)

# %%
print("\n[5/7] Computing domain features (PIT-safe)...")

# Extract sender domain
df_text['sender_domain'] = df_text['sender'].str.split('@').str[-1]

# PIT-safe domain spam rate (using only past emails)
df_text = df_text.sort_values('date').reset_index(drop=True)

# Calculate cumulative domain statistics
df_text['domain_email_count'] = df_text.groupby('sender_domain').cumcount()
df_text['domain_spam_cumsum'] = (
    df_text.groupby('sender_domain')['label']
    .apply(lambda x: x.shift().cumsum().fillna(0))
    .reset_index(level=0, drop=True)
)

# Domain spam rate (PIT-safe)
global_spam_rate = df_text['label'].mean()
df_text['domain_spam_rate'] = np.where(
    df_text['domain_email_count'] > 0,
    df_text['domain_spam_cumsum'] / df_text['domain_email_count'],
    global_spam_rate  # Use global mean for first occurrence
)

# Is suspicious domain (based on PIT domain spam rate)
df_text['is_suspicious_domain'] = (df_text['domain_spam_rate'] > 0.7).astype(int)

# Domain frequency (PIT-safe - count of past emails from this domain)
df_text['domain_frequency'] = df_text['domain_email_count']

# Is rare domain (PIT-safe - based on past frequency)
df_text['is_rare_domain'] = (df_text['domain_frequency'] <= 5).astype(int)

print("✓ Domain features computed (PIT-safe)")

# %% [markdown]
# ### Step 6: Additional Cleaning (if needed)

# %%
print("\n[6/7] Additional data cleaning...")

# Fill any remaining NaNs in numeric columns
numeric_cols = df_text.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_text[col].isna().any():
        df_text[col] = df_text[col].fillna(0)

print("✓ Additional cleaning complete")

# %% [markdown]
# ### Step 7: Summary and Save

# %%
print("\n[7/7] Summary and saving...")

# List all text/NLP features
text_features = [
    'cleaned_text', 'full_text',
    'subject_length', 'body_length', 'text_length', 'word_count',
    'uppercase_count', 'uppercase_ratio', 'exclamation_count', 'question_count',
    'dollar_count', 'percent_count', 'star_count', 'special_char_total',
    'digit_count', 'digit_ratio', 'avg_word_length',
    'subject_sentiment', 'body_sentiment',
    'urls', 'has_url', 'urls_log',
    'sender_domain', 'domain_spam_rate', 'is_suspicious_domain',
    'domain_frequency', 'is_rare_domain'
]

print("\n" + "="*80)
print("TEXT/NLP FEATURES SUMMARY")
print("="*80)
print(f"\nTotal text/NLP features: {len(text_features)}")
print(f"\nFeature list:")
for feature in text_features:
    if feature in df_text.columns:
        print(f"  • {feature}")

print(f"\n{'='*80}")
print("FEATURE STATISTICS")
print(f"{'='*80}\n")

numeric_text_features = [f for f in text_features if f in df_text.columns and df_text[f].dtype in ['float64', 'int64']]
for feature in numeric_text_features[:15]:  # Show first 15
    non_null = df_text[feature].notna().sum()
    print(f"{feature:40s}: {non_null:6,} / {len(df_text):,} ({non_null/len(df_text)*100:5.1f}%)")
    print(f"  Mean: {df_text[feature].mean():.4f}, Median: {df_text[feature].median():.4f}, "
          f"Max: {df_text[feature].max():.4f}")

# Save text/NLP features to CSV
output_text_path = "../data/processed/text_features_pit.csv"
df_text.to_csv(output_text_path, index=False)
print(f"\n💾 Text/NLP features saved to: {output_text_path}")

print("\n✅ Text/NLP feature engineering complete!")
print(f"Final dataset shape: {df_text.shape}")

# %%
