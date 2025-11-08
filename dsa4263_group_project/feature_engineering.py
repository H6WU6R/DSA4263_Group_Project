"""
Feature Engineering Module

This module contains functions for engineering features from email data,
including temporal features, URL/domain features, text meta-features, and graph features.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple, List


def engineer_temporal_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Engineer temporal risk scores and sender behavior features.
    
    Features created:
    - hour_risk_score, weekday_risk_score, region_risk_score
    - is_high_risk_region
    - sender_email_count, sender_historical_spam_rate
    - time_since_last_email
    
    Args:
        df: Cleaned email dataframe with temporal columns
        verbose: Print progress messages
        
    Returns:
        Dataframe with additional temporal features
    """
    if verbose:
        print("\n[Features] Engineering temporal features...")
    
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
        print(f"  ✓ Added 10 temporal features")
    
    return df_features


def engineer_url_domain_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Engineer URL and domain-based features.
    
    Features created:
    - urls, has_url, urls_log
    - sender_domain, domain_spam_rate, domain_frequency
    - is_suspicious_domain, is_rare_domain
    - is_night
    
    Args:
        df: Email dataframe
        verbose: Print progress messages
        
    Returns:
        Dataframe with URL/domain features
    """
    if verbose:
        print("\n[Features] Engineering URL and domain features...")
    
    df_features = df.copy()
    
    # URL features
    df_features['urls'] = df_features['urls'].fillna(0)
    df_features['has_url'] = (df_features['urls'] > 0).astype(int)
    df_features['urls_log'] = np.log1p(df_features['urls'])
    
    # Domain features
    df_features['sender_domain'] = df_features['sender'].apply(
        lambda x: str(x).split('@')[-1] if '@' in str(x) else 'unknown'
    )
    
    # Domain statistics (time-aware)
    df_features = df_features.sort_values(['sender_domain', 'date']).reset_index(drop=True)
    
    # Domain spam rate (cumulative, excluding current email)
    domain_cumulative_spam = df_features.groupby('sender_domain')['label'].cumsum() - df_features['label']
    domain_cumulative_total = df_features.groupby('sender_domain').cumcount()
    
    df_features['domain_spam_rate'] = np.where(
        domain_cumulative_total > 0,
        domain_cumulative_spam / domain_cumulative_total,
        0
    )
    
    # Domain frequency
    df_features['domain_frequency'] = df_features.groupby('sender_domain').cumcount() + 1
    
    # Suspicious domain flags
    df_features['is_suspicious_domain'] = (df_features['domain_spam_rate'] > 0.7).astype(int)
    df_features['is_rare_domain'] = (df_features['domain_frequency'] <= 3).astype(int)
    
    # Time-based features
    df_features['is_night'] = (
        df_features['hour'].isin(range(22, 24)).astype(int) | 
        df_features['hour'].isin(range(0, 6)).astype(int)
    )
    
    if verbose:
        print(f"  ✓ Added 8 URL/domain features")
    
    return df_features


def engineer_text_meta_features(df: pd.DataFrame, sentiment_analyzer=None, verbose: bool = True) -> pd.DataFrame:
    """
    Engineer text meta-features from email content.
    
    Features created:
    - subject_length, body_length, text_length, word_count
    - uppercase_ratio, exclamation_count, dollar_count
    - special_char_total, digit_ratio, avg_word_length
    - subject_sentiment, body_sentiment (if analyzer provided)
    
    Args:
        df: Email dataframe with text columns
        sentiment_analyzer: NLTK SentimentIntensityAnalyzer (optional)
        verbose: Print progress messages
        
    Returns:
        Dataframe with text meta-features
    """
    if verbose:
        print("\n[Features] Engineering text meta-features...")
    
    df_features = df.copy()
    
    # Length features
    df_features['subject_length'] = df_features['subject'].fillna('').astype(str).apply(len)
    df_features['body_length'] = df_features['body'].fillna('').astype(str).apply(len)
    df_features['text_length'] = df_features['subject_length'] + df_features['body_length']
    df_features['word_count'] = df_features['text_combined'].apply(lambda x: len(str(x).split()))
    
    # Character-based features
    def calculate_uppercase_ratio(text):
        if pd.isna(text) or len(str(text)) == 0:
            return 0
        text = str(text)
        uppercase = sum(1 for c in text if c.isupper())
        letters = sum(1 for c in text if c.isalpha())
        return uppercase / letters if letters > 0 else 0
    
    df_features['uppercase_ratio'] = df_features['text_combined'].apply(calculate_uppercase_ratio)
    df_features['exclamation_count'] = df_features['text_combined'].apply(lambda x: str(x).count('!'))
    df_features['dollar_count'] = df_features['text_combined'].apply(lambda x: str(x).count('$'))
    df_features['special_char_total'] = (
        df_features['exclamation_count'] + df_features['dollar_count']
    )
    
    # Digit features
    def calculate_digit_ratio(text):
        if pd.isna(text) or len(str(text)) == 0:
            return 0
        text = str(text)
        digits = sum(c.isdigit() for c in text)
        return digits / len(text)
    
    df_features['digit_ratio'] = df_features['text_combined'].apply(calculate_digit_ratio)
    
    # Average word length
    def calculate_avg_word_length(text):
        words = str(text).split()
        if len(words) == 0:
            return 0
        return sum(len(w) for w in words) / len(words)
    
    df_features['avg_word_length'] = df_features['text_combined'].apply(calculate_avg_word_length)
    
    # Sentiment analysis (if available)
    if sentiment_analyzer is not None:
        try:
            df_features['subject_sentiment'] = df_features['subject'].fillna('').apply(
                lambda x: sentiment_analyzer.polarity_scores(str(x))['compound']
            )
            df_features['body_sentiment'] = df_features['body'].fillna('').apply(
                lambda x: sentiment_analyzer.polarity_scores(str(x))['compound']
            )
            if verbose:
                print(f"  ✓ Added 12 text meta-features (including sentiment)")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Sentiment analysis failed: {e}")
            df_features['subject_sentiment'] = 0
            df_features['body_sentiment'] = 0
            if verbose:
                print(f"  ✓ Added 12 text meta-features (sentiment skipped)")
    else:
        df_features['subject_sentiment'] = 0
        df_features['body_sentiment'] = 0
        if verbose:
            print(f"  ✓ Added 12 text meta-features (no sentiment)")
    
    return df_features


def build_graph_from_data(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build directed graph from email data.
    
    Args:
        df: Email dataframe with sender, receiver, label columns
        
    Returns:
        NetworkX directed graph with edge weights and spam counts
    """
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
    
    return G


def extract_graph_features_for_senders(senders: pd.Series, G: nx.DiGraph, verbose: bool = False) -> pd.DataFrame:
    """
    Extract graph-based network features for given senders using a pre-built graph.
    
    This prevents data leakage by allowing us to build the graph only on training data.
    
    Features extracted:
    - graph_out_degree, graph_in_degree
    - graph_reciprocity, graph_total_sent
    - graph_pagerank, graph_clustering
    - graph_degree_centrality, graph_avg_weight
    
    Args:
        senders: Series of sender emails to extract features for
        G: Pre-built NetworkX graph
        verbose: Print progress messages
        
    Returns:
        Dataframe with graph features for each unique sender
    """
    if verbose:
        print(f"  ⏳ Extracting graph features for {len(senders.unique())} unique senders...")
    
    # Pre-calculate metrics
    out_degrees = dict(G.out_degree())
    in_degrees = dict(G.in_degree())
    pagerank = nx.pagerank(G, max_iter=50)
    clustering = nx.clustering(G.to_undirected())
    n_nodes = G.number_of_nodes()
    
    sender_graph_features = []
    
    for sender in senders.unique():
        if sender not in G:
            # Sender not in training graph - use default values
            sender_graph_features.append({
                'sender': sender,
                'graph_out_degree': 0,
                'graph_in_degree': 0,
                'graph_reciprocity': 0,
                'graph_total_sent': 0,
                'graph_pagerank': 0,
                'graph_clustering': 0,
                'graph_degree_centrality': 0,
                'graph_avg_weight': 0,
            })
            continue
        
        out_deg = out_degrees.get(sender, 0)
        
        # Basic metrics
        receivers = list(G.successors(sender))
        reciprocity = sum([1 for r in receivers if G.has_edge(r, sender)]) / len(receivers) if receivers else 0
        avg_weight = np.mean([G[sender][r]['weight'] for r in receivers]) if receivers else 0
        
        sender_graph_features.append({
            'sender': sender,
            'graph_out_degree': out_deg,
            'graph_in_degree': in_degrees.get(sender, 0),
            'graph_reciprocity': reciprocity,
            'graph_total_sent': len(receivers),
            'graph_pagerank': pagerank.get(sender, 0),
            'graph_clustering': clustering.get(sender, 0),
            'graph_degree_centrality': out_deg / (n_nodes - 1) if n_nodes > 1 else 0,
            'graph_avg_weight': avg_weight,
        })
    
    return pd.DataFrame(sender_graph_features)


def prepare_features_with_graph(
    df: pd.DataFrame,
    train_idx: pd.Index,
    test_idx: pd.Index,
    temporal_features: List[str],
    url_domain_features: List[str],
    text_meta_features: List[str],
    graph_feature_names: List[str],
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare features with graph features built ONLY on training data.
    This prevents data leakage.
    
    Args:
        df: Full dataframe with non-graph features
        train_idx, test_idx: Train/test indices
        temporal_features: List of temporal feature names
        url_domain_features: List of URL/domain feature names
        text_meta_features: List of text meta-feature names
        graph_feature_names: List of graph feature names
        verbose: Print progress messages
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if verbose:
        print(f"\n[4.2] Building graph features (leakage-free)...")
    
    # Build graph on training data only
    train_data = df.loc[train_idx]
    
    train_graph = build_graph_from_data(
        train_data['sender'].values,
        train_data['receiver'].values,
        train_data['label'].values
    )
    
    if verbose:
        print(f"  ✓ Graph built with {train_graph.number_of_nodes()} nodes, {train_graph.number_of_edges()} edges")
    
    # Extract graph features for train set
    train_graph_features = extract_graph_features_for_senders(
        train_data['sender'].values,
        train_graph
    )
    
    # Extract graph features for test set (using training graph)
    test_data = df.loc[test_idx]
    test_graph_features = extract_graph_features_for_senders(
        test_data['sender'].values,
        train_graph
    )
    
    # Combine all features
    all_features = temporal_features + url_domain_features + text_meta_features + graph_feature_names
    
    # Prepare X_train
    X_train = df.loc[train_idx, temporal_features + url_domain_features + text_meta_features].copy()
    for i, feat in enumerate(graph_feature_names):
        X_train[feat] = train_graph_features[:, i]
    X_train = X_train[all_features].fillna(0)
    
    # Prepare X_test
    X_test = df.loc[test_idx, temporal_features + url_domain_features + text_meta_features].copy()
    for i, feat in enumerate(graph_feature_names):
        X_test[feat] = test_graph_features[:, i]
    X_test = X_test[all_features].fillna(0)
    
    # Get labels
    y_train = df.loc[train_idx, 'label']
    y_test = df.loc[test_idx, 'label']
    
    if verbose:
        print(f"  ✓ Train features: {X_train.shape}")
        print(f"  ✓ Test features: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def engineer_temporal_features_leakage_free(
    df: pd.DataFrame,
    train_idx: pd.Index,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Engineer temporal features without data leakage.
    Risk scores and historical stats are computed ONLY on training data.
    
    Args:
        df: Full dataframe
        train_idx: Training set indices
        verbose: Print progress messages
        
    Returns:
        DataFrame with temporal features added
    """
    if verbose:
        print(f"\n[3.1] Engineering temporal features (leakage-free)...")
    
    df = df.copy()
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Basic temporal features (safe - no leakage)
    if 'hour' not in df.columns:
        df['hour'] = df['date'].dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['date'].dt.dayofweek
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # === LEAKAGE-FREE RISK SCORES ===
    # Compute risk scores ONLY on training data
    train_df = df.loc[train_idx]
    
    # Hour risk score (spam rate by hour in training data)
    hour_spam_rate_train = train_df.groupby('hour')['label'].mean()
    df['hour_risk_score'] = df['hour'].map(hour_spam_rate_train).fillna(
        train_df['label'].mean()  # Fill with overall spam rate
    )
    
    # Weekday risk score (spam rate by day of week in training data)
    weekday_spam_rate_train = train_df.groupby('day_of_week')['label'].mean()
    df['weekday_risk_score'] = df['day_of_week'].map(weekday_spam_rate_train).fillna(
        train_df['label'].mean()
    )
    
    # Region risk score (if timezone_region exists)
    if 'timezone_region' in df.columns:
        region_spam_rate_train = train_df.groupby('timezone_region')['label'].mean()
        df['region_risk_score'] = df['timezone_region'].map(region_spam_rate_train).fillna(
            train_df['label'].mean()
        )
        overall_spam_rate = train_df['label'].mean()
        df['is_high_risk_region'] = (df['region_risk_score'] > overall_spam_rate * 1.2).astype(int)
    else:
        df['region_risk_score'] = train_df['label'].mean()
        df['is_high_risk_region'] = 0
    
    # === LEAKAGE-FREE SENDER STATISTICS ===
    # Sort by date to ensure chronological order
    df_sorted = df.sort_values('date').copy()
    
    # Calculate sender statistics using ONLY training data up to each point
    sender_stats = {}
    sender_spam_count = {}
    sender_total_count = {}
    sender_last_email = {}
    
    for idx in df_sorted.index:
        sender = df_sorted.loc[idx, 'sender']
        current_date = df_sorted.loc[idx, 'date']
        
        # Only use training data that occurred BEFORE current email
        if idx in train_idx:
            # For training data, use historical training data
            historical_train = train_df[
                (train_df['sender'] == sender) & 
                (train_df['date'] < current_date)
            ]
        else:
            # For test data, use ALL training data before current date
            historical_train = train_df[
                (train_df['sender'] == sender) & 
                (train_df['date'] < current_date)
            ]
        
        # Sender historical spam rate
        if len(historical_train) > 0:
            spam_rate = historical_train['label'].mean()
            email_count = len(historical_train)
        else:
            spam_rate = train_df['label'].mean()  # Default to overall spam rate
            email_count = 0
        
        df_sorted.loc[idx, 'sender_historical_spam_rate'] = spam_rate
        df_sorted.loc[idx, 'sender_email_count'] = email_count
        
        # Time since last email
        if len(historical_train) > 0:
            last_email_date = historical_train['date'].max()
            time_diff = (current_date - last_email_date).total_seconds() / 3600  # Hours
            df_sorted.loc[idx, 'time_since_last_email'] = min(time_diff, 168)  # Cap at 1 week
        else:
            df_sorted.loc[idx, 'time_since_last_email'] = 168  # Default to 1 week
    
    # Merge back to original order
    df = df.join(df_sorted[['sender_historical_spam_rate', 'sender_email_count', 'time_since_last_email']])
    
    if verbose:
        print(f"  ✓ Created 10 temporal features (leakage-free)")
        print(f"    - Basic: hour, day_of_week, is_weekend")
        print(f"    - Risk scores: hour_risk_score, weekday_risk_score, region_risk_score")
        print(f"    - Sender stats: sender_historical_spam_rate, sender_email_count, time_since_last_email")
    
    return df


def engineer_url_domain_features_leakage_free(
    df: pd.DataFrame,
    train_idx: pd.Index,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Engineer URL and domain features without data leakage.
    Domain statistics are computed ONLY on training data.
    
    Args:
        df: Full dataframe
        train_idx: Training set indices
        verbose: Print progress messages
        
    Returns:
        DataFrame with URL/domain features added
    """
    if verbose:
        print(f"\n[3.2] Engineering URL/domain features (leakage-free)...")
    
    df = df.copy()
    
    # URL features (safe - no leakage)
    df['urls'] = df['urls'].fillna(0)
    df['has_url'] = (df['urls'] > 0).astype(int)
    df['urls_log'] = np.log1p(df['urls'])
    
    # Time-based feature (safe)
    if 'hour' in df.columns:
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    else:
        df['is_night'] = 0
    
    # Extract sender domain
    df['sender_domain'] = df['sender'].fillna('').astype(str).apply(
        lambda x: x.split('@')[-1] if '@' in x else 'unknown'
    )
    
    # === LEAKAGE-FREE DOMAIN STATISTICS ===
    # Compute domain stats ONLY on training data
    train_df = df.loc[train_idx]
    
    # Sort by date for chronological processing
    df_sorted = df.sort_values('date').copy()
    train_df_sorted = train_df.sort_values('date')
    
    # Calculate domain statistics using cumulative training data
    for idx in df_sorted.index:
        domain = df_sorted.loc[idx, 'sender_domain']
        current_date = df_sorted.loc[idx, 'date']
        
        if idx in train_idx:
            # For training data, use historical training data
            historical_domain = train_df_sorted[
                (train_df_sorted['sender_domain'] == domain) & 
                (train_df_sorted['date'] < current_date)
            ]
        else:
            # For test data, use ALL training data before current date
            historical_domain = train_df_sorted[
                (train_df_sorted['sender_domain'] == domain) & 
                (train_df_sorted['date'] < current_date)
            ]
        
        # Domain spam rate
        if len(historical_domain) > 0:
            spam_rate = historical_domain['label'].mean()
            frequency = len(historical_domain)
        else:
            spam_rate = 0.5  # Unknown domain
            frequency = 0
        
        df_sorted.loc[idx, 'domain_spam_rate'] = spam_rate
        df_sorted.loc[idx, 'domain_frequency'] = frequency
    
    # Merge back
    df = df.join(df_sorted[['domain_spam_rate', 'domain_frequency']])
    
    # Derived features
    df['is_suspicious_domain'] = (df['domain_spam_rate'] > 0.7).astype(int)
    df['is_rare_domain'] = (df['domain_frequency'] <= 5).astype(int)
    
    if verbose:
        print(f"  ✓ Created 8 URL/domain features (leakage-free)")
        print(f"    - URL: urls, has_url, urls_log")
        print(f"    - Domain: domain_spam_rate, is_suspicious_domain, domain_frequency, is_rare_domain")
        print(f"    - Time: is_night")
    
    return df
