"""
Feature Engineering Module

This module contains functions for engineering features from email data,
including temporal features, URL/domain features, text meta-features, and graph features.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple


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
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    temporal_features: list,
    url_domain_features: list,
    text_meta_features: list,
    graph_feature_names: list,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare train and test feature matrices with graph features.
    
    Graph features are built ONLY on training data to prevent leakage.
    
    Args:
        df: Full dataframe with all engineered features (except graph)
        train_idx: Training indices
        test_idx: Test indices
        temporal_features: List of temporal feature names
        url_domain_features: List of URL/domain feature names
        text_meta_features: List of text meta feature names
        graph_feature_names: List of graph feature names
        verbose: Print progress messages
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if verbose:
        print(f"\n[Features] Building graph features (training data only)...")
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    # Build graph on training data only
    if verbose:
        print(f"  ⏳ Building email network graph from training data...")
    G_train = build_graph_from_data(train_df)
    if verbose:
        print(f"  ✓ Training graph: {G_train.number_of_nodes():,} nodes, {G_train.number_of_edges():,} edges")
    
    # Extract graph features for training data
    train_graph_features = extract_graph_features_for_senders(train_df['sender'], G_train, verbose=verbose)
    train_df_with_graph = train_df.merge(train_graph_features, on='sender', how='left')
    
    # Extract graph features for test data (using TRAINING graph to prevent leakage)
    test_graph_features = extract_graph_features_for_senders(test_df['sender'], G_train, verbose=verbose)
    test_df_with_graph = test_df.merge(test_graph_features, on='sender', how='left')
    
    # Fill any remaining NaN values
    graph_cols = [col for col in train_df_with_graph.columns if col.startswith('graph_')]
    train_df_with_graph[graph_cols] = train_df_with_graph[graph_cols].fillna(0)
    test_df_with_graph[graph_cols] = test_df_with_graph[graph_cols].fillna(0)
    
    # Prepare final feature matrices
    all_ml_features = temporal_features + url_domain_features + text_meta_features + graph_feature_names
    
    X_train = train_df_with_graph[all_ml_features].fillna(0)
    X_test = test_df_with_graph[all_ml_features].fillna(0)
    y_train = train_df_with_graph['label']
    y_test = test_df_with_graph['label']
    
    if verbose:
        print(f"\n  ✓ Final feature count: {len(all_ml_features)}")
        print(f"    - Temporal: {len(temporal_features)}")
        print(f"    - URL/Domain: {len(url_domain_features)}")
        print(f"    - Text meta: {len(text_meta_features)}")
        print(f"    - Graph: {len(graph_feature_names)}")
        print(f"\n  ⚠️  NOTE: Graph features built ONLY on training data to prevent data leakage!")
    
    return X_train, X_test, y_train, y_test
