"""
Graph Feature Extraction Module
================================
Extracts graph-based features from email network data for spam detection.
Excludes: total_degree, triangles
"""

import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class EmailGraphFeatureExtractor:
    """
    Extracts graph-based features from email communication networks.
    
    Features extracted:
    - out_degree: Number of unique recipients
    - in_degree: Number of unique senders who emailed this address
    - total_sent: Total number of emails sent
    - reciprocity: Proportion of recipients who replied back
    - clustering: Clustering coefficient (social embeddedness)
    - eigenvector: Eigenvector centrality (network importance)
    - closeness: Closeness centrality (network reachability)
    - avg_weight: Average number of emails per recipient
    - is_spammer: Binary label (1 if >80% spam rate)
    """
    
    def __init__(self):
        self.G = None
        self.df = None
        self.features_df = None
        
    def load_and_clean_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load and clean email data from CSV.
        
        Args:
            csv_path: Path to the CSV file containing email data
            
        Returns:
            Cleaned dataframe
        """
        print(f"Loading data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"Dataset loaded: {len(self.df):,} emails")
        print(f"  Columns: {self.df.columns.tolist()}")
        print(f"  Label distribution:\n{self.df['label'].value_counts()}")
        print(f"  Spam rate: {self.df['label'].mean()*100:.1f}%\n")
        
        # Clean data
        self.df = self.df.dropna(subset=['sender', 'receiver'])
        self.df['sender'] = self.df['sender'].astype(str).str.strip().str.lower()
        self.df['receiver'] = self.df['receiver'].astype(str).str.strip().str.lower()
        self.df = self.df[(self.df['sender'] != 'nan') & (self.df['receiver'] != 'nan')]
        
        print(f"After cleaning: {len(self.df):,} emails\n")
        return self.df
    
    def build_graph(self) -> nx.DiGraph:
        """
        Build directed graph from email data.
        
        Returns:
            NetworkX directed graph
        """
        print("Building email network graph...")
        self.G = nx.DiGraph()
        
        for idx, row in self.df.iterrows():
            sender = row['sender']
            receiver = row['receiver']
            is_spam = row['label']
            
            if self.G.has_edge(sender, receiver):
                self.G[sender][receiver]['weight'] += 1
                self.G[sender][receiver]['spam_count'] += is_spam
                self.G[sender][receiver]['ham_count'] += (1 - is_spam)
            else:
                self.G.add_edge(sender, receiver, 
                               weight=1, 
                               spam_count=is_spam,
                               ham_count=1-is_spam)
        
        print(f"{'='*60}")
        print(f" GRAPH STATISTICS:")
        print(f"{'='*60}")
        print(f"  Nodes (unique emails):     {self.G.number_of_nodes():,}")
        print(f"  Edges (connections):       {self.G.number_of_edges():,}")
        print(f"  Network density:           {nx.density(self.G):.6f}")
        print(f"  Avg degree:                {sum(dict(self.G.degree()).values())/self.G.number_of_nodes():.2f}\n")
        
        return self.G
    
    def extract_basic_features(self) -> pd.DataFrame:
        """
        Extract basic graph features for each sender.
        
        Returns:
            DataFrame with basic features
        """
        print("[1/2] Extracting basic features from graph...")
        
        out_degrees = dict(self.G.out_degree())
        in_degrees = dict(self.G.in_degree())
        sender_features = []
        
        # Group by sender for efficiency
        for sender, group in self.df.groupby('sender'):
            if sender not in self.G:
                continue
            
            out_deg = out_degrees.get(sender, 0)
            if out_deg == 0:  # Skip non-senders
                continue
            
            in_deg = in_degrees.get(sender, 0)
            
            # Email statistics
            total_sent = len(group)
            spam_sent = group['label'].sum()
            
            # Reciprocity: How many recipients have mutual connections?
            unique_receivers = set(group['receiver'])
            mutual_count = sum(1 for receiver in unique_receivers if self.G.has_edge(receiver, sender))
            reciprocity = mutual_count / len(unique_receivers) if unique_receivers else 0
            
            sender_features.append({
                'sender': sender,
                'out_degree': out_deg,
                'in_degree': in_deg,
                'total_sent': total_sent,
                'reciprocity': reciprocity,
                'is_spammer': 1 if spam_sent / total_sent > 0.8 else 0
            })
        
        self.features_df = pd.DataFrame(sender_features)
        print(f"✓ Basic features extracted for {len(self.features_df):,} senders\n")
        
        return self.features_df
    
    def extract_advanced_features(self) -> pd.DataFrame:
        """
        Extract advanced graph features (clustering, centrality, etc.).
        
        Returns:
            DataFrame with all features
        """
        print("[2/2] Extracting advanced features...")
        
        # Clustering coefficient
        print("  - Computing clustering coefficient...")
        clustering = nx.clustering(self.G.to_undirected())
        
        # Centrality metrics
        print("  - Computing closeness centrality...")
        closeness = nx.closeness_centrality(self.G)
        
        print("  - Computing eigenvector centrality...")
        try:
            eigenvector = nx.eigenvector_centrality(self.G, max_iter=100)
        except:
            print("    Warning: Eigenvector centrality failed, using zeros")
            eigenvector = {node: 0 for node in self.G.nodes()}
        
        # Average weight
        print("  - Computing average edge weights...")
        
        # Add advanced features to dataframe
        self.features_df['clustering'] = self.features_df['sender'].map(lambda x: clustering.get(x, 0))
        self.features_df['eigenvector'] = self.features_df['sender'].map(lambda x: eigenvector.get(x, 0))
        self.features_df['closeness'] = self.features_df['sender'].map(lambda x: closeness.get(x, 0))
        self.features_df['avg_weight'] = self.features_df['sender'].map(
            lambda s: np.mean([self.G[s][r]['weight'] for r in self.G.successors(s)]) if list(self.G.successors(s)) else 0
        )
        
        print("✓ All graph features computed!\n")
        
        return self.features_df
    
    def get_features(self, csv_path: str) -> pd.DataFrame:
        """
        Complete pipeline: load data, build graph, extract all features.
        
        Args:
            csv_path: Path to the CSV file containing email data
            
        Returns:
            DataFrame with all extracted features
        """
        self.load_and_clean_data(csv_path)
        self.build_graph()
        self.extract_basic_features()
        self.extract_advanced_features()
        
        print("="*70)
        print(" FEATURE EXTRACTION COMPLETE")
        print("="*70)
        print(f"Total features: {len(self.features_df.columns)}")
        print(f"Senders analyzed: {len(self.features_df):,}")
        print(f"Spammers: {self.features_df['is_spammer'].sum():,}")
        print(f"Legitimate: {(self.features_df['is_spammer'] == 0).sum():,}")
        print(f"\nFeatures: {list(self.features_df.columns)}\n")
        
        return self.features_df
    
    def get_feature_comparison(self) -> pd.DataFrame:
        """
        Compare features between spammers and legitimate senders.
        
        Returns:
            DataFrame with feature comparison statistics
        """
        if self.features_df is None:
            raise ValueError("Features not extracted yet. Run get_features() first.")
        
        print(f"{'='*80}")
        print(f" FEATURES COMPARISON: SPAMMER vs LEGITIMATE")
        print(f"{'='*80}\n")
        
        # Features to compare (excluding sender and is_spammer)
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['sender', 'is_spammer']]
        
        spam_users = self.features_df[self.features_df['is_spammer'] == 1]
        legit_users = self.features_df[self.features_df['is_spammer'] == 0]
        
        comparison_rows = []
        for feature in feature_cols:
            spam_mean = spam_users[feature].mean()
            legit_mean = legit_users[feature].mean()
            
            ratio = spam_mean / legit_mean if legit_mean > 0 else float('inf')
            
            comparison_rows.append({
                'Feature': feature,
                'Spam Mean': spam_mean,
                'Legit Mean': legit_mean,
                'Ratio': ratio,
                'Interpretation': "🔴 Spam higher" if ratio > 1.2 else ("🟢 Legit higher" if ratio < 0.8 else "⚪ Similar")
            })
        
        comparison_df = pd.DataFrame(comparison_rows)
        print(comparison_df.to_string(index=False))
        
        # Top discriminative features
        print(f"\n{'─'*80}")
        print(f" TOP DISCRIMINATIVE FEATURES:")
        print(f"{'─'*80}")
        
        comparison_df['abs_diff'] = abs(comparison_df['Ratio'] - 1.0)
        top_features = comparison_df.nlargest(len(feature_cols), 'abs_diff')
        print(top_features[['Feature', 'Spam Mean', 'Legit Mean', 'Ratio', 'Interpretation']].to_string(index=False))
        
        return comparison_df


# Convenience function for quick usage
def extract_features(csv_path: str) -> pd.DataFrame:
    """
    Quick function to extract all features from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing email data
        
    Returns:
        DataFrame with all extracted features
        
    Example:
        >>> features_df = extract_features("data/processed/graph_merge.csv")
    """
    extractor = EmailGraphFeatureExtractor()
    return extractor.get_features(csv_path)


if __name__ == "__main__":
    # Example usage
    print("Email Graph Feature Extraction Module")
    print("="*70)
    print("\nUsage example:")
    print("  from feature_extraction import extract_features")
    print("  features_df = extract_features('path/to/your/data.csv')")
    print("\nOr use the class for more control:")
    print("  extractor = EmailGraphFeatureExtractor()")
    print("  features_df = extractor.get_features('path/to/your/data.csv')")
    print("  comparison = extractor.get_feature_comparison()")
