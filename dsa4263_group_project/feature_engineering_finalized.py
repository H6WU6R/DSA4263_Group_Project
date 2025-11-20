# Feature Engineering Pipeline with Multiprocessing Support
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import warnings
from dsa4263_group_project.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from typing import Optional, Tuple

warnings.filterwarnings('ignore')


class FeatureEngineer:
	"""
	Template for email feature engineering.
	Includes:
	  - Graph features (PIT-safe with network analysis)
	  - Time series features (PIT-safe temporal patterns)
	  - Text/NLP features (meta-features, sentiment, domain analysis)
	
	Supports both sequential and parallel execution via multiprocessing.
	"""
	
	def __init__(self, verbose: bool = True):
		"""
		Initialize the feature engineer.
		
		Args:
			verbose: Whether to print progress messages
		"""
		self.verbose = verbose
		self._nltk_initialized = False
	
	def _log(self, message: str):
		"""Print message if verbose mode is enabled."""
		if self.verbose:
			print(message)
	
	def _initialize_nltk(self):
		"""Initialize NLTK resources (download if needed)."""
		if self._nltk_initialized:
			return
		
		try:
			import nltk
			# Check and download required NLTK data
			required_data = [
				('tokenizers/punkt', 'punkt'),
				('corpora/stopwords', 'stopwords'),
				('corpora/wordnet', 'wordnet')
			]
			
			for path, name in required_data:
				try:
					nltk.data.find(path)
				except LookupError:
					if self.verbose:
						print(f"    • Downloading NLTK data: {name}")
					nltk.download(name, quiet=not self.verbose)
			
			self._nltk_initialized = True
		except ImportError:
			if self.verbose:
				print("    ⚠️  WARNING: NLTK not available. Text tokenization will use simple splitting.")
	
	# ============================================================================
	# GRAPH FEATURE ENGINEERING
	# ============================================================================
	
	def _compute_node_features(self, G: nx.DiGraph, node: str) -> dict:
		"""
		Compute graph features for a specific node.
		
		Args:
			G: NetworkX directed graph
			node: Node identifier (sender email)
		
		Returns:
			Dictionary of graph features
		"""
		features = {
			'out_degree': 0,
			'in_degree': 0,
			'total_degree': 0,
			'reciprocity': 0.0,
			'avg_weight': 0.0,
			'clustering': 0.0,
			'eigenvector': 0.0,
			'closeness': 0.0
		}
		
		if node not in G:
			return features
		
		out_deg = G.out_degree(node)
		in_deg = G.in_degree(node)
		
		features['out_degree'] = out_deg
		features['in_degree'] = in_deg
		features['total_degree'] = out_deg + in_deg
		
		# Compute reciprocity: How many recipients have mutual connections?
		# (following feature_extraction.py formula)
		receivers = list(G.successors(node))
		if receivers:
			reciprocal_count = sum(1 for r in receivers if G.has_edge(r, node))
			features['reciprocity'] = reciprocal_count / len(receivers)
			
			# Average weight of outgoing edges
			weights = [G[node][r]['weight'] for r in receivers]
			features['avg_weight'] = np.mean(weights)
		
		# Compute clustering coefficient (only if node has degree > 0)
		if G.number_of_nodes() > 1:
			try:
				G_undirected = G.to_undirected()
				features['clustering'] = nx.clustering(G_undirected, node)
			except:
				features['clustering'] = 0.0
		
		# Compute closeness centrality
		if G.number_of_nodes() > 1:
			try:
				closeness = nx.closeness_centrality(G)
				features['closeness'] = closeness.get(node, 0.0)
			except:
				features['closeness'] = 0.0
		
		# Compute eigenvector centrality
		if G.number_of_nodes() > 1:
			try:
				eigenvector = nx.eigenvector_centrality(G, max_iter=100)
				features['eigenvector'] = eigenvector.get(node, 0.0)
			except:
				features['eigenvector'] = 0.0
		
		return features
	
	def _compute_sender_history(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Compute sender historical statistics (PIT-safe).
		
		Args:
			df: DataFrame sorted by date with 'sender' and 'label' columns
		
		Returns:
			DataFrame with historical features added
		"""
		df = df.copy()
		
		# Sort by sender and date to ensure PIT-safe computation
		df = df.sort_values(['sender', 'date']).reset_index(drop=True)
		
		# Historical email count (cumulative, excluding current)
		df['sender_historical_email_count'] = df.groupby('sender').cumcount()
		
		# Historical spam count (cumulative sum of past emails, excluding current)
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
		
		# Time since last email (in seconds)
		df['sender_time_since_last_email'] = (
			df.groupby('sender')['date']
			.diff()
			.dt.total_seconds()
			.fillna(-1)  # -1 for first email from sender
		)
		
		return df
	
	def compute_graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Compute graph features using point-in-time methods.
		Builds network incrementally to ensure no data leakage.
		
		Args:
			df: DataFrame with 'sender', 'receiver', 'date', 'label' columns
		
		Returns:
			DataFrame with graph features added
		"""
		self._log("\n" + "="*80)
		self._log("GRAPH FEATURE ENGINEERING (PIT)")
		self._log("="*80)
		
		df = df.copy()
		df = df.sort_values('date').reset_index(drop=True)
		
		# Initialize feature columns
		graph_feature_cols = [
			'sender_out_degree',
			'sender_in_degree',
			'sender_total_degree',
			'sender_reciprocity',
			'sender_avg_weight',
			'sender_clustering',
			'sender_eigenvector',
			'sender_closeness'
		]
		
		for col in graph_feature_cols:
			df[col] = 0.0
		
		# Build graph incrementally (PIT-safe)
		G = nx.DiGraph()
		
		for idx in tqdm(range(len(df)), desc="Computing graph features", disable=not self.verbose):
			current_sender = df.loc[idx, 'sender']
			current_receiver = df.loc[idx, 'receiver']
			
			# Compute features from PAST graph state (before adding current edge)
			features = self._compute_node_features(G, current_sender)
			
			df.loc[idx, 'sender_out_degree'] = features['out_degree']
			df.loc[idx, 'sender_in_degree'] = features['in_degree']
			df.loc[idx, 'sender_total_degree'] = features['total_degree']
			df.loc[idx, 'sender_reciprocity'] = features['reciprocity']
			df.loc[idx, 'sender_avg_weight'] = features['avg_weight']
			df.loc[idx, 'sender_clustering'] = features['clustering']
			df.loc[idx, 'sender_eigenvector'] = features['eigenvector']
			df.loc[idx, 'sender_closeness'] = features['closeness']
			
			# Add current edge to graph for future iterations
			if G.has_edge(current_sender, current_receiver):
				G[current_sender][current_receiver]['weight'] += 1
			else:
				G.add_edge(current_sender, current_receiver, weight=1)
		
		# Add sender historical statistics
		df = self._compute_sender_history(df)
		
		self._log(f"✅ Graph features computed: {len(graph_feature_cols) + 4} features")
		self._log(f"   Basic: out_degree, in_degree, total_degree, reciprocity, avg_weight")
		self._log(f"   Centrality: clustering, eigenvector, closeness")
		self._log(f"   Historical: email_count, spam_count, spam_rate, time_since_last_email")
		
		return df
	
	# ============================================================================
	# TIME SERIES FEATURE ENGINEERING
	# ============================================================================
	
	def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Extract basic temporal features from date column.
		
		Args:
			df: DataFrame with 'date' column
		
		Returns:
			DataFrame with temporal features added
		"""
		df = df.copy()
		
		df['hour'] = df['date'].dt.hour
		df['day_of_week'] = df['date'].dt.dayofweek
		df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
		df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
		
		# Timezone region if available
		if 'timezone_region' in df.columns:
			df['is_middle_east'] = (df['timezone_region'] == 'Middle East/South Asia').astype(int)
		else:
			df['is_middle_east'] = 0
		
		return df
	
	def _compute_pit_risk_score(self, df: pd.DataFrame, group_col: str, value_col: str = 'label') -> pd.Series:
		"""
		Compute point-in-time risk score for a grouping column.
		Uses expanding window on past data only.
		
		Args:
			df: DataFrame sorted by date
			group_col: Column to group by (e.g., 'hour', 'day_of_week')
			value_col: Column to compute risk score for (default: 'label')
		
		Returns:
			Series with risk scores
		"""
		df = df.sort_values('date').reset_index(drop=True)
		
		# For each group, compute expanding mean of past values
		risk_scores = (
			df.groupby(group_col)[value_col]
			.apply(lambda x: x.shift().expanding().mean())
			.reset_index(level=0, drop=True)
		)
		
		# Fill NaN (first occurrence of each group) with a past-only global prior
		global_prior = df[value_col].expanding().mean().shift(1).fillna(0.0)
		risk_scores = risk_scores.fillna(global_prior)
		
		return risk_scores
	
	def _compute_sender_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Compute sender-level temporal features (PIT-safe).
		
		Args:
			df: DataFrame sorted by sender and date
		
		Returns:
			DataFrame with sender temporal features added
		"""
		df = df.copy()
		df = df.sort_values(['sender', 'date']).reset_index(drop=True)
		
		# Historical counts
		df['sender_historical_count'] = df.groupby('sender').cumcount()
		
		df['sender_historical_spam_count'] = (
			df.groupby('sender')['label']
			.apply(lambda x: x.shift().cumsum().fillna(0))
			.reset_index(level=0, drop=True)
		)
		
		# Historical phishing rate
		global_prior = df['label'].expanding().mean().shift(1).fillna(0.0)
		df['sender_historical_phishing_rate'] = np.where(
			df['sender_historical_count'] > 0,
			df['sender_historical_spam_count'] / df['sender_historical_count'],
			global_prior
		)
		
		# Time gap between emails from same sender
		df['sender_time_gap'] = (
			df.groupby('sender')['date']
			.diff()
			.dt.total_seconds()
			.fillna(-1)
		)
		
		# Standard deviation of time gaps (expanding window)
		df['sender_time_gap_std'] = (
			df.groupby('sender')['sender_time_gap']
			.apply(lambda x: x.shift().expanding().std())
			.reset_index(level=0, drop=True)
			.fillna(0)
		)
		
		# Sender lifespan (days since first email)
		df['sender_first_date'] = df.groupby('sender')['date'].transform('first')
		df['sender_lifespan_days'] = (
			(df['date'] - df['sender_first_date']).dt.total_seconds() / 86400
		)
		df = df.drop(columns=['sender_first_date'])
		
		return df
	
	def compute_timeseries_features(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Compute time series features using point-in-time methods.
		
		Args:
			df: DataFrame with 'date', 'sender', 'label' columns
		
		Returns:
			DataFrame with time series features added
		"""
		self._log("\n" + "="*80)
		self._log("TIME SERIES FEATURE ENGINEERING (PIT)")
		self._log("="*80)
		
		df = df.copy()
		df = df.sort_values('date').reset_index(drop=True)
		
		# Basic temporal features
		df = self._extract_temporal_features(df)
		
		# PIT-safe risk scores
		self._log("    • Computing risk scores...")
		df['hour_risk_score'] = self._compute_pit_risk_score(df, 'hour', 'label')
		df['weekday_risk_score'] = self._compute_pit_risk_score(df, 'day_of_week', 'label')
		
		if 'timezone_region' in df.columns:
			df['region_risk_score'] = self._compute_pit_risk_score(df, 'timezone_region', 'label')
			df['region_hour'] = df['timezone_region'].astype(str) + '_' + df['hour'].astype(str)
			df['region_hour_risk'] = self._compute_pit_risk_score(df, 'region_hour', 'label')
			df = df.drop(columns=['region_hour'])
		else:
			df['region_risk_score'] = df['label'].mean()
			df['region_hour_risk'] = df['label'].mean()
		
		# Sender temporal features
		self._log("    • Computing sender temporal features...")
		df = self._compute_sender_temporal_features(df)
		
		# Re-sort by date for consistency
		df = df.sort_values('date').reset_index(drop=True)
		
		self._log(f"✅ Time series features computed: 16+ features")
		
		return df
	
	# ============================================================================
	# TEXT/NLP FEATURE ENGINEERING
	# ============================================================================
	
	def _clean_text_for_features(self, text: str) -> str:
		"""
		Clean text for NLP feature extraction.
		Uses the full cleaning pipeline: lowercase, punctuation removal, 
		tokenization, stopword removal, lemmatization, stemming.
		
		Args:
			text: Raw text string
		
		Returns:
			Cleaned text string
		"""
		if pd.isna(text) or text == "":
			return ""
		
		import re
		
		# Lowercase
		text = str(text).lower()
		
		# Remove punctuation and numbers
		text = re.sub(r'[^a-zA-Z\s]', ' ', text)
		text = re.sub(r'\s+', ' ', text).strip()
		
		# Tokenize
		try:
			from nltk.tokenize import word_tokenize
			tokens = word_tokenize(text)
		except:
			tokens = text.split()
		
		# Remove stopwords
		try:
			from nltk.corpus import stopwords
			stop_words = set(stopwords.words('english'))
			tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
		except:
			pass
		
		# Lemmatize and stem
		try:
			from nltk.stem import WordNetLemmatizer, PorterStemmer
			lemmatizer = WordNetLemmatizer()
			stemmer = PorterStemmer()
			tokens = [lemmatizer.lemmatize(word) for word in tokens]
			tokens = [stemmer.stem(word) for word in tokens]
		except:
			pass
		
		return ' '.join(tokens)
	
	def _extract_text_meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Extract meta-features from text (length, character counts, ratios).
		
		Args:
			df: DataFrame with 'subject', 'body', 'full_text' columns
		
		Returns:
			DataFrame with text meta-features added
		"""
		df = df.copy()
		
		# Length features
		df['subject_length'] = df['subject'].str.len()
		df['body_length'] = df['body'].str.len()
		df['text_length'] = df['full_text'].str.len()
		df['word_count'] = df['full_text'].str.split().str.len()
		
		# Character count features
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
		
		# Digit features
		df['digit_count'] = df['full_text'].str.count(r'\d')
		df['digit_ratio'] = df['digit_count'] / (df['text_length'] + 1)
		
		# Average word length
		df['avg_word_length'] = np.where(
			df['word_count'] > 0,
			df['text_length'] / df['word_count'],
			0
		)
		
		return df
	
	def _extract_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Extract sentiment features using VADER.
		
		Args:
			df: DataFrame with 'subject' and 'body' columns
		
		Returns:
			DataFrame with sentiment features added
		"""
		df = df.copy()
		
		try:
			from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
			analyzer = SentimentIntensityAnalyzer()
			
			self._log("    • Computing sentiment scores...")
			
			df['subject_sentiment'] = df['subject'].apply(
				lambda x: analyzer.polarity_scores(str(x) if pd.notna(x) else "")['compound']
			)
			df['body_sentiment'] = df['body'].apply(
				lambda x: analyzer.polarity_scores(str(x) if pd.notna(x) else "")['compound']
			)
			
			self._log(f"      ✓ Sentiment computed (mean subject: {df['subject_sentiment'].mean():.3f}, "
			         f"mean body: {df['body_sentiment'].mean():.3f})")
		
		except Exception as e:
			self._log(f"    ⚠️  WARNING: Sentiment analysis failed: {e}")
			self._log(f"       Setting sentiment scores to 0. Install: pip install vaderSentiment")
			df['subject_sentiment'] = 0.0
			df['body_sentiment'] = 0.0
		
		return df
	
	def _extract_url_features(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Extract URL-related features.
		
		Args:
			df: DataFrame with 'urls' column
		
		Returns:
			DataFrame with URL features added
		"""
		df = df.copy()
		
		df['urls'] = df['urls'].fillna(0)
		df['has_url'] = (df['urls'] > 0).astype(int)
		df['urls_log'] = np.log1p(df['urls'])
		
		return df
	
	def _extract_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Extract domain-related features (PIT-safe).
		
		Args:
			df: DataFrame with 'sender', 'date', 'label' columns
		
		Returns:
			DataFrame with domain features added
		"""
		df = df.copy()
		
		# Extract domain from sender
		df['sender_domain'] = df['sender'].str.split('@').str[-1]
		
		# Sort by date for PIT-safe computation
		df = df.sort_values('date').reset_index(drop=True)
		
		# Domain historical counts (cumulative, excluding current)
		df['domain_email_count'] = df.groupby('sender_domain').cumcount()
		
		# Domain historical spam count
		df['domain_spam_cumsum'] = (
			df.groupby('sender_domain')['label']
			.apply(lambda x: x.shift().cumsum().fillna(0))
			.reset_index(level=0, drop=True)
		)
		
		# Domain spam rate
		global_prior = df['label'].expanding().mean().shift(1).fillna(0.0)
		df['domain_spam_rate'] = np.where(
			df['domain_email_count'] > 0,
			df['domain_spam_cumsum'] / df['domain_email_count'],
			global_prior
		)
		
		# Derived features
		df['is_suspicious_domain'] = (df['domain_spam_rate'] > 0.7).astype(int)
		df['domain_frequency'] = df['domain_email_count']
		df['is_rare_domain'] = (df['domain_frequency'] <= 5).astype(int)
		
		# Drop intermediate columns
		df = df.drop(columns=['domain_spam_cumsum'], errors='ignore')
		
		return df
	
	def compute_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Compute text/NLP features including meta-features, sentiment, and domain analysis.
		
		Args:
			df: DataFrame with 'subject', 'body', 'urls', 'sender', 'date', 'label' columns
		
		Returns:
			DataFrame with text features added
		"""
		self._log("\n" + "="*80)
		self._log("TEXT/NLP FEATURE ENGINEERING")
		self._log("="*80)
		
		df = df.copy()
		
		# Initialize NLTK
		self._initialize_nltk()
		
		# Fill missing text
		df['subject'] = df['subject'].fillna('')
		df['body'] = df['body'].fillna('')
		df['full_text'] = df['subject'] + ' ' + df['body']
		
		# Clean text
		self._log("    • Cleaning text...")
		df['cleaned_text'] = df['full_text'].apply(self._clean_text_for_features)
		
		# Extract features
		self._log("    • Extracting meta-features...")
		df = self._extract_text_meta_features(df)
		
		df = self._extract_sentiment_features(df)
		
		self._log("    • Extracting URL features...")
		df = self._extract_url_features(df)
		
		self._log("    • Extracting domain features (PIT)...")
		df = self._extract_domain_features(df)
		
		# Fill any remaining NaNs in numeric columns
		numeric_cols = df.select_dtypes(include=[np.number]).columns
		for col in numeric_cols:
			if df[col].isna().any():
				df[col] = df[col].fillna(0)
		
		self._log(f"✅ Text/NLP features computed: 25+ features")
		
		return df
	
	# ============================================================================
	# UTILITY METHODS
	# ============================================================================
	
	def save_features(self, df: pd.DataFrame, output_path: str):
		"""
		Save feature-engineered DataFrame to CSV.
		
		Args:
			df: DataFrame with features
			output_path: Path to save CSV file
		"""
		df.to_csv(PROCESSED_DATA_DIR / output_path, index=False)
		self._log(f"💾 Features saved to: {output_path}")


# ============================================================================
# MULTIPROCESSING WRAPPER FUNCTIONS
# ============================================================================

def compute_graph_features_parallel(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
	"""
	Wrapper function for parallel execution of graph feature engineering.
	
	Args:
		df: Input DataFrame
	
	Returns:
		Tuple of (DataFrame with features, feature type name)
	"""
	engineer = FeatureEngineer(verbose=True)
	df_features = engineer.compute_graph_features(df)
	return df_features, "graph"


def compute_timeseries_features_parallel(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
	"""
	Wrapper function for parallel execution of time series feature engineering.
	
	Args:
		df: Input DataFrame
	
	Returns:
		Tuple of (DataFrame with features, feature type name)
	"""
	engineer = FeatureEngineer(verbose=True)
	df_features = engineer.compute_timeseries_features(df)
	return df_features, "timeseries"


def compute_text_features_parallel(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
	"""
	Wrapper function for parallel execution of text feature engineering.
	
	Args:
		df: Input DataFrame
	
	Returns:
		Tuple of (DataFrame with features, feature type name)
	"""
	engineer = FeatureEngineer(verbose=True)
	df_features = engineer.compute_text_features(df)
	return df_features, "text"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
	import time
	from multiprocessing import Pool
	
	print("\n" + "="*80)
	print("FEATURE ENGINEERING PIPELINE DEMO")
	print("="*80)
	
	# Example: Sequential execution
	print("\n📌 SEQUENTIAL EXECUTION")
	print("-" * 80)
	
	# Load your data
	df = pd.read_csv(PROCESSED_DATA_DIR / "cleaned_date_merge.csv")
	df['date'] = pd.to_datetime(df['date'], errors='coerce')
	df = df.dropna(subset=['sender', 'receiver', 'date', 'label'])
	df = df.sort_values('date').reset_index(drop=True)
	
	engineer = FeatureEngineer(verbose=True)
	
	# Compute features sequentially
	start = time.time()
	df_graph = engineer.compute_graph_features(df.copy())
	df_ts = engineer.compute_timeseries_features(df.copy())
	df_text = engineer.compute_text_features(df.copy())
	elapsed_sequential = time.time() - start
	
	print(f"\n⏱️  Sequential time: {elapsed_sequential:.2f} seconds")
	
	# Example: Parallel execution
	print("\n" + "="*80)
	print("📌 PARALLEL EXECUTION (3 processes)")
	print("-" * 80)
	
	start = time.time()
	
	with Pool(processes=3) as pool:
		results = [
			pool.apply_async(compute_graph_features_parallel, (df.copy(),)),
			pool.apply_async(compute_timeseries_features_parallel, (df.copy(),)),
			pool.apply_async(compute_text_features_parallel, (df.copy(),))
		]
		results = [r.get() for r in results]
	
	elapsed_parallel = time.time() - start
	
	# Extract results
	df_graph_parallel = results[0][0]
	df_ts_parallel = results[1][0]
	df_text_parallel = results[2][0]
	
	print(f"\n⏱️  Parallel time: {elapsed_parallel:.2f} seconds")
	print(f"🚀 Speedup: {elapsed_sequential/elapsed_parallel:.2f}x")
	
	# Save features
	engineer.save_features(df_graph_parallel, "graph_features_pit.csv")
	engineer.save_features(df_ts_parallel, "timeseries_features_pit.csv")
	engineer.save_features(df_text_parallel, "text_features_pit.csv")
	
	print("\n✅ Feature engineering complete!")
