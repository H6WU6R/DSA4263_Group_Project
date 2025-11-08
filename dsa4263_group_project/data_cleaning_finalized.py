# Template class for data cleaning (date and text)
import pandas as pd
import re

class DataCleaner:
	"""
	Template for email data cleaning.
	Includes:
	  - Date cleaning (timezone, parsing, anomaly removal)
	  - Text cleaning (body, subject, etc.)
	"""
	def __init__(self, stop_words=None, lemmatizer=None):
		self.stop_words = stop_words
		self.lemmatizer = lemmatizer

	def clean_and_merge(self, df, sender_col = 'sender', receiver_col = 'receiver'):
		"""
		Fill missing receiver values with unique labels and drop rows where sender is missing
		"""
		df = df.copy()
		
		na_indices = df[df[receiver_col].isnull()].index

		for i, idx in enumerate(na_indices, 1):
			df.at[idx, receiver_col] = f"na{i}"
		
		df = df.dropna(subset=[sender_col])

		return df
	
	def _parse_timezone_offset(self, tz_str):
		"""Convert timezone string like '+0800' or '-0700' to hours."""
		if pd.isna(tz_str):
			return 0.0
		try:
			sign = 1 if tz_str[0] == '+' else -1
			hours = int(tz_str[1:3])
			minutes = int(tz_str[3:5])
			return sign * (hours + minutes / 60.0)
		except (ValueError, IndexError):
			return 0.0

	def _get_simple_region(self, offset_hours):
		"""Map timezone offset to region."""
		if pd.isna(offset_hours):
			return 'Unknown'
		elif -12 <= offset_hours < -4:
			return 'Americas'
		elif -4 <= offset_hours <= 2:
			return 'Europe/Africa'
		elif 2 < offset_hours <= 6:
			return 'Middle East/South Asia'
		elif 6 < offset_hours <= 10:
			return 'APAC'
		elif 10 < offset_hours <= 14:
			return 'Oceania/Pacific'
		else:
			return 'Unknown'

	def clean_dates(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
		"""
		Clean and standardize date columns, extract timezone, remove anomalies.
		"""
		df = df.copy()
		df['timezone_offset'] = df[date_col].astype(str).str.extract(r'([+-]\d{4})$')[0]
		df['timezone_hours'] = df['timezone_offset'].apply(self._parse_timezone_offset)
		try:
			df[date_col] = pd.to_datetime(df[date_col], format='%a, %d %b %Y %H:%M:%S %z', errors='coerce', utc=True)
		except Exception:
			df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
		df[date_col] = df[date_col].dt.tz_localize(None)
		df['timezone_region'] = df['timezone_hours'].apply(self._get_simple_region)
		min_valid_date = pd.Timestamp('1990-01-01')
		max_valid_date = pd.Timestamp('2025-12-31')
		df = df[(df[date_col] >= min_valid_date) & (df[date_col] <= max_valid_date)].copy()
		df = df.drop(columns=['timezone_offset', 'timezone_hours'], errors='ignore')
		return df

	def _remove_punctuation_numbers(self, text):
		"""Remove punctuation and numbers, keeping only letters and spaces."""
		text = re.sub(r'[^a-zA-Z\s]', ' ', str(text))
		text = re.sub(r'\s+', ' ', text).strip()
		return text

	def _tokenize_text(self, text):
		"""Tokenize text using NLTK word_tokenize."""
		try:
			from nltk.tokenize import word_tokenize
			tokens = word_tokenize(str(text))
			return ' '.join(tokens)
		except ImportError:
			# Fallback to simple split if NLTK not available
			return text

	def _remove_stopwords(self, text):
		"""Remove stopwords from text."""
		if not self.stop_words:
			return text
		tokens = text.split()
		filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words and len(word) > 2]
		return ' '.join(filtered_tokens)

	def _lemmatize_text(self, text):
		"""Lemmatize text tokens."""
		if not self.lemmatizer:
			return text
		tokens = text.split()
		lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
		return ' '.join(lemmatized_tokens)
	
	def clean_text(self, text: str) -> str:
		"""
		Clean email text using all cleaning steps from notebooks:
		1. Handle NaN/empty
		2. Lowercase
		3. Remove punctuation and numbers
		4. Tokenize (NLTK word_tokenize)
		5. Remove stopwords
		6. Lemmatize
		"""
		# Step 1: Handle NaN/empty
		if pd.isna(text) or text == "":
			return ""
		
		# Step 2: Lowercase
		text = str(text).lower()
		
		# Step 3: Remove punctuation and numbers
		text = self._remove_punctuation_numbers(text)
		
		# Step 4: Tokenize (optional, NLTK word_tokenize for better tokenization)
		text = self._tokenize_text(text)
		
		# Step 5: Remove stopwords
		text = self._remove_stopwords(text)
		
		# Step 6: Lemmatize
		text = self._lemmatize_text(text)
		
		return text
	
	def clean_text_column(self, series: pd.Series, show_progress: bool = True) -> pd.Series:
		"""
		Clean an entire text column (vectorized operation).
		Use this for cleaning subject/body columns.
		
		Args:
			series: pandas Series containing text
			show_progress: Whether to show progress bar (requires tqdm)
		
		Returns:
			Cleaned pandas Series
		"""
		if show_progress:
			try:
				from tqdm import tqdm
				tqdm.pandas(desc="Cleaning text")
				return series.progress_apply(self.clean_text)
			except ImportError:
				pass
		
		return series.apply(self.clean_text)
