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

	def clean_text(self, text: str) -> str:
		"""
		Clean email text: lowercase, remove punctuation, stopwords, lemmatize.
		"""
		if pd.isna(text) or text == "":
			return ""
		text = str(text).lower()
		text = re.sub(r'[^a-zA-Z\s]', ' ', text)
		text = re.sub(r'\s+', ' ', text).strip()
		tokens = text.split()
		if self.stop_words:
			tokens = [w for w in tokens if w not in self.stop_words and len(w) > 2]
		if self.lemmatizer:
			try:
				tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
			except Exception:
				pass
		return ' '.join(tokens)
