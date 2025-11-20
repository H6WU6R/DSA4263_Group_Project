import pandas as pd
import re
from email.utils import parsedate_to_datetime

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
	

	def _parse_email_date_flexible(self, date_str):
		"""
		Flexible RFC-5322 email date parser with fallback for dates without day names.
		
		This is more robust than strict format strings and matches the behavior
		of the original notebook's parse_email_date_preserve_tz function.
		
		Handles:
		  - "Tue, 05 Aug 2008 16:31:02 -0700" (standard with day name)
		  - "05 Aug 2008 16:31:02 -0700" (missing day name)
		  - "Tue, 05 Aug 2008 16:31:02" (no timezone)
		  - Various other RFC-5322 compliant formats
		"""
		try:
			# Try standard RFC-5322 parsing
			dt = parsedate_to_datetime(date_str)
			# Convert to timezone-naive UTC
			return dt.replace(tzinfo=None) if dt.tzinfo is None else dt.astimezone(None).replace(tzinfo=None)
		except:
			pass
		
		try:
			# Fallback: Remove day name prefix and retry
			# Handles dates like "05 Aug 2008 16:31:02 -0700"
			cleaned = re.sub(r'^\w{3},\s*', '', str(date_str))
			dt = parsedate_to_datetime(cleaned)
			return dt.replace(tzinfo=None) if dt.tzinfo is None else dt.astimezone(None).replace(tzinfo=None)
		except:
			return pd.NaT

	def clean_dates(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
		"""
		Parse and clean date column, extract timezone offset.

		Key behaviors:
		- Unparseable/missing dates → dropped from dataset
		- Missing timezones → timezone_offset will be NaN 


		Args:
			df: Input DataFrame
			date_col: Name of the date column to clean

		Returns:
			DataFrame with:
			- Cleaned date column (datetime type)
			- timezone_offset column (string like '+0800', '-0700', or NaN if missing)
		"""
		df = df.copy()

		# Extract timezone offset BEFORE parsing (keep as string)
		df['timezone_offset'] = df[date_col].astype(str).str.extract(r'([+-]\d{4})$')[0]

		# Parse dates using flexible RFC-5322 parser
		df[date_col] = df[date_col].apply(self._parse_email_date_flexible)

		# Ensure date column is datetime type
		df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

		# Drop rows with missing/unparseable dates
		initial_count = len(df)
		df = df.dropna(subset=[date_col])
		dropped_count = initial_count - len(df)
		if dropped_count > 0:
			print(f"Dropped {dropped_count:,} rows with missing/unparseable dates ({dropped_count/initial_count*100:.2f}%)")

		# Keep timezone_offset column (do NOT drop it)
		# Geographic processing should be done separately using GeographicProcessor
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