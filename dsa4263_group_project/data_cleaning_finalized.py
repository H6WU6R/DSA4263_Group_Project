import pandas as pd
import re
from datetime import timezone
from email.utils import parsedate_to_datetime

class DataCleaner:
	"""
	Email data cleaning for date parsing and text preprocessing.
	
	Features:
	- RFC-5322 compliant date parsing with timezone extraction
	- Text cleaning: lowercase, punctuation removal, stopwords, lemmatization
	"""
	
	def __init__(self, stop_words=None, lemmatizer=None, stemmer=None):
		"""
		Initialize DataCleaner.
		
		Args:
			stop_words: Set/list of stopwords to remove (optional)
			lemmatizer: Lemmatizer object with .lemmatize() method (optional)
			stemmer: Stemmer object with .stem() method (optional)
		"""
		self.stop_words = stop_words
		self.lemmatizer = lemmatizer
		self.stemmer = stemmer

	@staticmethod
	def load_raw_data(filename: str) -> pd.DataFrame:
		"""Load raw data from RAW_DATA_DIR."""
		from dsa4263_group_project.config import RAW_DATA_DIR
		return pd.read_csv(RAW_DATA_DIR / filename)

	@staticmethod
	def load_processed_data(filename: str) -> pd.DataFrame:
		"""Load processed data from PROCESSED_DATA_DIR."""
		from dsa4263_group_project.config import PROCESSED_DATA_DIR
		return pd.read_csv(PROCESSED_DATA_DIR / filename)

	@staticmethod
	def save_processed_data(df: pd.DataFrame, filename: str):
		"""Save processed data to PROCESSED_DATA_DIR."""
		from dsa4263_group_project.config import PROCESSED_DATA_DIR
		df.to_csv(PROCESSED_DATA_DIR / filename, index=False)

	def clean_and_merge(self, dfs, sender_col='sender', receiver_col='receiver'):
		"""
		Clean and merge a list of DataFrames:
		- Fill missing receiver values with unique labels
		- Drop rows where sender is missing
		- Fill missing subject/body with empty strings
		- Combine subject and body into 'full_text'
		- Merge all DataFrames into one

		Args:
			dfs: List of DataFrames to clean and merge
			sender_col: Sender column name
			receiver_col: Receiver column name

		Returns:
			Merged and cleaned DataFrame
		"""
		cleaned_dfs = []

		for df in dfs:
			df = df.copy()
			# Fill missing receivers
			na_indices = df[df[receiver_col].isnull()].index
			for i, idx in enumerate(na_indices, 1):
				df.at[idx, receiver_col] = f"na{i}"
			# Drop missing senders
			df = df.dropna(subset=[sender_col])

			# Fill missing subject and body
			if 'subject' in df.columns:
				df['subject'] = df['subject'].fillna('')
			if 'body' in df.columns:
				df['body'] = df['body'].fillna('')
				
			# Combine subject and body
			if 'subject' in df.columns and 'body' in df.columns:
				df['full_text'] = df['subject'] + " " + df['body']
			elif 'body' in df.columns:
				df['full_text'] = df['body']
			elif 'subject' in df.columns:
				df['full_text'] = df['subject']

			cleaned_dfs.append(df)
			
		# Merge all cleaned DataFrames
		merged_df = pd.concat(cleaned_dfs, ignore_index=True)
		return merged_df

	def _parse_email_date_flexible(self, date_str):
		"""
		Parse RFC-5322 email date and preserve LOCAL datetime.
		
		Returns:
			Tuple of (local_datetime, timezone_offset_str)
			- local_datetime: Sender's local time (timezone-naive)
			- timezone_offset_str: Original offset like '+0800', '-0700', or None
		"""
		try:
			# Parse with timezone awareness
			dt = parsedate_to_datetime(date_str)
			
			# Extract timezone offset BEFORE stripping tzinfo
			if dt.tzinfo is not None:
				offset = dt.utcoffset()
				total_seconds = int(offset.total_seconds())
				hours, remainder = divmod(abs(total_seconds), 3600)
				minutes = remainder // 60
				sign = '+' if total_seconds >= 0 else '-'
				tz_offset_str = f"{sign}{hours:02d}{minutes:02d}"
				
				# Keep LOCAL time, just remove tzinfo
				local_dt = dt.replace(tzinfo=None)
				return local_dt, tz_offset_str
			else:
				# No timezone info - keep as-is
				return dt, None
				
		except (ValueError, TypeError, AttributeError, OverflowError):
			pass
		
		try:
			# Fallback: Remove day name prefix and retry
			cleaned = re.sub(r'^\w{3},\s*', '', str(date_str))
			dt = parsedate_to_datetime(cleaned)
			
			if dt.tzinfo is not None:
				offset = dt.utcoffset()
				total_seconds = int(offset.total_seconds())
				hours, remainder = divmod(abs(total_seconds), 3600)
				minutes = remainder // 60
				sign = '+' if total_seconds >= 0 else '-'
				tz_offset_str = f"{sign}{hours:02d}{minutes:02d}"
				
				# Keep LOCAL time, just remove tzinfo
				local_dt = dt.replace(tzinfo=None)
				return local_dt, tz_offset_str
			else:
				return dt, None
				
		except (ValueError, TypeError, AttributeError, OverflowError):
			return pd.NaT, None

	def clean_dates(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
		"""
		Parse dates and preserve local time + timezone offset.
		
		Args:
			df: Input DataFrame
			date_col: Date column name (default: 'date')
		
		Returns:
			DataFrame with:
			- date: Sender's local datetime (timezone-naive, replaces original)
			- timezone_offset: Offset string ('+0800', '-0700', or NaN)
			- Unparseable dates removed
		"""
		initial_count = len(df)
		df = df.copy()

		# Parse dates and extract timezone offsets
		parsed_results = df[date_col].apply(self._parse_email_date_flexible)
		df[date_col] = parsed_results.apply(lambda x: x[0])  # Overwrite original with local time
		df['timezone_offset'] = parsed_results.apply(lambda x: x[1])

		# Convert date column to pandas datetime64 format (from Python datetime objects)
		# Use errors='coerce' to convert out-of-bounds dates to NaT
		df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

		# Count timezone distribution
		tz_present = df['timezone_offset'].notna().sum()
		tz_missing = df['timezone_offset'].isna().sum()
		print(f"Timezone present         : {tz_present:,} ({tz_present/initial_count*100:.2f}%)")
		print(f"Timezone missing         : {tz_missing:,} ({tz_missing/initial_count*100:.2f}%)")

		# Drop unparseable dates (including out-of-bounds dates converted to NaT)
		df = df.dropna(subset=[date_col])
		final_count = len(df)
		dropped = initial_count - final_count

		if dropped > 0:
			print(f"Dropped                  : {dropped:,} unparseable dates ({dropped/initial_count*100:.2f}%)")

		return df

	def _remove_punctuation_numbers(self, text):
		"""Remove punctuation and numbers, keep only letters and spaces."""
		text = re.sub(r'[^a-zA-Z\s]', ' ', str(text))
		text = re.sub(r'\s+', ' ', text).strip()
		return text

	def _tokenize_text(self, text):
		"""Tokenize text using NLTK word_tokenize with fallback."""
		try:
			from nltk.tokenize import word_tokenize
			tokens = word_tokenize(str(text))
			return ' '.join(tokens)
		except ImportError:
			return text

	def _remove_stopwords(self, text):
		"""Remove stopwords (words in stop_words set with length <= 2)."""
		if not self.stop_words:
			return text
		tokens = text.split()
		filtered = [w for w in tokens if w.lower() not in self.stop_words and len(w) > 2]
		return ' '.join(filtered)

	def _lemmatize_text(self, text):
		"""Lemmatize tokens using configured lemmatizer."""
		if not self.lemmatizer:
			return text
		try:
			tokens = text.split()
			lemmatized = [self.lemmatizer.lemmatize(w) for w in tokens]
			return ' '.join(lemmatized)
		except (AttributeError, TypeError):
			return text

	def _stem_text(self, text):
		"""Stem tokens using configured stemmer."""
		if not self.stemmer:
			return text
		try:
			tokens = text.split()
			stemmed = [self.stemmer.stem(w) for w in tokens]
			return ' '.join(stemmed)
		except (AttributeError, TypeError):
			return text

	def clean_text(self, text: str) -> str:
		"""
		Clean email text: lowercase → remove punctuation → stopwords → lemmatize → stem.
		
		Args:
			text: Raw text string
		
		Returns:
			Cleaned text string (empty if input invalid)
		"""
		# Handle empty/invalid
		if pd.isna(text) or text == "" or not str(text).strip():
			return ""
		
		# Lowercase
		text = str(text).lower()
		
		# Remove punctuation and numbers
		text = self._remove_punctuation_numbers(text)
		
		# Check if empty after cleaning
		if not text:
			return ""
		
		# Tokenize
		text = self._tokenize_text(text)
		
		# Remove stopwords
		text = self._remove_stopwords(text)
		
		# Lemmatize
		text = self._lemmatize_text(text)
		
		# Stem
		text = self._stem_text(text)
		
		return text

	def clean_text_column(self, series: pd.Series, show_progress: bool = True) -> pd.Series:
		"""
		Clean entire text column with optional progress bar.
		
		Args:
			series: pandas Series with text
			show_progress: Show progress bar if tqdm available
		
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

	def complete_cleaning_pipeline(self, df: pd.DataFrame, 
	                                sender_col: str = 'sender',
	                                receiver_col: str = 'receiver',
	                                date_col: str = 'date',
	                                text_cols: list = None) -> pd.DataFrame:
		"""
		Complete data cleaning pipeline with comprehensive reporting.
		
		Args:
			df: Input DataFrame
			sender_col: Sender column name
			receiver_col: Receiver column name  
			date_col: Date column name
			text_cols: List of text columns to clean (optional)
		
		Returns:
			Cleaned DataFrame with summary report
		"""
		initial_count = len(df)
		
		print("\n" + "=" * 64)
		print("DATA CLEANING PIPELINE")
		print("=" * 64)
		print(f"Input rows               : {initial_count:,}")
		print()
		
		# Step 1: Handle missing values
		print("[Step 1/3] Missing Value Handling")
		# Ensure input is a list for clean_and_merge
		input_data = [df] if isinstance(df, pd.DataFrame) else df
		df = self.clean_and_merge(input_data, sender_col=sender_col, receiver_col=receiver_col)
		after_missing = len(df)
		print()
		
		# Step 2: Clean dates
		print("[Step 2/3] Date Parsing")
		df = self.clean_dates(df, date_col=date_col)
		after_dates = len(df)
		print()
		
		# Step 3: Clean text (optional)
		if text_cols:
			print("[Step 3/3] Text Cleaning")
			for col in text_cols:
				if col in df.columns:
					before_text = df[col].notna().sum()
					df[col] = self.clean_text_column(df[col], show_progress=False)
					# Count empty strings after cleaning
					empty_after = (df[col] == "").sum()
					if empty_after > 0:
						print(f"  {col:<20} : {empty_after:,} became empty after cleaning")
			print()
		else:
			print("[Step 3/3] Text Cleaning - SKIPPED")
			print()
		
		# Final summary
		final_count = len(df)
		total_dropped = initial_count - final_count
		
		print("=" * 64)
		print("DATA CLEANING SUMMARY")
		print("=" * 64)
		print(f"Input rows               : {initial_count:,}")
		print(f"Dropped                  : {total_dropped:,} rows ({total_dropped/initial_count*100:.2f}%)")
		print(f"Output rows              : {final_count:,} ({final_count/initial_count*100:.2f}% retained)")
		print("=" * 64 + "\n")
		
		return df
