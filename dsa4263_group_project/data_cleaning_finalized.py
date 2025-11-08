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
		"""Convert timezone string like '+0800' or '-0700' to hours"""
		if pd.isna(tz_str):
			return 0.0  # Treat missing timezone as UTC (+0000)
		try:
			sign = 1 if tz_str[0] == '+' else -1
			hours = int(tz_str[1:3])
			minutes = int(tz_str[3:5])
			return sign * (hours + minutes / 60.0)
		except (ValueError, IndexError):
			return 0.0  # Default to UTC on error

	def _get_simple_region(self, offset_hours):
		"""
		Simplified 6-region model for cleaner visualization
		Valid range: UTC-12:00 to UTC+14:00

		Boundary handling:
		- UTC-4 belongs to Americas (e.g., Atlantic Time)
		- UTC+2 belongs to Europe/Africa (e.g., South Africa)
		- UTC+6 belongs to Middle East/South Asia (e.g., Bangladesh)
		- UTC+10 belongs to APAC (e.g., Australian Eastern Time)
		"""
		if pd.isna(offset_hours):
			return 'Unknown'
		elif -12 <= offset_hours < -4:
			return 'Americas'
		elif -4 <= offset_hours <= 2:
			return 'Europe/Africa'
		elif 2 < offset_hours <= 6:
			return 'Middle East/South Asia'  # Includes all .5 offsets (Iran, India, etc.)
		elif 6 < offset_hours <= 10:
			return 'APAC'  # Includes Myanmar (6.5), Australia Central (9.5)
		elif 10 < offset_hours <= 14:
			return 'Oceania/Pacific'  # Includes Fiji, New Zealand, etc.
		else:
			return 'Unknown'  # Out of valid range

	def clean_dates(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
		"""
		Clean and standardize date columns, extract timezone, remove anomalies.
		
		This method:
		1. Extracts timezone information from date strings
		2. Parses dates and converts to pandas datetime
		3. Maps timezone offsets to geographic regions
		4. Removes date anomalies (dates outside 1990-2025 range)
		5. Creates timezone_region column for geographic analysis
		
		Args:
			df: DataFrame with email data
			date_col: Name of the date column (default: 'date')
			
		Returns:
			DataFrame with cleaned dates and timezone_region column
		"""
		df = df.copy()
		
		# ===== STEP 1: Extract Timezone Information =====
		# Extract timezone offset from date string (e.g., '+0800', '-0700')
		df['timezone_offset'] = df[date_col].astype(str).str.extract(r'([+-]\d{4})$')[0]
		
		# Convert timezone offset to hours for easier handling
		def parse_timezone_offset(tz_str):
			"""Convert timezone string like '+0800' or '-0700' to hours"""
			if pd.isna(tz_str):
				# Treat missing timezone as UTC (+0000)
				return 0.0
			try:
				sign = 1 if tz_str[0] == '+' else -1
				hours = int(tz_str[1:3])
				minutes = int(tz_str[3:5])
				return sign * (hours + minutes / 60.0)
			except (ValueError, IndexError):
				return 0.0  # Default to UTC on error
		
		df['timezone_hours'] = df['timezone_offset'].apply(parse_timezone_offset)
		
		# ===== STEP 2: Parse Dates =====
		# Parse the date column to datetime
		# First try standard format, then use infer_datetime_format for flexibility
		try:
			df[date_col] = pd.to_datetime(df[date_col], format='%a, %d %b %Y %H:%M:%S %z', errors='coerce', utc=True)
		except:
			df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
		
		# Convert to timezone-naive (remove timezone info while keeping the datetime values)
		# This allows for easier comparison and is standard practice for analysis
		df[date_col] = df[date_col].dt.tz_localize(None)
		
		# ===== STEP 3: Map Timezone Hours to Geographic Regions =====
		def get_simple_region(offset_hours):
			"""
			Simplified 6-region model for cleaner visualization
			Valid range: UTC-12:00 to UTC+14:00
			
			Boundary handling:
			- UTC-4 belongs to Americas (e.g., Atlantic Time)
			- UTC+2 belongs to Europe/Africa (e.g., South Africa)
			- UTC+6 belongs to Middle East/South Asia (e.g., Bangladesh)
			- UTC+10 belongs to APAC (e.g., Australian Eastern Time)
			"""
			if pd.isna(offset_hours):
				return 'Unknown'
			elif -12 <= offset_hours < -4:
				return 'Americas'
			elif -4 <= offset_hours <= 2:
				return 'Europe/Africa'
			elif 2 < offset_hours <= 6:
				return 'Middle East/South Asia'  # Includes all .5 offsets (Iran, India, etc.)
			elif 6 < offset_hours <= 10:
				return 'APAC'  # Includes Myanmar (6.5), Australia Central (9.5)
			elif 10 < offset_hours <= 14:
				return 'Oceania/Pacific'  # Includes Fiji, New Zealand, etc.
			else:
				return 'Unknown'  # Out of valid range
		
		df['timezone_region'] = df['timezone_hours'].apply(get_simple_region)
		
		# ===== STEP 4: Validate and Clean Date Anomalies =====
		# Define reasonable date range
		min_valid_date = pd.Timestamp('1990-01-01')
		max_valid_date = pd.Timestamp('2025-12-31')
		
		# Filter out dates outside valid range
		df = df[
			(df[date_col] >= min_valid_date) & 
			(df[date_col] <= max_valid_date)
		].copy()
		
		# ===== STEP 5: Clean up temporary columns =====
		# Remove intermediate columns, keeping only timezone_region
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
