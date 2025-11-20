import pandas as pd
import numpy as np

class TemporalFeatureEngineer:
	"""
	Process timezone offsets into geographic regions and extract temporal features.

	Handles:
	- Converting timezone offset strings to geographic regions
	- Marking invalid/missing timezones as 'Unknown'
	- Extracting temporal features (year, month, hour, etc.)
	- Creating advanced temporal features for phishing detection
	"""

	def __init__(self):
		pass

	def _parse_timezone_offset(self, tz_str):
		"""Convert timezone string like '+0800' or '-0700' to hours."""
		if pd.isna(tz_str):
			return float('nan')
		try:
			sign = 1 if tz_str[0] == '+' else -1
			hours = int(tz_str[1:3])
			minutes = int(tz_str[3:5])
			return sign * (hours + minutes / 60.0)
		except (ValueError, IndexError):
			return float('nan')

	def _get_simple_region(self, offset_hours):
		"""
		Map timezone offset to region.
		Invalid timezones (outside UTC-12 to UTC+14) are marked as 'Unknown'.
		"""
		if pd.isna(offset_hours):
			return 'Unknown'
		# Check for invalid timezone range (outside UTC-12 to UTC+14)
		elif offset_hours < -12 or offset_hours > 14:
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

	def add_timezone_region(self, df: pd.DataFrame,
	                       tz_offset_col: str = 'timezone_offset',
	                       drop_offset: bool = True) -> pd.DataFrame:
		"""
		Convert timezone_offset column to timezone_region.

		Key behaviors:
		- Missing timezone_offset → timezone_region = 'Unknown'
		- Invalid timezone (outside UTC-12 to UTC+14) → timezone_region = 'Unknown'
		- Valid timezone → mapped to geographic region

		Args:
			df: DataFrame with timezone_offset column
			tz_offset_col: Name of timezone offset column (default: 'timezone_offset')
			drop_offset: Whether to drop timezone_offset after conversion (default: True)

		Returns:
			DataFrame with timezone_region column added
		"""
		df = df.copy()

		# Convert offset string to hours
		df['timezone_hours'] = df[tz_offset_col].apply(self._parse_timezone_offset)

		# Map to regions (missing/invalid → 'Unknown')
		df['timezone_region'] = df['timezone_hours'].apply(self._get_simple_region)

		# Clean up intermediate columns
		df = df.drop(columns=['timezone_hours'], errors='ignore')

		if drop_offset:
			df = df.drop(columns=[tz_offset_col], errors='ignore')

		return df

	def extract_temporal_features(self, df: pd.DataFrame,
	                              date_col: str = 'date') -> pd.DataFrame:
		"""
		Extract temporal features from date column.

		Features extracted:
		- year: Year (e.g., 2008)
		- month: Month number (1-12)
		- month_name: Month name (e.g., 'August')
		- day: Day of month (1-31)
		- day_of_week: Day of week (0=Monday, 6=Sunday)
		- day_name: Day name (e.g., 'Tuesday')
		- hour: Hour of day (0-23)
		- minute: Minute (0-59)
		- date_only: Date without time (date object)

		Args:
			df: DataFrame with datetime column
			date_col: Name of the date column

		Returns:
			DataFrame with temporal feature columns added
		"""
		df = df.copy()

		# Ensure date column is datetime type
		if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
			df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

		# Extract temporal features
		df['year'] = df[date_col].dt.year
		df['month'] = df[date_col].dt.month
		df['month_name'] = df[date_col].dt.month_name()
		df['day'] = df[date_col].dt.day
		df['day_of_week'] = df[date_col].dt.dayofweek
		df['day_name'] = df[date_col].dt.day_name()
		df['hour'] = df[date_col].dt.hour
		df['minute'] = df[date_col].dt.minute
		df['date_only'] = df[date_col].dt.date

		return df

	def process_all(self, df: pd.DataFrame,
	               date_col: str = 'date',
	               tz_offset_col: str = 'timezone_offset',
	               drop_offset: bool = True,
	               extract_features: bool = True) -> pd.DataFrame:
		"""
		Process timezone and extract temporal features in one call.

		Args:
			df: DataFrame with date and timezone_offset columns
			date_col: Name of the date column
			tz_offset_col: Name of timezone offset column
			drop_offset: Whether to drop timezone_offset after conversion
			extract_features: Whether to extract temporal features

		Returns:
			DataFrame with timezone_region and temporal features added
		"""
		df = df.copy()

		# Add timezone region
		df = self.add_timezone_region(df, tz_offset_col=tz_offset_col, drop_offset=drop_offset)

		# Extract temporal features if requested
		if extract_features:
			df = self.extract_temporal_features(df, date_col=date_col)

		return df

	def add_is_night(self, df: pd.DataFrame, 
	                 hour_col: str = 'hour') -> pd.DataFrame:
		"""
		Add is_night feature (binary flag for nighttime hours).
		
		Nighttime defined as: 22:00-06:00 (10 PM to 6 AM)
		
		Args:
			df: DataFrame with hour column
			hour_col: Name of hour column (default: 'hour')
			
		Returns:
			DataFrame with is_night column added
		"""
		df = df.copy()
		
		# Nighttime: 22:00-23:59 or 00:00-05:59
		df['is_night'] = ((df[hour_col] >= 22) | (df[hour_col] < 6)).astype(int)
		
		return df
	
	def add_sender_historical_features(self, df: pd.DataFrame,
	                                   sender_col: str = 'sender',
	                                   date_col: str = 'date',
	                                   label_col: str = 'label') -> pd.DataFrame:
		"""
		Add time-aware sender historical features to prevent data leakage.
		
		Features added:
		- sender_historical_phishing_rate: Historical phishing rate of sender (before current email)
		- sender_historical_count: Number of previous emails from sender
		
		**CRITICAL**: These features are calculated in a TIME-AWARE manner:
		- For each email at time T, only uses information from emails BEFORE time T
		- First email from sender gets global baseline rate
		- Features update as sender history grows
		- Ready for production deployment (no data leakage)
		
		Args:
			df: DataFrame with sender, date, and label columns
			sender_col: Name of sender column
			date_col: Name of date column
			label_col: Name of label column (0=legitimate, 1=phishing)
			
		Returns:
			DataFrame with sender_historical_phishing_rate and sender_historical_count columns
		"""
		df = df.copy()
		
		# CRITICAL: Sort by sender and date for time-aware calculation
		df = df.sort_values([sender_col, date_col]).reset_index(drop=True)

		# Count previous emails for this sender (0 for first, 1 for second, etc.)
		df['sender_historical_count'] = df.groupby(sender_col).cumcount()
		
		# Calculate cumulative phishing count
		df['_cumsum_phishing'] = df.groupby(sender_col)[label_col].cumsum()
		
		# Shift to get PREVIOUS phishing count (excludes current email)
		df['_prev_phishing'] = df.groupby(sender_col)['_cumsum_phishing'].shift(1).fillna(0)
		
		# Calculate time-aware rate: previous_phishing / previous_count
		df['sender_historical_phishing_rate'] = np.where(
			df['sender_historical_count'] > 0,
			df['_prev_phishing'] / df['sender_historical_count'],
			np.nan  # First email from sender has no history
		)
		
		# Handle first-time senders: fill NaN with global baseline
		global_baseline = df[label_col].mean()
		df['sender_historical_phishing_rate'].fillna(global_baseline, inplace=True)

		# Clean up temporary columns
		df.drop(['_cumsum_phishing', '_prev_phishing'], axis=1, inplace=True)

		return df
	
	def add_sender_temporal_features(self, df: pd.DataFrame,
	                                 sender_col: str = 'sender',
	                                 date_col: str = 'date') -> pd.DataFrame:
		"""
		Add time-aware sender temporal features in MINUTES.
		
		Features added:
		- current_time_gap: Minutes since sender's LAST email (time-aware)
		- sender_time_gap_std: Standard deviation of HISTORICAL time gaps in minutes (time-aware)
		
		**CRITICAL**: These features are calculated in a TIME-AWARE manner:
		- For each email at time T, only uses time gaps from emails BEFORE time T
		- First email from sender: current_time_gap = 0, sender_time_gap_std = 0
		- Features update as sender sends more emails
		- Ready for production deployment (no data leakage)
		
		**Note**: Features are in MINUTES for better granularity in burst detection
		
		Args:
			df: DataFrame with sender and date columns
			sender_col: Name of sender column
			date_col: Name of date column
			
		Returns:
			DataFrame with current_time_gap and sender_time_gap_std columns (both in minutes)
		"""
		df = df.copy()

		# CRITICAL: Sort by sender and date for time-aware calculation
		df = df.sort_values([sender_col, date_col]).reset_index(drop=True)

		# ============================================================
		# Feature 1: current_time_gap (in MINUTES)
		# ============================================================
		# Minutes since sender's LAST email (gap to previous email)
		df['current_time_gap'] = df.groupby(sender_col)[date_col].diff().dt.total_seconds() / 60
		df['current_time_gap'] = df['current_time_gap'].fillna(0)  # First emails have no previous

		# ============================================================
		# Feature 2: sender_time_gap_std (in MINUTES)
		# ============================================================
		# Standard deviation of HISTORICAL gaps (expanding window)
		# For each email, calculate std of PREVIOUS gaps only
		df['sender_time_gap_std'] = (
			df.groupby(sender_col)['current_time_gap']
			.expanding()
			.std()
			.shift(1)  # Exclude current gap from current row's statistic
			.reset_index(level=0, drop=True)
		)
		df['sender_time_gap_std'] = df['sender_time_gap_std'].fillna(0)

		return df
	
	def add_all_advanced_features(self, df: pd.DataFrame,
	                              sender_col: str = 'sender',
	                              date_col: str = 'date',
	                              label_col: str = 'label',
	                              hour_col: str = 'hour') -> pd.DataFrame:
		"""
		Add all advanced temporal features in one call.
		
		Features added:
		1. is_night: Binary flag for nighttime hours (22:00-06:00)
		2. sender_historical_phishing_rate: Time-aware historical phishing rate
		3. sender_historical_count: Number of previous emails from sender
		4. current_time_gap: Minutes since sender's last email (time-aware)
		5. sender_time_gap_std: Std dev of historical time gaps in minutes (time-aware)
		
		**IMPORTANT**: All sender-based features are TIME-AWARE to prevent data leakage.
		
		Args:
			df: DataFrame with required columns
			sender_col: Name of sender column
			date_col: Name of date column
			label_col: Name of label column
			hour_col: Name of hour column
			
		Returns:
			DataFrame with all advanced features added
		"""
		df = df.copy()

		# Add is_night feature
		df = self.add_is_night(df, hour_col=hour_col)

		# Add sender historical features (time-aware)
		df = self.add_sender_historical_features(df, sender_col=sender_col,
		                                        date_col=date_col, label_col=label_col)

		# Add sender temporal features (time-aware, in minutes)
		df = self.add_sender_temporal_features(df, sender_col=sender_col, date_col=date_col)

		return df
	
	def create_model_features(self, df: pd.DataFrame,
	                         sender_col: str = 'sender',
	                         date_col: str = 'date',
	                         label_col: str = 'label',
	                         tz_offset_col: str = 'timezone_offset') -> pd.DataFrame:
		"""
		Create only the 7 model-ready features and drop timezone_offset.
		
		This method creates a clean dataset with only the features needed for modeling:
		
		**Baseline Temporal (2 features):**
		- hour: Hour of day (0-23)
		- is_night: Binary flag for nighttime hours (22:00-06:00)
		
		**Regional Features (1 feature):**
		- timezone_region: Geographic region (Americas, Europe/Africa, Middle East/South Asia, 
		                   APAC, Oceania/Pacific, Unknown)
		
		**Sender Behavioral Patterns (4 features):**
		- sender_historical_phishing_rate: Time-aware historical phishing rate
		- sender_historical_count: Number of previous emails from sender
		- current_time_gap: Minutes since sender's last email (time-aware)
		- sender_time_gap_std: Std dev of historical time gaps in minutes (time-aware)
		
		**IMPORTANT**: 
		- All sender features are TIME-AWARE to prevent data leakage
		- timezone_offset is dropped after creating timezone_region
		- Returns only model-ready features (drops descriptive columns)
		
		Args:
			df: DataFrame with required columns (date, timezone_offset, sender, label, hour)
			sender_col: Name of sender column
			date_col: Name of date column
			label_col: Name of label column
			tz_offset_col: Name of timezone offset column
			
		Returns:
			DataFrame with only the 7 model features + original columns except timezone_offset
		"""
		df = df.copy()
		
		# Step 1: Add timezone region and drop timezone_offset
		df = self.add_timezone_region(df, tz_offset_col=tz_offset_col, drop_offset=True)
		
		# Step 2: Add all advanced temporal features
		df = self.add_all_advanced_features(df, sender_col=sender_col, 
		                                   date_col=date_col,
		                                   label_col=label_col, 
		                                   hour_col='hour')
		
		return df
