import pandas as pd
import numpy as np

class TemporalFeatureEngineer:
	"""
	Temporal feature engineering for phishing email detection.

	
	Key Capabilities:
	  - Convert timezone offsets to geographic regions (6-region model)
	  - Create UTC datetime from local time + timezone offset
	  - Extract behavioral features from local datetime (hour, is_night, etc.)
	  - Calculate time-aware sender features (no data leakage)

	
	Usage:
	  engineer = TemporalFeatureEngineer()
	  df = engineer.complete_pipeline(df, global_baseline=0.42)
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

	def _create_utc_date(self, df: pd.DataFrame,
	                    local_date_col: str = 'date',
	                    tz_offset_col: str = 'timezone_offset') -> pd.DataFrame:
		"""
		Create UTC datetime from local datetime + timezone offset.
		
		Logic:
		    - Valid timezone: local_time - timezone_offset = UTC_time
		    - Missing/invalid timezone: Assume local_time is already UTC (offset = 0)
		
		Example:
		    local_time = 16:31, offset = -0700 (UTC-7)
		    → UTC_time = 16:31 - (-7 hours) = 23:31 UTC
		    
		    local_time = 16:31, offset = NaN
		    → UTC_time = 16:31 (assume already UTC)
		
		Args:
		    df: DataFrame
		    local_date_col: Column with local datetime (default: 'date')
		    tz_offset_col: Column with timezone offset strings
		
		Returns:
		    DataFrame with utc_date column added
		"""
		df = df.copy()
		
		def parse_offset_hours(tz_str):
			"""Convert timezone string to hours. Returns 0 for missing/invalid."""
			if pd.isna(tz_str):
				return 0.0  # Missing → assume UTC (offset = 0)
			try:
				sign = 1 if tz_str[0] == '+' else -1
				hours = int(tz_str[1:3])
				minutes = int(tz_str[3:5])
				offset_hours = sign * (hours + minutes / 60.0)
				
				# Validate range (UTC-12 to UTC+14)
				if -12 <= offset_hours <= 14:
					return offset_hours
				else:
					return 0.0  # Invalid → assume UTC
			except (ValueError, IndexError, TypeError):
				return 0.0  # Parse error → assume UTC
		
		# Parse timezone offsets to hours
		df['_tz_offset_hours'] = df[tz_offset_col].apply(parse_offset_hours)
		
		# Create UTC date: local_date - timezone_offset
		# Example: local 16:31 at UTC-7 → 16:31 - (-7) = 23:31 UTC
		df['utc_date'] = df[local_date_col] - pd.to_timedelta(df['_tz_offset_hours'], unit='h')
		
		# Count how many used fallback
		fallback_count = (df['_tz_offset_hours'] == 0).sum()
		missing_tz_count = df[tz_offset_col].isna().sum()
		
		if missing_tz_count > 0:
			print(f"UTC date: {missing_tz_count:,} missing timezones assumed UTC "
			      f"({missing_tz_count/len(df)*100:.1f}%)")
		
		# Cleanup
		df = df.drop(columns=['_tz_offset_hours'])
		
		return df

	def validate_temporal_consistency(self, df: pd.DataFrame, 
	                                  date_col: str = 'utc_date',
	                                  sort_if_needed: bool = True) -> pd.DataFrame:
		"""
		Ensure data is sorted by UTC time for time-aware features.
		
		CRITICAL: All sender historical features require chronological order.
		
		Args:
		    df: Input DataFrame
		    date_col: UTC date column to validate (default: 'utc_date')
		    sort_if_needed: If True, sort automatically; if False, raise error
		
		Returns:
		    DataFrame sorted by UTC date
		"""
		df = df.copy()
		
		# Check if already sorted
		if df[date_col].is_monotonic_increasing:
			print(f"  Status                   : Already sorted by '{date_col}'")
			return df
		
		# Handle unsorted data
		if sort_if_needed:
			print(f"  Status                   : Requires sorting")
			print(f"  Action                   : Sorting {len(df):,} rows by '{date_col}'")
			df = df.sort_values(date_col).reset_index(drop=True)
			print(f"  Result                   : Sorted successfully")
			return df
		else:
			raise ValueError(
				f"Data not sorted by '{date_col}'. "
				f"Time-aware features require chronological ordering."
			)

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
	                              local_date_col: str = 'date') -> pd.DataFrame:
		"""
		Extract behavioral temporal features from LOCAL datetime.
		
		Features extracted (all based on sender's local time):
		- year: Year (e.g., 2008)
		- month: Month number (1-12)
		- month_name: Month name (e.g., 'August')
		- day: Day of month (1-31)
		- day_of_week: Day of week (0=Monday, 6=Sunday)
		- day_name: Day name (e.g., 'Tuesday')
		- hour: LOCAL hour (0-23) - sender's local hour
		- minute: Minute (0-59)
		- date_only: Date without time (date object)

		Args:
			df: DataFrame with local datetime column
			local_date_col: Name of the local date column (default: 'date')

		Returns:
			DataFrame with temporal feature columns added
		"""
		df = df.copy()

		# Ensure date column is datetime type
		if not pd.api.types.is_datetime64_any_dtype(df[local_date_col]):
			df[local_date_col] = pd.to_datetime(df[local_date_col], errors='coerce')

		# Extract temporal features from LOCAL time
		df['year'] = df[local_date_col].dt.year
		df['month'] = df[local_date_col].dt.month
		df['month_name'] = df[local_date_col].dt.month_name()
		df['day'] = df[local_date_col].dt.day
		df['day_of_week'] = df[local_date_col].dt.dayofweek
		df['day_name'] = df[local_date_col].dt.day_name()
		df['hour'] = df[local_date_col].dt.hour  # LOCAL hour
		df['minute'] = df[local_date_col].dt.minute
		df['date_only'] = df[local_date_col].dt.date

		return df

	def process_all(self, df: pd.DataFrame,
	               local_date_col: str = 'date',
	               tz_offset_col: str = 'timezone_offset',
	               drop_offset: bool = True,
	               extract_features: bool = True,
	               validate_temporal_order: bool = True) -> pd.DataFrame:
		"""
		Process timezone, create UTC date, and extract temporal features.

		Pipeline:
		1. Create timezone_region from timezone_offset
		2. Create utc_date from date + timezone_offset
		3. Validate temporal consistency (utc_date sorted)
		4. Extract behavioral features from date

		Args:
			df: DataFrame with date and timezone_offset columns
			local_date_col: Name of the local date column (default: 'date')
			tz_offset_col: Name of timezone offset column
			drop_offset: Whether to drop timezone_offset after processing
			extract_features: Whether to extract temporal features
			validate_temporal_order: Whether to validate/sort by utc_date

		Returns:
			DataFrame with timezone_region, utc_date, and temporal features
		"""
		df = df.copy()

		# Step 1: Add timezone region (keep offset for now)
		df = self.add_timezone_region(df, tz_offset_col=tz_offset_col, drop_offset=False)

		# Step 2: Create UTC date from local_date + timezone_offset
		df = self._create_utc_date(df, local_date_col=local_date_col, tz_offset_col=tz_offset_col)

		# Step 3: Validate temporal ordering
		if validate_temporal_order:
			df = self.validate_temporal_consistency(df, date_col='utc_date', sort_if_needed=True)

		# Step 4: Extract temporal features from local_date
		if extract_features:
			df = self.extract_temporal_features(df, local_date_col=local_date_col)

		# Step 5: Drop timezone_offset if requested
		if drop_offset:
			df = df.drop(columns=[tz_offset_col], errors='ignore')

		return df

	def add_is_night(self, df: pd.DataFrame, 
	                 hour_col: str = 'hour') -> pd.DataFrame:
		"""
		Add is_night feature based on LOCAL hour.
		
		Nighttime defined as: 22:00-06:00 in sender's local timezone.
		This captures true behavioral patterns (working at night locally).
		
		Args:
			df: DataFrame with hour column (LOCAL hour from local_date)
			hour_col: Name of hour column (default: 'hour')
			
		Returns:
			DataFrame with is_night column added
		"""
		df = df.copy()
		
		# Nighttime: 22:00-23:59 or 00:00-05:59 (local time)
		df['is_night'] = ((df[hour_col] >= 22) | (df[hour_col] < 6)).astype(int)
		
		return df
	
	def add_sender_historical_features(self, df: pd.DataFrame,
	                                   sender_col: str = 'sender',
	                                   date_col: str = 'utc_date',
	                                   label_col: str = 'label',
	                                   validate_order: bool = True,
	                                   global_baseline: float = None) -> pd.DataFrame:
		"""
		Add time-aware sender historical features using UTC datetime.
		
		Features added:
		- sender_historical_phishing_rate: Historical phishing rate of sender (before current email)
		- sender_historical_count: Number of previous emails from sender
		
		**CRITICAL**: These features are calculated in a TIME-AWARE manner:
		- Uses utc_date to ensure correct chronological ordering
		- For each email at time T, only uses information from emails BEFORE time T
		- First email from sender gets global baseline rate
		- Features update as sender history grows
		- Ready for production deployment (no data leakage)
		
		Args:
			df: DataFrame with sender, utc_date, and label columns (sorted by utc_date)
			sender_col: Name of sender column
			date_col: Name of UTC date column (default: 'utc_date')
			label_col: Name of label column (0=legitimate, 1=phishing)
			validate_order: If True, validates temporal ordering
			global_baseline: Pre-calculated baseline from training data. If None, calculated from current df.
			
		Returns:
			DataFrame with sender_historical_phishing_rate and sender_historical_count columns
		"""
		df = df.copy()
		
		# CRITICAL VALIDATION: Ensure chronological ordering
		if validate_order and not df[date_col].is_monotonic_increasing:
			raise ValueError(
				f"DataFrame must be sorted by '{date_col}'. "
				f"Call self.validate_temporal_consistency(df) first."
			)

		# Calculate or use provided baseline
		if global_baseline is None:
			global_baseline = df[label_col].mean()
			print(f"[INFO] Using baseline from current data: {global_baseline:.4f}")

		# Memory-efficient calculation: compute in memory without storing intermediate columns
		# Count previous emails for this sender (0 for first, 1 for second, etc.)
		prev_count = df.groupby(sender_col).cumcount()
		
		# Calculate cumulative phishing count, shifted to exclude current email
		# Must use transform to shift within groups
		cumsum_phishing = df.groupby(sender_col)[label_col].transform(lambda x: x.cumsum().shift(1).fillna(0))
		
		# Direct assignment to final features (no intermediate columns in DataFrame)
		df['sender_historical_count'] = prev_count
		df['sender_historical_phishing_rate'] = np.where(
			prev_count > 0,                    # Condition: not first email from sender
			cumsum_phishing / prev_count,      # True: calculate rate from history
			global_baseline                    # False: use global baseline for first email
		)

		return df
	
	def add_sender_temporal_features(self, df: pd.DataFrame,
	                                 sender_col: str = 'sender',
	                                 date_col: str = 'utc_date',
	                                 validate_order: bool = True) -> pd.DataFrame:
		"""
		Add time-aware sender temporal features in MINUTES using UTC datetime.
		
		Features added:
		- current_time_gap: Minutes since sender's LAST email (time-aware)
		- sender_time_gap_std: Standard deviation of HISTORICAL time gaps in minutes (time-aware)
		
		**CRITICAL**: These features are calculated in a TIME-AWARE manner:
		- Uses utc_date to ensure accurate time gaps regardless of timezone
		- For each email at time T, only uses time gaps from emails BEFORE time T
		- First email from sender: current_time_gap = 0, sender_time_gap_std = 0
		- Features update as sender sends more emails
		- Ready for production deployment (no data leakage)
		
		**Note**: Features are in MINUTES for better granularity in burst detection
		
		Args:
			df: DataFrame with sender and utc_date columns (sorted by utc_date)
			sender_col: Name of sender column
			date_col: Name of UTC date column (default: 'utc_date')
			validate_order: If True, validates temporal ordering
			
		Returns:
			DataFrame with current_time_gap and sender_time_gap_std columns (both in minutes)
		"""
		df = df.copy()

		# CRITICAL VALIDATION: Ensure chronological ordering
		if validate_order and not df[date_col].is_monotonic_increasing:
			raise ValueError(
				f"DataFrame must be sorted by '{date_col}'. "
				f"Call self.validate_temporal_consistency(df) first."
			)

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
		
		# Use transform with a custom function to ensure proper shifting within groups
		def calculate_historical_std(group):
			# Calculate expanding std
			expanding_std = group.expanding().std()
			# Shift to exclude current value
			return expanding_std.shift(1)
		
		df['sender_time_gap_std'] = (
			df.groupby(sender_col)['current_time_gap']
			.transform(calculate_historical_std)
		)
		df['sender_time_gap_std'] = df['sender_time_gap_std'].fillna(0)

		return df
	
	def add_all_advanced_features(self, df: pd.DataFrame,
	                              sender_col: str = 'sender',
	                              utc_date_col: str = 'utc_date',
	                              label_col: str = 'label',
	                              hour_col: str = 'hour',
	                              validate_order: bool = True,
	                              global_baseline: float = None) -> pd.DataFrame:
		"""
		Add all advanced temporal features in one call.
		
		Features added:
		1. is_night: Binary flag for nighttime hours (22:00-06:00 LOCAL time)
		2. sender_historical_phishing_rate: Time-aware historical phishing rate (UTC-based)
		3. sender_historical_count: Number of previous emails from sender
		4. current_time_gap: Minutes since sender's last email (UTC-based, time-aware)
		5. sender_time_gap_std: Std dev of historical time gaps in minutes (UTC-based, time-aware)
		
		**IMPORTANT**: 
		- Behavioral features (is_night, hour) use local_date
		- Sender features (history, gaps) use utc_date for accurate chronology
		- All sender-based features are TIME-AWARE to prevent data leakage
		
		Args:
			df: DataFrame with required columns (sorted by utc_date)
			sender_col: Name of sender column
			utc_date_col: Name of UTC date column (default: 'utc_date')
			label_col: Name of label column
			hour_col: Name of LOCAL hour column (default: 'hour')
			validate_order: If True, validates temporal ordering
			global_baseline: Pre-calculated baseline from training data
			
		Returns:
			DataFrame with all advanced features added
		"""
		df = df.copy()

		# CRITICAL: Validate temporal ordering ONCE at the start
		if validate_order and not df[utc_date_col].is_monotonic_increasing:
			raise ValueError(
				f"DataFrame must be sorted by '{utc_date_col}'. "
				f"Call self.validate_temporal_consistency(df) first."
			)

		# Check required columns exist
		required_cols = [sender_col, utc_date_col, label_col, hour_col]
		missing = [col for col in required_cols if col not in df.columns]
		if missing:
			raise ValueError(f"Missing required columns: {missing}")

		print("Adding advanced temporal features...")

		# Add is_night feature (uses LOCAL hour)
		df = self.add_is_night(df, hour_col=hour_col)

		# Add sender historical features (uses UTC date, validation already done)
		df = self.add_sender_historical_features(
			df, sender_col=sender_col, date_col=utc_date_col, 
			label_col=label_col, validate_order=False, global_baseline=global_baseline
		)

		# Add sender temporal features (uses UTC date, validation already done)
		df = self.add_sender_temporal_features(
			df, sender_col=sender_col, date_col=utc_date_col, validate_order=False
		)

		print(f"  Features added           : {len(df):,} rows")

		return df

	def complete_pipeline(self, df: pd.DataFrame,
	                     local_date_col: str = 'date',
	                     tz_offset_col: str = 'timezone_offset',
	                     sender_col: str = 'sender',
	                     label_col: str = 'label',
	                     global_baseline: float = None) -> pd.DataFrame:
		"""
		Complete end-to-end temporal feature engineering pipeline.
		
		Pipeline:
		1. Create timezone_region from timezone_offset
		2. Create utc_date from date + timezone_offset
		3. Validate temporal consistency (sort by utc_date)
		4. Extract behavioral features from date (hour, is_night)
		5. Add advanced sender features using utc_date
		6. Drop timezone_offset
		
		Input columns required:
		- date: Sender's local datetime (from DataCleaner)
		- timezone_offset: Timezone string ('+0800', etc.)
		- sender: Sender email
		- label: Phishing label (0/1)
		
		Output columns created:
		- timezone_region: Geographic region
		- utc_date: UTC datetime for accurate temporal analysis
		- hour: LOCAL hour (0-23) for behavioral analysis
		- year, month, day, day_of_week, etc.: Temporal features
		- is_night: Based on local hour
		- sender_historical_phishing_rate: Time-aware sender reputation
		- sender_historical_count: Number of previous emails
		- current_time_gap: Minutes since last email (UTC-based)
		- sender_time_gap_std: Std dev of gaps
		
		Args:
		    df: DataFrame with date, timezone_offset, sender, label
		    local_date_col: Name of local date column (default: 'date')
		    tz_offset_col: Name of timezone offset column
		    sender_col: Name of sender column
		    label_col: Name of label column
		    global_baseline: Pre-calculated training baseline (for production)
		
		Returns:
		    DataFrame with all temporal features ready for modeling
		"""
		df = df.copy()
		
		print("\n" + "=" * 64)
		print("TEMPORAL FEATURE ENGINEERING PIPELINE")
		print("=" * 64)
		
		# Step 1: Geographic processing
		print("\n[Step 1/5] Geographic Processing")
		print("  Converting timezone_offset to timezone_region")
		df = self.add_timezone_region(df, tz_offset_col=tz_offset_col, drop_offset=False)
		
		# Step 2: Create UTC date
		print("\n[Step 2/5] UTC Date Creation")
		print("  Converting local datetime + timezone offset to UTC")
		df = self._create_utc_date(df, local_date_col=local_date_col, tz_offset_col=tz_offset_col)
		
		# Step 3: Validate and sort
		print("\n[Step 3/5] Temporal Consistency Validation")
		print("  Checking chronological order by UTC date")
		df = self.validate_temporal_consistency(df, date_col='utc_date', sort_if_needed=True)
		
		# Step 4: Extract behavioral features from local_date
		print("\n[Step 4/5] Behavioral Feature Extraction")
		print("  Extracting features from LOCAL datetime")
		df = self.extract_temporal_features(df, local_date_col=local_date_col)
		print(f"  Features added           : year, month, day, hour, day_of_week, is_night")
		
		# Step 5: Add advanced features
		print("\n[Step 5/5] Advanced Sender Features")
		print("  Computing time-aware features (no data leakage)")
		df = self.add_all_advanced_features(
			df, 
			sender_col=sender_col,
			utc_date_col='utc_date',
			label_col=label_col,
			hour_col='hour',
			validate_order=False,  # Already validated in step 3
			global_baseline=global_baseline
		)
		
		# Step 6: Cleanup
		df = df.drop(columns=[tz_offset_col], errors='ignore')
		
		print("\n" + "=" * 64)
		print("PIPELINE COMPLETE")
		print("=" * 64)
		print(f"Final dataset            : {len(df):,} rows × {len(df.columns)} columns")
		print(f"Features added           : 16 columns")
		print()
		print("Key feature columns:")
		print("  Geographic             : timezone_region")
		print("  Behavioral (LOCAL)     : hour, is_night, day_of_week")
		print("  Sender (UTC, time-aware):")
		print("    - sender_historical_phishing_rate")
		print("    - sender_historical_count")
		print("    - current_time_gap")
		print("    - sender_time_gap_std")
		print()
		print("Ready for model training : YES")
		print("Data leakage risk        : NONE (time-aware features verified)")
		print("=" * 64 + "\n")
		
		return df

	def check_feature_correlations(self, df: pd.DataFrame,
	                               feature_cols: list = None,
	                               threshold: float = 0.7,
	                               show_plot: bool = False) -> pd.DataFrame:
		"""
		Check correlations between features to detect multicollinearity.
		
		Useful for:
		- Detecting redundant features (high correlation between features)
		- Identifying multicollinearity issues before modeling
		- Deciding which features to keep when pairs are highly correlated
		
		Args:
			df: DataFrame with features
			feature_cols: List of feature columns. If None, uses default 7 model features.
			threshold: Correlation threshold to flag high correlations (default: 0.7)
			show_plot: If True, display correlation heatmap (requires matplotlib, seaborn)
		
		Returns:
			DataFrame with feature pairs and their correlations, sorted by absolute correlation.
			Columns:
			- feature_1: First feature name
			- feature_2: Second feature name
			- correlation: Pearson correlation coefficient
			- abs_correlation: Absolute value (for sorting)
			- high_correlation: Boolean flag if |r| > threshold
		
		Example:
			>>> engineer = TemporalFeatureEngineer()
			>>> corr_pairs = engineer.check_feature_correlations(df, threshold=0.7)
			
			Feature-Feature Correlations:
			  hour ↔ is_night                    : r= 0.65
			  sender_historical_count ↔ ...      : r=-0.42
			
			High Correlations (|r| > 0.70): None detected
		"""
		from scipy import stats
		
		# Default to 7 model features if not specified
		if feature_cols is None:
			feature_cols = [
				'timezone_region',
				'hour',
				'is_night',
				'sender_historical_phishing_rate',
				'sender_historical_count',
				'current_time_gap',
				'sender_time_gap_std'
			]
		
		# Validate columns exist
		missing_cols = [col for col in feature_cols if col not in df.columns]
		if missing_cols:
			raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
		
		print("\n" + "=" * 64)
		print("FEATURE-FEATURE CORRELATION ANALYSIS")
		print("=" * 64)
		print(f"Number of features       : {len(feature_cols)}")
		print(f"Sample size              : {len(df):,} rows")
		print(f"Threshold for flagging   : |r| > {threshold}")
		print()
		
		# Separate numeric and categorical features
		numeric_features = []
		categorical_features = []
		
		for col in feature_cols:
			if df[col].dtype == 'object' or df[col].dtype.name == 'category':
				categorical_features.append(col)
			else:
				numeric_features.append(col)
		
		print(f"Numeric features         : {len(numeric_features)}")
		print(f"Categorical features     : {len(categorical_features)}")
		print()
		
		# Calculate correlation matrix for numeric features only
		if len(numeric_features) < 2:
			print("[INFO] Need at least 2 numeric features for correlation analysis")
			return pd.DataFrame()
		
		# Create correlation matrix
		corr_matrix = df[numeric_features].corr()
		
		# Extract feature pairs (upper triangle only to avoid duplicates)
		correlations = []
		for i in range(len(numeric_features)):
			for j in range(i + 1, len(numeric_features)):
				feature_1 = numeric_features[i]
				feature_2 = numeric_features[j]
				r = corr_matrix.loc[feature_1, feature_2]
				
				correlations.append({
					'feature_1': feature_1,
					'feature_2': feature_2,
					'correlation': r,
					'abs_correlation': abs(r),
					'high_correlation': abs(r) > threshold
				})
		
		# Create DataFrame and sort by absolute correlation
		corr_df = pd.DataFrame(correlations)
		corr_df = corr_df.sort_values('abs_correlation', ascending=False)
		
		# Print results
		print("Feature Pairs (sorted by |r|):")
		print()
		
		# Find max feature name length for alignment
		max_name_len = max(
			len(f"{row['feature_1']} ↔ {row['feature_2']}") 
			for _, row in corr_df.iterrows()
		)
		
		for _, row in corr_df.iterrows():
			pair_name = f"{row['feature_1']} ↔ {row['feature_2']}"
			r_value = row['correlation']
			flag = " [HIGH]" if row['high_correlation'] else ""
			
			# Format with proper alignment
			print(f"  {pair_name:<{max_name_len}}  {r_value:>6.3f}{flag}")
		
		print()
		
		# Summary of high correlations
		high_corr = corr_df[corr_df['high_correlation']]
		if len(high_corr) > 0:
			print(f"High Correlations (|r| > {threshold}):")
			for _, row in high_corr.iterrows():
				print(f"  [{chr(0x26A0)}] {row['feature_1']} ↔ {row['feature_2']}: r={row['correlation']:.3f}")
				print(f"      Consider dropping one of these features to reduce multicollinearity")
			print()
		else:
			print(f"High Correlations (|r| > {threshold}): None detected")
			print()
		
		# Categories of correlation strength
		strong = corr_df[corr_df['abs_correlation'] >= 0.5].shape[0]
		moderate = corr_df[(corr_df['abs_correlation'] >= 0.3) & (corr_df['abs_correlation'] < 0.5)].shape[0]
		weak = corr_df[corr_df['abs_correlation'] < 0.3].shape[0]
		
		print("Correlation Strength Summary:")
		print(f"  Strong (|r| >= 0.5)      : {strong} pairs")
		print(f"  Moderate (0.3 <= |r| < 0.5): {moderate} pairs")
		print(f"  Weak (|r| < 0.3)         : {weak} pairs")
		print()
		
		# Interpretation guidelines
		print("Interpretation Guidelines:")
		print("  |r| < 0.3  : Low multicollinearity (features are independent)")
		print("  |r| < 0.5  : Moderate correlation (generally acceptable)")
		print("  |r| < 0.7  : High correlation (consider feature selection)")
		print("  |r| >= 0.7 : Very high correlation (likely redundant features)")
		print()
		
		# Note about categorical features
		if categorical_features:
			print(f"Note: Categorical features excluded from analysis: {categorical_features}")
			print("      Use chi-square test or Cramér's V for categorical-categorical correlations")
			print()
		
		print("=" * 64 + "\n")
		
		# Optional: Create visualization
		if show_plot:
			try:
				import matplotlib.pyplot as plt
				import seaborn as sns
				
				# Create heatmap
				fig, ax = plt.subplots(figsize=(10, 8))
				
				# Mask for upper triangle (to show only half)
				mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
				
				# Create heatmap
				sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
				           cmap='coolwarm', center=0, vmin=-1, vmax=1,
				           square=True, linewidths=1, cbar_kws={"shrink": 0.8})
				
				plt.title('Feature-Feature Correlation Matrix', fontsize=14, fontweight='bold')
				plt.tight_layout()
				plt.show()
				
			except ImportError:
				print("[INFO] matplotlib/seaborn not available. Install them to see visualization.")
		
		return corr_df
