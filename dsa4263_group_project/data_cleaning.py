"""
Data Cleaning Module

This module contains functions for cleaning and preprocessing email data,
including timezone extraction, date parsing, and temporal feature extraction.
"""

import re
import pandas as pd
from typing import Optional


def extract_timezone_offset(date_str: str) -> Optional[str]:
    """Extract timezone offset from date string (e.g., '+0800', '-0500')"""
    if pd.isna(date_str):
        return None
    
    # Match timezone patterns: +HHMM or -HHMM
    match = re.search(r'([+-]\d{4})(?:\s|$)', str(date_str))
    return match.group(1) if match else None


def timezone_offset_to_hours(offset_str: str) -> Optional[float]:
    """Convert timezone offset string to hours (e.g., '+0800' → 8.0)"""
    if not offset_str or len(offset_str) != 5:
        return None
    
    try:
        sign = 1 if offset_str[0] == '+' else -1
        hours = int(offset_str[1:3])
        minutes = int(offset_str[3:5])
        return sign * (hours + minutes / 60.0)
    except (ValueError, IndexError):
        return None


def map_timezone_to_region(tz_hours: float) -> str:
    """Map timezone offset to geographic region"""
    if pd.isna(tz_hours):
        return 'Unknown'
    
    if -12 <= tz_hours < -3:
        return 'Americas'
    elif -3 <= tz_hours < 3:
        return 'Europe/Africa'
    elif 3 <= tz_hours < 6:
        return 'Middle East/South Asia'
    elif 6 <= tz_hours < 10:
        return 'APAC'
    elif 10 <= tz_hours <= 14:
        return 'Oceania/Pacific'
    else:
        return 'Unknown'


def clean_email_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean and preprocess email data with timezone awareness.
    
    Steps:
    1. Extract and validate timezone information
    2. Parse and standardize dates
    3. Extract temporal features
    4. Remove invalid/anomalous data
    5. Add derived features
    
    Args:
        df: Raw email dataframe
        verbose: Print progress messages
        
    Returns:
        Cleaned dataframe with temporal features
    """
    if verbose:
        print("\n[Cleaning] Starting data cleaning...")
        print(f"  Initial size: {len(df):,} emails")
    
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # Step 1: Timezone extraction
    if verbose:
        print("\n[Cleaning] Extracting timezone information...")
    
    df_clean['timezone_offset_str'] = df_clean['date'].apply(extract_timezone_offset)
    df_clean['timezone_hours'] = df_clean['timezone_offset_str'].apply(timezone_offset_to_hours)
    df_clean['timezone_region'] = df_clean['timezone_hours'].apply(map_timezone_to_region)
    
    # Remove rows with invalid timezone
    df_clean = df_clean[df_clean['timezone_region'] != 'Unknown'].copy()
    
    if verbose:
        tz_removed = initial_count - len(df_clean)
        print(f"  ✓ Removed {tz_removed:,} rows with invalid timezone")
    
    # Step 2: Date parsing
    if verbose:
        print("\n[Cleaning] Parsing dates...")
    
    df_clean['date'] = pd.to_datetime(df_clean['date'], format='ISO8601', errors='coerce')
    
    # Remove unparseable dates
    before_date_filter = len(df_clean)
    df_clean = df_clean[df_clean['date'].notna()].copy()
    
    # Remove anomalous dates (outside 1990-2025)
    df_clean = df_clean[
        (df_clean['date'].dt.year >= 1990) & 
        (df_clean['date'].dt.year <= 2025)
    ].copy()
    
    if verbose:
        date_removed = before_date_filter - len(df_clean)
        print(f"  ✓ Removed {date_removed:,} rows with invalid dates")
    
    # Step 3: Extract temporal features
    if verbose:
        print("\n[Cleaning] Extracting temporal features...")
    
    df_clean['year'] = df_clean['date'].dt.year
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['day'] = df_clean['date'].dt.day
    df_clean['hour'] = df_clean['date'].dt.hour
    df_clean['day_of_week'] = df_clean['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df_clean['day_name'] = df_clean['date'].dt.day_name()
    df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
    
    # Step 4: Clean sender/receiver
    if verbose:
        print("\n[Cleaning] Cleaning sender/receiver fields...")
    
    df_clean = df_clean.dropna(subset=['sender', 'receiver'])
    df_clean['sender'] = df_clean['sender'].astype(str).str.strip().str.lower()
    df_clean['receiver'] = df_clean['receiver'].astype(str).str.strip().str.lower()
    df_clean = df_clean[(df_clean['sender'] != 'nan') & (df_clean['receiver'] != 'nan')].copy()
    
    # Step 5: Remove duplicates
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['sender', 'receiver', 'date'], keep='first')
    
    if verbose:
        dedup_removed = before_dedup - len(df_clean)
        print(f"  ✓ Removed {dedup_removed:,} duplicate emails")
        print(f"\n  ✅ Final: {len(df_clean):,} emails ({len(df_clean)/initial_count*100:.1f}% retention)")
    
    return df_clean


def clean_text(text: str, stop_words, lemmatizer) -> str:
    """
    Clean text: lowercase, remove punctuation, tokenize, remove stopwords, lemmatize.
    
    Args:
        text: Raw text string
        stop_words: Set of stopwords to remove
        lemmatizer: NLTK lemmatizer object
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text) or text == "":
        return ""
    
    # Lowercase
    text = str(text).lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Lemmatize
    try:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    except:
        pass
    
    return ' '.join(tokens)
