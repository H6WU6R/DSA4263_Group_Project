# %%
import sys
import os
import time
import argparse
import pandas as pd
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Import our modules
from data_cleaning_finalized import DataCleaner
from feature_engineering_finalized import FeatureEngineer
from modeling_finalized import ModelTrainer

warnings.filterwarnings('ignore')

# Setup NLTK resources
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        print(f"  • Downloading {resource}...")
        nltk.download(resource, quiet=True)

stop_words = set(stopwords.words('english'))
stop_words.update(['re', 'fwd', 'subject'])
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

cleaner = DataCleaner(stop_words=stop_words, lemmatizer=lemmatizer, stemmer=stemmer)

# Load raw datasets
print("\n📂 Loading raw datasets...")
raw_datasets = [
    cleaner.load_raw_data('CEAS_08.csv'),
    cleaner.load_raw_data('Nazario.csv'),
    cleaner.load_raw_data('Nigerian_Fraud.csv'),
    cleaner.load_raw_data('SpamAssasin.csv')
]
print(f"   Loaded {len(raw_datasets)} datasets")
print(f"   Total rows: {sum(len(df) for df in raw_datasets):,}")

# Step 1: Clean and merge datasets
print("\n" + "=" * 64)
print("STEP 1: CLEAN AND MERGE DATASETS")
print("=" * 64)
merged_df = cleaner.clean_and_merge(raw_datasets, sender_col='sender', receiver_col='receiver')
print(f"Merged rows              : {len(merged_df):,}")

# Add email_id after merging
merged_df['email_id'] = merged_df.index

# Step 2: Clean dates
print("\n" + "=" * 64)
print("STEP 2: CLEAN DATES")
print("=" * 64)
cleaned_df = cleaner.clean_dates(merged_df, date_col='date')

# Step 3: Clean text
print("\n" + "=" * 64)
print("STEP 3: CLEAN TEXT")
print("=" * 64)
if 'full_text' in cleaned_df.columns:
    print("Cleaning 'full_text' column...")
    cleaned_df['full_text'] = cleaner.clean_text_column(cleaned_df['full_text'], show_progress=True)
    empty_after = (cleaned_df['full_text'] == "").sum()
    print(f"Empty after cleaning     : {empty_after:,} rows")
else:
    print("⚠️  'full_text' column not found!")

# Final summary
print("\n" + "=" * 64)
print("FINAL SUMMARY")
print("=" * 64)
print(f"Output rows              : {len(cleaned_df):,}")
print(f"Columns                  : {list(cleaned_df.columns)}")

# Save cleaned data
cleaner.save_processed_data(cleaned_df, 'cleaned_date_merge.csv')
print(f"\n💾 Cleaned data saved to: data/processed/cleaned_date_merge.csv")
print("=" * 64 + "\n")
# %%
