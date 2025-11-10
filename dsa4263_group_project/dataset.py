# %%
"""Data acquisition and generation helpers."""
# Step-by-step script to import four raw CSV files as pandas DataFrames
"""Step-by-step script to import four raw CSV files as pandas DataFrames."""

import pandas as pd

from dsa4263_group_project.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ensure_directories,
)

# Use package configuration for the four raw CSV files
ceas_df = pd.read_csv(RAW_DATA_DIR / "CEAS_08.csv")
nazario_df = pd.read_csv(RAW_DATA_DIR / "Nazario.csv")
nigerian_df = pd.read_csv(RAW_DATA_DIR / "Nigerian_Fraud.csv")
spamassassin_df = pd.read_csv(RAW_DATA_DIR / "SpamAssasin.csv")

# %%

# Fill missing receiver values with unique labels and drop rows where sender is missing
def fill_receiver_and_drop_sender(df, receiver_col="receiver", sender_col="sender"):
	# Fill missing receiver with unique labels
	na_indices = df[df[receiver_col].isnull()].index
	for i, idx in enumerate(na_indices, 1):
		df.at[idx, receiver_col] = f"na{i}"
	# Drop rows where sender is missing
	df = df.dropna(subset=[sender_col])
	return df

ceas_df = fill_receiver_and_drop_sender(ceas_df)
nazario_df = fill_receiver_and_drop_sender(nazario_df)
nigerian_df = fill_receiver_and_drop_sender(nigerian_df)
spamassassin_df = fill_receiver_and_drop_sender(spamassassin_df)


# Merge all DataFrames
merged_df = pd.concat([ceas_df, nazario_df, nigerian_df, spamassassin_df], ignore_index=True)


# Ensure required directories exist before writing outputs
ensure_directories()

# Output to processed folder as graph_merge.csv
output_path = PROCESSED_DATA_DIR / "graph_merge.csv"
merged_df.to_csv(output_path, index=False)
print(f"Merged data saved to {output_path}")

# Output another file with rows missing 'date' dropped
merged_nodate_df = merged_df.dropna(subset=["date"])
output_nodate_path = PROCESSED_DATA_DIR / "date_merge.csv"
merged_nodate_df.to_csv(output_nodate_path, index=False)
print(f"Merged data with no missing date saved to {output_nodate_path}")
# %%
