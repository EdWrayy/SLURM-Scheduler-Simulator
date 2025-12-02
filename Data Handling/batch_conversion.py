import pandas as pd
from pathlib import Path
from datetime import datetime

# Configuration
logs_dir = Path("logs-txt")
output_file = "slurm_accounting_full_year.parquet"

# Find all log files
log_files = sorted(logs_dir.glob("slurm_accounting_*.txt"))
print(f"Found {len(log_files)} log files")

# Increase CSV field size limit for large fields
import csv

# Read and concatenate all files
dataframes = []
failed_files = []

for log_file in log_files:
    print(f"Processing {log_file.name}...")

    # Try multiple strategies to handle problematic files
    strategies = [
        # Strategy 1: Default (UTF-8, strict parsing)
        ("Default", {
            'sep': '|',
            'engine': 'python',
            'skipinitialspace': True,
        }),
        # Strategy 2: Latin-1 encoding (handles UTF-8 decode errors)
        ("Latin-1 encoding", {
            'sep': '|',
            'engine': 'python',
            'skipinitialspace': True,
            'encoding': 'latin-1',
        }),
        # Strategy 3: Skip bad lines (handles field count mismatches)
        ("Skip bad lines", {
            'sep': '|',
            'engine': 'python',
            'skipinitialspace': True,
            'on_bad_lines': 'skip',
        }),
        # Strategy 4: Combination of both
        ("Latin-1 + skip bad lines", {
            'sep': '|',
            'engine': 'python',
            'skipinitialspace': True,
            'encoding': 'latin-1',
            'on_bad_lines': 'skip',
        }),
    ]

    success = False
    for strategy_name, kwargs in strategies:
        try:
            df = pd.read_csv(log_file, **kwargs)
            dataframes.append(df)
            if strategy_name != "Default":
                print(f"  ✓ {len(df):,} rows ({strategy_name})")
            else:
                print(f"  ✓ {len(df):,} rows")
            success = True
            break
        except Exception as e:
            continue

    if not success:
        print(f"  ✗ All strategies failed")
        failed_files.append(log_file.name)

# Combine all dataframes
print("\nCombining all data...")
combined_df = pd.concat(dataframes, ignore_index=True)

# Filter out .batch and .extern jobs
print("Filtering out .batch and .extern jobs...")
if "JobID" in combined_df.columns:
    rows_before = len(combined_df)
    combined_df["JobID"] = combined_df["JobID"].astype(str)
    combined_df = combined_df[
        ~combined_df["JobID"].str.endswith(".batch") &
        ~combined_df["JobID"].str.endswith(".extern")
    ]
    rows_after = len(combined_df)
    print(f"  Removed {rows_before - rows_after:,} .batch/.extern jobs")

# Filter out jobs with invalid JobIDs
print("Filtering out invalid job IDs...")
rows_before = len(combined_df)
combined_df = combined_df[
    (combined_df["JobID"] != "nan") &
    (combined_df["JobID"] != "None") &
    (combined_df["JobID"] != "") &
    (combined_df["JobID"] != "0") &
    combined_df["JobID"].notna()
]
if rows_before > len(combined_df):
    print(f"  Removed {rows_before - len(combined_df):,} jobs with invalid IDs")

# Fix data types before saving
print("Fixing data types...")

# Convert time columns to datetime for better compression
for col in ["Start", "End", "Submit", "Eligible"]:
    if col in combined_df.columns:
        combined_df[col] = pd.to_datetime(combined_df[col], errors='coerce')

# Filter out jobs with invalid Start/End times
print("Filtering out jobs with invalid start/end times...")
if "Start" in combined_df.columns and "End" in combined_df.columns:
    rows_before = len(combined_df)
    combined_df = combined_df[
        combined_df["Start"].notna() &
        combined_df["End"].notna() &
        (combined_df["Start"] < combined_df["End"])  # End must be after Start
    ]
    if rows_before > len(combined_df):
        print(f"  Removed {rows_before - len(combined_df):,} jobs with invalid times")

# Convert ALL numeric-like columns to proper types
# This is safer than trying to list them all manually
for col in combined_df.columns:
    # Skip datetime columns we already converted
    if col in ["Start", "End", "Submit", "Eligible"]:
        continue

    # Try to convert to numeric if possible
    # This handles mixed types (strings/floats) gracefully
    if combined_df[col].dtype == 'object':
        # First try numeric conversion
        converted = pd.to_numeric(combined_df[col], errors='coerce')
        # Only keep conversion if at least some values were numeric
        if converted.notna().sum() > 0:
            combined_df[col] = converted
        else:
            # Keep as string
            combined_df[col] = combined_df[col].astype(str)

# Optional: Remove duplicates if any
print(f"Total rows before deduplication: {len(combined_df):,}")
combined_df = combined_df.drop_duplicates()
print(f"Total rows after deduplication: {len(combined_df):,}")

# Optional: Sort by Start time for better compression and query performance
if "Start" in combined_df.columns:
    combined_df = combined_df.sort_values("Start")

# Save as Parquet with optimal compression
print(f"\nSaving to {output_file}...")
combined_df.to_parquet(
    output_file,
    index=False,
    engine="pyarrow",
    compression="snappy",  # Good balance of speed and compression
    # Alternative: compression="gzip" for better compression but slower
)

print(f"\n✓ Complete!")
print(f"  Output file: {output_file}")
print(f"  Total rows: {len(combined_df):,}")
print(f"  Columns: {len(combined_df.columns)}")
print(f"  Date range: {combined_df['Start'].min()} to {combined_df['Start'].max()}" if 'Start' in combined_df.columns else "")

# Show file size
import os
file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"  File size: {file_size_mb:.2f} MB")

# Report any failed files
if failed_files:
    print(f"\n⚠ Warning: {len(failed_files)} files could not be processed:")
    for filename in failed_files:
        print(f"  - {filename}")
else:
    print(f"\n✓ All {len(log_files)} files processed successfully!")