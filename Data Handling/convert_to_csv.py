import pandas as pd


"""
Converts a
"""
# Path to a single log file
input_file = "logs-txt/slurm_accounting_2025-10-31.txt"
output_file = "slurm_accounting_2025-10-31.csv"

# Read the log file
df = pd.read_csv(
    input_file,
    sep="|",              # pipe-delimited
    engine="python",
    skipinitialspace=True # remove spaces around entries
)

# Save as CSV
df.to_csv(output_file, index=False)

print(f"Converted {input_file} → {output_file} ({len(df):,} rows)")