import pandas as pd

# Path to a single log file
input_file = "logs-txt/slurm_accounting_2025-10-31.txt"
output_file = "slurm_accounting_2025-10-31.parquet"

df = pd.read_csv(
    input_file,
    sep="|",              # pipe-delimited
    engine="python",
    skipinitialspace=True # remove spaces around entries
)

# Save as Parquet
df.to_parquet(output_file, index=False, engine="pyarrow", compression="snappy")

print(f"Converted {input_file} -> {output_file} ({len(df):,} rows)")
