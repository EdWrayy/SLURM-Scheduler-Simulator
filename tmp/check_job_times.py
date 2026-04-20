import pandas as pd

PARQUET_PATH = r"C:\Users\edjwr\Documents\Part 3 Project\Iridis X Data\Simulation Input Parquet Files\newest_logs_year.parquet"

df = pd.read_parquet(PARQUET_PATH)
starts = df[df["action"] == "start"].copy()
starts["start_time"] = pd.to_datetime(starts["start_time"])
starts["end_time"] = pd.to_datetime(starts["end_time"])

print(f"Total jobs: {len(starts)}")
print(f"\nstart_time range: {starts['start_time'].min()}  ->  {starts['start_time'].max()}")
print(f"end_time range:   {starts['end_time'].min()}  ->  {starts['end_time'].max()}")

print(f"\n--- Sample (10 per quarter) ---")
n = len(starts)
for q, label in enumerate(["Q1", "Q2", "Q3", "Q4"]):
    lo = int(q * n / 4)
    hi = int((q + 1) * n / 4)
    sample = starts.iloc[lo:hi:max(1, (hi - lo) // 10)].head(10)
    print(f"\n  {label}:")
    for _, row in sample.iterrows():
        duration = row['end_time'] - row['start_time']
        print(f"    job_id={row['job_id']}  start={row['start_time']}  end={row['end_time']}  duration={duration}")
