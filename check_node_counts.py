import pandas as pd

FILE_PATH = r"C:\Users\edjwr\Documents\Part 3 Project\Iridis X Data\Simulation Input Parquet Files\entire-history-logs.parquet"

df = pd.read_parquet(FILE_PATH)

multi_node_jobs = (df['nodes_required'] >= 2).sum()
total_jobs = len(df)

print(f"Total jobs:           {total_jobs}")
print(f"Jobs with 2+ nodes:   {multi_node_jobs} ({multi_node_jobs / total_jobs * 100:.1f}%)")
print()
print("Node count distribution:")
print(df['nodes_required'].value_counts().sort_index().to_string())
