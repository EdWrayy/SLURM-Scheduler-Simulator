import pandas as pd


def workload_breakdown(parquet_path: str) -> None:
    df = pd.read_parquet(parquet_path)

    # Only count each job once via submit events
    if 'action' in df.columns:
        jobs = df[df['action'] == 'submit'].copy()
        if jobs.empty:
            jobs = df.drop_duplicates(subset=['job_id'])
    else:
        jobs = df.drop_duplicates(subset=['job_id'])

    total = len(jobs)
    multi = (jobs['nodes_required'] >= 2).sum()
    single = total - multi

    print(f"Total jobs:              {total}")
    print(f"Single-node jobs (1):    {single} ({single / total * 100:.1f}%)")
    print(f"Multi-node jobs (2+):    {multi} ({multi / total * 100:.1f}%)")
    print()
    print("Node count distribution:")
    dist = jobs['nodes_required'].value_counts().sort_index()
    for n_nodes, count in dist.items():
        print(f"  {n_nodes:>3} node(s): {count:>6} jobs  ({count / total * 100:.1f}%)")
