import ast
import json
import re
import pandas as pd


def _build_capacity_lookup(nodes_config: dict) -> dict:
    capacity = {}
    for node in nodes_config["nodes"]:
        nt = node["node_type"]
        if nt not in capacity:
            capacity[nt] = {
                "cpu": node["cpus_per_node"],
                "gpu": node["gpus_per_node"],
                "memory_bytes": node["memory_mb"] * 1024 * 1024,
            }
    return capacity


def _node_type(node_name: str) -> str:
    return re.sub(r"\d+$", "", node_name)


def single_node_utilisation(jobs: pd.DataFrame, nodes_config_path: str) -> None:
    with open(nodes_config_path) as f:
        capacity = _build_capacity_lookup(json.load(f))

    single = jobs[jobs["nodes_required"] == 1].copy()
    if single.empty:
        print("No single-node jobs found.")
        return

    cpu_pcts, gpu_pcts, mem_pcts = [], [], []
    skipped = 0

    for _, row in single.iterrows():
        raw = row.get("real_node_selection")
        if pd.isna(raw):
            skipped += 1
            continue
        try:
            nodes = ast.literal_eval(str(raw))
        except (ValueError, SyntaxError):
            skipped += 1
            continue
        if not nodes:
            skipped += 1
            continue

        nt = _node_type(nodes[0])
        cap = capacity.get(nt)
        if cap is None:
            skipped += 1
            continue

        cpu_req = row.get("CPUs_required") or 0
        gpu_req = row.get("GPUs_required") or 0
        mem_req = row.get("memory_required") or 0

        cpu_pcts.append(cpu_req / cap["cpu"] * 100 if cap["cpu"] > 0 else None)
        gpu_pcts.append(gpu_req / cap["gpu"] * 100 if cap["gpu"] > 0 else None)
        mem_pcts.append(mem_req / cap["memory_bytes"] * 100 if cap["memory_bytes"] > 0 else None)

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print("Single-node job resource utilisation (mean % of node capacity):")
    print(f"  CPU:    {_mean(cpu_pcts):5.1f}%")
    print(f"  GPU:    {_mean(gpu_pcts):5.1f}%  (GPU jobs only)")
    print(f"  Memory: {_mean(mem_pcts):5.1f}%")
    if skipped:
        print(f"  ({skipped} jobs skipped due to missing/unknown node data)")


def workload_breakdown(parquet_path: str, nodes_config_path: str | None = None) -> None:
    df = pd.read_parquet(parquet_path)

    # Only count each job once via start events
    if 'action' in df.columns:
        jobs = df[df['action'] == 'start'].copy()
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

    if nodes_config_path:
        print()
        single_node_utilisation(jobs, nodes_config_path)
