import ast
import re
import pandas as pd
from pathlib import Path


def _node_family(node_name: str) -> str:
    return re.sub(r"\d+$", "", node_name)


def filter_to_single_node(parquet_path: str, output_dir: str, output_filename: str) -> Path:
    df = pd.read_parquet(parquet_path)

    if "action" in df.columns:
        starts = df[df["action"] == "start"].copy()
    else:
        starts = df.drop_duplicates(subset=["job_id"]).copy()

    matching_job_ids = set(starts[starts["nodes_required"] == 1]["job_id"])
    filtered = df[df["job_id"].isin(matching_job_ids)].copy()

    total_starts = len(starts)
    kept = len(matching_job_ids)
    dropped = total_starts - kept
    print(f"Single node filter:")
    print(f"  Total jobs:   {total_starts}")
    print(f"  Kept:         {kept}  ({kept / total_starts * 100:.1f}%)")
    print(f"  Dropped:      {dropped}  ({dropped / total_starts * 100:.1f}%)")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{output_filename}.parquet"
    filtered.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  Saved to:     {out_path}")
    return out_path


def filter_to_node_family(parquet_path: str, family: str, output_dir: str, output_filename: str) -> Path:
    """
    Keep only jobs whose real node selection is entirely within the given node family.
    Saves a new parquet with both start and finish events for matching jobs.
    Also rewrites allowed_node_types to just the target family.
    """
    df = pd.read_parquet(parquet_path)

    if "action" in df.columns:
        starts = df[df["action"] == "start"].copy()
    else:
        starts = df.drop_duplicates(subset=["job_id"]).copy()

    matching_job_ids = set()
    for _, row in starts.iterrows():
        raw = row.get("real_node_selection")
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        try:
            nodes = ast.literal_eval(str(raw))
        except (ValueError, SyntaxError):
            continue
        if not nodes:
            continue
        if all(_node_family(n) == family for n in nodes):
            matching_job_ids.add(row["job_id"])

    filtered = df[df["job_id"].isin(matching_job_ids)].copy()

    # Rewrite allowed_node_types to just the target family so the simulator uses only it
    filtered["allowed_node_types"] = str([family])

    total_starts = len(starts)
    kept = len(matching_job_ids)
    dropped = total_starts - kept
    print(f"Node family filter: '{family}'")
    print(f"  Total jobs:   {total_starts}")
    print(f"  Kept:         {kept}  ({kept / total_starts * 100:.1f}%)")
    print(f"  Dropped:      {dropped}  ({dropped / total_starts * 100:.1f}%)")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{output_filename}.parquet"
    filtered.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  Saved to:     {out_path}")
    return out_path
