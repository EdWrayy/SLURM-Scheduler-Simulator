import ast
import json
import pandas as pd
from pathlib import Path
from backend.common.config import load_config


def _build_capacity_lookup(nodes_config: dict) -> dict:
    """
    Build {node_type: {cpu, gpu, memory_bytes}} from nodes config.
    When a node_type appears in multiple config entries (e.g. different ruby ranges),
    the first entry's capacities are used (they are identical for the same type).
    """
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


def compute_resource_weights(parquet_path: str, nodes_config_path: str) -> dict:
    """
    Compute normalized average resource-demand weights per node family.

    Args:
        parquet_path:      Path to the job-logs parquet file.
        nodes_config_path: Path to the nodes JSON config.

    Returns:
        dict mapping node_type -> {"cpu": w, "gpu": w, "memory": w, "num_jobs": n}
        Only families that had at least one job are included.
    """
    with open(nodes_config_path) as f:
        nodes_config = json.load(f)

    capacity = _build_capacity_lookup(nodes_config)

    df = pd.read_parquet(parquet_path)

    # One record per job (start events)
    if "action" in df.columns:
        jobs = df[df["action"] == "start"].copy()
    else:
        jobs = df.drop_duplicates(subset=["job_id"]).copy()

    # {family: {dim: [fractional demands]}}
    accum: dict[str, dict[str, list[float]]] = {
        nt: {"cpu": [], "gpu": [], "memory": []} for nt in capacity
    }

    for _, row in jobs.iterrows():
        raw = row.get("allowed_node_types")
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue

        try:
            families: list[str] = ast.literal_eval(str(raw))
        except (ValueError, SyntaxError):
            continue

        if not families:
            continue

        n_nodes = row["nodes_required"]
        if not n_nodes or n_nodes <= 0:
            continue

        cpu_demand = row.get("CPUs_required") or 0
        gpu_demand = row.get("GPUs_required") or 0
        mem_demand = row.get("memory_required") or 0

        for family in families:
            if family not in capacity:
                continue
            cap = capacity[family]

            cpu_frac = (
                cpu_demand / (n_nodes * cap["cpu"])
                if cap["cpu"] > 0
                else 0.0
            )
            gpu_frac = (
                gpu_demand / (n_nodes * cap["gpu"])
                if cap["gpu"] > 0
                else 0.0
            )
            mem_frac = (
                mem_demand / (n_nodes * cap["memory_bytes"])
                if cap["memory_bytes"] > 0
                else 0.0
            )

            accum[family]["cpu"].append(cpu_frac)
            accum[family]["gpu"].append(gpu_frac)
            accum[family]["memory"].append(mem_frac)

    weights = {}
    for family, dims in accum.items():
        n = len(dims["cpu"])
        if n == 0:
            continue
        weights[family] = {
            "cpu": sum(dims["cpu"]) / n,
            "gpu": sum(dims["gpu"]) / n,
            "memory": sum(dims["memory"]) / n,
        }

    return weights


def save_weights(weights: dict, output_path: str) -> None:
    """Save per-family resource weights to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"Resource weights saved to: {output_path}")


if __name__ == "__main__":
    config = load_config(Path(__file__).with_name("config.json"))
    weights = compute_resource_weights(
        parquet_path=config["weights_parquet_file"],
        nodes_config_path=config["nodes_config_file"],
    )
    output_file = str(Path(config["weights_output_directory"]) / "resource_weights.json")
    save_weights(weights, output_file)
