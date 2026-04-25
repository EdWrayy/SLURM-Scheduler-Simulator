from pathlib import Path

import pandas as pd


NODE_BATCH_SIZE = 500_000


NODE_COLUMNS = [
    "event_index",
    "total_cpus",
    "total_gpus",
    "total_mem",
    "cpus_used",
    "gpus_used",
    "mem_used",
    "cluster_power_W",
    "active_nodes",
    "free_cpus_sum",
    "free_gpus_sum",
    "free_mem_sum",
    "gpu_active_nodes",
]


def load_simulation_output(config):
    events_dir = Path(config["input_events_directory"])
    nodes_dir = Path(config["input_nodes_directory"])
    node_batch_size = int(config.get("node_batch_size", NODE_BATCH_SIZE))
    return create_dataframes(events_dir, nodes_dir, node_batch_size=node_batch_size)


def create_dataframes(events_dir, nodes_dir, node_batch_size=NODE_BATCH_SIZE):
    event_files = sorted(events_dir.glob("*.parquet"))
    node_files = sorted(nodes_dir.glob("*.parquet"))

    events_df = pd.concat((pd.read_parquet(f) for f in event_files), ignore_index=True)
    events_df = events_df.set_index("event_index", drop=True)

    if not node_files:
        node_agg = pd.DataFrame(
            columns=[
                "total_cpus",
                "total_gpus",
                "total_mem",
                "cpus_used",
                "gpus_used",
                "mem_used",
                "cluster_power_W",
                "active_nodes",
            ]
        )
        active_agg = pd.DataFrame(columns=["free_cpus_sum", "free_gpus_sum", "free_mem_sum", "gpu_active_nodes"])
        return events_df, node_agg, active_agg

    # node_batch_size kept for config compatibility, but node outputs are now compact.
    _ = node_batch_size

    node_df = pd.concat(
        (pd.read_parquet(f, columns=NODE_COLUMNS) for f in node_files),
        ignore_index=True,
    )

    # Keep one snapshot per event index if duplicates are present.
    node_df = node_df.drop_duplicates(subset=["event_index"], keep="last")
    node_df = node_df.set_index("event_index", drop=True).sort_index()

    node_agg = node_df[
        [
            "total_cpus",
            "total_gpus",
            "total_mem",
            "cpus_used",
            "gpus_used",
            "mem_used",
            "cluster_power_W",
            "active_nodes",
        ]
    ].copy()

    active_agg = node_df[
        [
            "free_cpus_sum",
            "free_gpus_sum",
            "free_mem_sum",
            "gpu_active_nodes",
        ]
    ].copy()

    int_cols = [
        "total_cpus",
        "total_gpus",
        "total_mem",
        "cpus_used",
        "gpus_used",
        "mem_used",
        "active_nodes",
    ]
    node_agg[int_cols] = node_agg[int_cols].round().astype("int64")
    node_agg["cluster_power_W"] = node_agg["cluster_power_W"].astype("float64")

    active_cols = ["free_cpus_sum", "free_gpus_sum", "free_mem_sum", "gpu_active_nodes"]
    active_agg[active_cols] = active_agg[active_cols].round().astype("int64")

    return events_df, node_agg, active_agg
