from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


NODE_BATCH_SIZE = 500_000

NODE_COLUMNS = [
    "event_index",
    "total_CPUs",
    "total_GPUs",
    "total_memory",
    "CPUs_in_use",
    "GPUs_in_use",
    "memory_in_use",
    "power_consumption",
]


def _add_aggregate_frame(accumulator, chunk_agg):
    if accumulator is None:
        return chunk_agg
    return accumulator.add(chunk_agg, fill_value=0.0)


def load_simulation_output(config):
    events_dir = Path(config["input_events_directory"])
    nodes_dir  = Path(config["input_nodes_directory"])
    node_batch_size = int(config.get("node_batch_size", NODE_BATCH_SIZE))
    return create_dataframes(events_dir, nodes_dir, node_batch_size=node_batch_size)


def create_dataframes(events_dir, nodes_dir, node_batch_size=NODE_BATCH_SIZE):
    event_files = sorted(events_dir.glob("*.parquet"))
    node_files  = sorted(nodes_dir.glob("*.parquet"))

    events_df = pd.concat((pd.read_parquet(f) for f in event_files), ignore_index=True)
    events_df = events_df.set_index("event_index", drop=True)

    # Aggregate node records in record batches so single large parquet files stay memory-safe.
    node_agg = None
    active_agg = None

    for f in node_files:
        parquet_file = pq.ParquetFile(f)
        for batch in parquet_file.iter_batches(columns=NODE_COLUMNS, batch_size=node_batch_size):
            chunk = batch.to_pandas()
            chunk["is_active"] = chunk["power_consumption"] > 0

            chunk_node_agg = chunk.groupby("event_index", sort=False).agg(
                total_cpus=("total_CPUs", "sum"),
                total_gpus=("total_GPUs", "sum"),
                total_mem=("total_memory", "sum"),
                cpus_used=("CPUs_in_use", "sum"),
                gpus_used=("GPUs_in_use", "sum"),
                mem_used=("memory_in_use", "sum"),
                cluster_power_W=("power_consumption", "sum"),
                active_nodes=("is_active", "sum"),
            )
            node_agg = _add_aggregate_frame(node_agg, chunk_node_agg)

            active_chunk = chunk.loc[chunk["is_active"]]
            if active_chunk.empty:
                continue

            free_gpu = np.where(
                active_chunk["total_GPUs"] > 0,
                active_chunk["total_GPUs"] - active_chunk["GPUs_in_use"],
                0,
            )

            active_metrics = pd.DataFrame(
                {
                    "event_index": active_chunk["event_index"].to_numpy(),
                    "free_cpus": (active_chunk["total_CPUs"] - active_chunk["CPUs_in_use"]).to_numpy(),
                    "free_mem": (active_chunk["total_memory"] - active_chunk["memory_in_use"]).to_numpy(),
                    "free_gpus": free_gpu,
                    "gpu_active_nodes": (active_chunk["total_GPUs"] > 0).astype("int64").to_numpy(),
                }
            )
            chunk_active_agg = active_metrics.groupby("event_index", sort=False).agg(
                free_cpus_sum=("free_cpus", "sum"),
                free_mem_sum=("free_mem", "sum"),
                free_gpus_sum=("free_gpus", "sum"),
                gpu_active_nodes=("gpu_active_nodes", "sum"),
            )
            active_agg = _add_aggregate_frame(active_agg, chunk_active_agg)

    if node_agg is None:
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
    else:
        node_agg = node_agg.sort_index()
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

    if active_agg is None:
        active_agg = pd.DataFrame(columns=["free_cpus_sum", "free_gpus_sum", "free_mem_sum", "gpu_active_nodes"])
    else:
        active_agg = active_agg.sort_index()
        active_cols = ["free_cpus_sum", "free_gpus_sum", "free_mem_sum", "gpu_active_nodes"]
        active_agg[active_cols] = active_agg[active_cols].round().astype("int64")

    return events_df, node_agg, active_agg
