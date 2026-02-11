import numpy as np
import pandas as pd



def calculate_all_metrics(events_df: pd.DataFrame, nodes_df: pd.DataFrame):

    total_time_s = 0.0
    total_energy_J = 0.0

    cpu_util_dt = 0.0
    gpu_util_dt = 0.0
    mem_util_dt = 0.0
    cluster_power_dt = 0.0

    snapshots = []

    nodes_groups = dict(tuple(nodes_df.groupby("event_index", sort=True)))

    for event_index, event in events_df.iterrows():

        node_states = nodes_groups.get(event_index)
        if node_states is None:
            continue  # or raise ValueError(f"No node rows for event_index={event_index}")

        time = event["time"]
        dt = float(event["dt"])

        if dt <= 0.0:
            continue

        total_time_s += dt

        total_cpus = node_states["total_CPUs"].sum()
        total_gpus = node_states["total_GPUs"].sum()
        total_mem  = node_states["total_memory"].sum()

        cpus_used = node_states["CPUs_in_use"].sum()
        gpus_used = node_states["GPUs_in_use"].sum()
        mem_used  = node_states["memory_in_use"].sum()

        cpu_util = cpus_used / total_cpus if total_cpus > 0 else 0.0
        gpu_util = gpus_used / total_gpus if total_gpus > 0 else 0.0
        mem_util = mem_used  / total_mem  if total_mem  > 0 else 0.0

        cluster_power_W = node_states["power_usage"].sum()
        total_energy_J += cluster_power_W * dt

        cpu_util_dt += cpu_util * dt
        gpu_util_dt += gpu_util * dt
        mem_util_dt += mem_util * dt
        cluster_power_dt += cluster_power_W * dt

        snapshots.append({
            "event_index": event_index,
            "time": time,
            "dt": dt,
            "cpu_util": cpu_util,
            "gpu_util": gpu_util,
            "mem_util": mem_util,
            "cluster_power_W": cluster_power_W,
            "cumulative_energy_J": total_energy_J,
        })

    snapshots_df = pd.DataFrame(snapshots)

    results = {
        "total_time_s": total_time_s,
        "avg_cpu_util": cpu_util_dt / total_time_s if total_time_s > 0 else 0.0,
        "avg_gpu_util": gpu_util_dt / total_time_s if total_time_s > 0 else 0.0,
        "avg_mem_util": mem_util_dt / total_time_s if total_time_s > 0 else 0.0,
        "total_energy_J": total_energy_J,
        "avg_power_W": cluster_power_dt / total_time_s if total_time_s > 0 else 0.0,
    }

    return snapshots_df, results
