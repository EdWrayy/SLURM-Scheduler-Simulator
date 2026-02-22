import numpy as np
import pandas as pd


def weighted_quantile(values, quantiles, sample_weight):
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)

    mask = np.isfinite(values) & np.isfinite(sample_weight) & (sample_weight > 0)
    values = values[mask]
    sample_weight = sample_weight[mask]

    if values.size == 0:
        return np.array([np.nan] * len(quantiles), dtype=float)

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    cumulative = np.cumsum(sample_weight)
    cutoff = quantiles * cumulative[-1]
    return np.interp(cutoff, cumulative, values)


def calculate_all_metrics(events_df, nodes_df, config):
    total_time_s = 0.0
    total_energy_J = 0.0


    # time-weighted sums for means
    cpu_util_dt = 0.0
    gpu_util_dt = 0.0
    mem_util_dt = 0.0

    active_nodes_dt = 0.0
    node_hours_active = 0.0
    peak_active_nodes = 0

    # fragmentation on active nodes (time-weighted)
    free_cpu_active_dt = 0.0
    free_gpu_active_dt = 0.0
    free_mem_active_dt = 0.0

    snapshots = []

    nodes_groups = dict(tuple(nodes_df.groupby("event_index", sort=True)))

    for event_index, event in events_df.iterrows():
        node_states = nodes_groups.get(event_index)
        if node_states is None:
            continue

        dt = event["dt_seconds"]
        if pd.isna(dt) or dt is None:
            continue
        dt = float(dt)
        if dt <= 0.0:
            continue

        total_time_s += dt

        # totals
        total_cpus = node_states["total_CPUs"].sum()
        total_gpus = node_states["total_GPUs"].sum()
        total_mem = node_states["total_memory"].sum()

        cpus_used = node_states["CPUs_in_use"].sum()
        gpus_used = node_states["GPUs_in_use"].sum()
        mem_used = node_states["memory_in_use"].sum()

        cpu_util = (cpus_used / total_cpus) if total_cpus > 0 else 0.0
        gpu_util = (gpus_used / total_gpus) if total_gpus > 0 else 0.0
        mem_util = (mem_used / total_mem) if total_mem > 0 else 0.0

        # cluster power and energy
        cluster_power_W = node_states["power_consumption"].sum()
        total_energy_J += cluster_power_W * dt

        # active nodes (for consolidation)
        # if sleep model: power==0 means off
        active_mask = node_states["power_consumption"] > 0
        active_nodes = int(active_mask.sum())
        peak_active_nodes = max(peak_active_nodes, active_nodes)

        active_nodes_dt += active_nodes * dt
        node_hours_active += (active_nodes * dt) / 3600.0

        # fragmentation on active nodes: mean free resources per active node
        if active_nodes > 0:
            free_cpus = (node_states.loc[active_mask, "total_CPUs"] - node_states.loc[active_mask, "CPUs_in_use"]).sum()
            free_gpus = (node_states.loc[active_mask, "total_GPUs"] - node_states.loc[active_mask, "GPUs_in_use"]).sum()
            free_mem = (node_states.loc[active_mask, "total_memory"] - node_states.loc[active_mask, "memory_in_use"]).sum()

            free_cpu_active_dt += (free_cpus / active_nodes) * dt
            free_gpu_active_dt += (free_gpus / active_nodes) * dt
            free_mem_active_dt += (free_mem / active_nodes) * dt

        # weighted means
        cpu_util_dt += cpu_util * dt
        gpu_util_dt += gpu_util * dt
        mem_util_dt += mem_util * dt

        snapshots.append({
            "event_index": event_index,
            "time": event["time"],
            "dt_seconds": dt,
            "cluster_power_W": cluster_power_W,
            "cpu_util": cpu_util,
            "gpu_util": gpu_util,
            "mem_util": mem_util,
            "active_nodes": active_nodes,
            "cumulative_energy_J": total_energy_J,
        })

    snapshots_df = pd.DataFrame(snapshots)
    if snapshots_df.empty:
        # return empty but consistent
        return snapshots_df, {}

    # p95 utilisation (time-weighted using dt)
    cpu_p95 = weighted_quantile(snapshots_df["cpu_util"], [0.95], snapshots_df["dt_seconds"])[0]
    gpu_p95 = weighted_quantile(snapshots_df["gpu_util"], [0.95], snapshots_df["dt_seconds"])[0]
    mem_p95 = weighted_quantile(snapshots_df["mem_util"], [0.95], snapshots_df["dt_seconds"])[0]

    # schedule preservation from events_df
    starts = events_df[events_df["action"] == "start"].copy()
    total_jobs = len(starts)
    dropped = starts[starts["success"] == False]

    dropped_jobs_pct = (len(dropped) / total_jobs * 100.0) if total_jobs > 0 else 0.0

    # Dropped work using accounting duration
    total_work_s = starts["job_duration_seconds"].fillna(0.0).sum()
    dropped_work_s = dropped["job_duration_seconds"].fillna(0.0).sum()
    dropped_work_pct = (dropped_work_s / total_work_s * 100.0) if total_work_s > 0 else 0.0

    # GPU-seconds dropped (useful headline on Iridis X)
    total_gpu_s = starts["job_gpu_seconds"].fillna(0.0).sum()
    dropped_gpu_s = dropped["job_gpu_seconds"].fillna(0.0).sum()
    dropped_gpu_seconds_pct = (dropped_gpu_s / total_gpu_s * 100.0) if total_gpu_s > 0 else 0.0

    # CPU-seconds dropped 
    total_cpu_s = starts["job_cpu_seconds"].fillna(0.0).sum()
    dropped_cpu_s = dropped["job_cpu_seconds"].fillna(0.0).sum()
    dropped_cpu_seconds_pct = (dropped_cpu_s / total_cpu_s * 100.0) if total_cpu_s > 0 else 0.0

    # Memory-seconds dropped 
    total_memory_s = starts["job_mem_gb_seconds"].fillna(0.0).sum()
    dropped_memory_s = dropped["job_mem_gb_seconds"].fillna(0.0).sum()
    dropped_memory_seconds_pct = (dropped_memory_s / total_memory_s * 100.0) if total_memory_s > 0 else 0.0

    # Energy metrics
    total_energy_kwh = total_energy_J / 3_600_000.0
    avg_power_kw = (total_energy_J / total_time_s) / 1000.0 if total_time_s > 0 else 0.0

    baseline_energy_kwh = config.get("baseline_total_energy_kwh")

    if baseline_energy_kwh is None or float(baseline_energy_kwh) <= 0.0:
        energy_saving_pct = None
    else:
        baseline_energy_kwh = float(baseline_energy_kwh)
        energy_saving_pct = 100.0 * (baseline_energy_kwh - total_energy_kwh) / baseline_energy_kwh

    mean_active_nodes = active_nodes_dt / total_time_s if total_time_s > 0 else 0.0

    results = {
        # energy
        "total_energy_kwh": float(total_energy_kwh),
        "average_power_kw": float(avg_power_kw),
        "energy_saving_pct": energy_saving_pct,

        # schedule preservation
        "dropped_jobs_pct": float(dropped_jobs_pct),
        "dropped_work_pct": float(dropped_work_pct),
        "dropped_gpu_seconds_pct": float(dropped_gpu_seconds_pct),
        "dropped_cpu_seconds_pct": float(dropped_cpu_seconds_pct),
        "dropped_mem_gb_seconds_pct": float(dropped_memory_seconds_pct),

        # consolidation
        "mean_active_nodes": float(mean_active_nodes),
        "peak_active_nodes": int(peak_active_nodes),
        "node_hours_active": float(node_hours_active),

        # utilisation
        "cpu_util_mean": float(cpu_util_dt / total_time_s) if total_time_s > 0 else 0.0,
        "cpu_util_p95": float(cpu_p95),
        "gpu_util_mean": float(gpu_util_dt / total_time_s) if total_time_s > 0 else 0.0,
        "gpu_util_p95": float(gpu_p95),
        "mem_util_mean": float(mem_util_dt / total_time_s) if total_time_s > 0 else 0.0,
        "mem_util_p95": float(mem_p95),

        # fragmentation on active nodes (mean free per active node)
        "mean_free_cpus_active_nodes": float(free_cpu_active_dt / total_time_s) if total_time_s > 0 else 0.0,
        "mean_free_gpus_active_nodes": float(free_gpu_active_dt / total_time_s) if total_time_s > 0 else 0.0,
        "mean_free_memory_active_nodes": float(free_mem_active_dt / total_time_s) if total_time_s > 0 else 0.0,

        # run length
        "total_time_s": float(total_time_s),
    }

    return snapshots_df, results
