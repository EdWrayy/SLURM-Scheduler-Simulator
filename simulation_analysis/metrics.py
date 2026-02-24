import numpy as np
import pandas as pd



def calculate_all_metrics(events_df, nodes_df, config):
    """
    EVENTS TABLE:

    event_index (int64) 
    time (datetime64[ns])
    action (object)
    job_id (object)
    active_jobs (int64)
    dt_seconds (float64)
    success (bool)
    failure_reason (object)
    limiting_resources (object)
    job_start_time (datetime64[ns])
    job_end_time (datetime64[ns])
    job_duration_seconds (float64)
    job_cpu_seconds (float64)
    job_gpu_seconds (float64)
    job_mem_gb_seconds (float64)

    
    NODES TABLE:
    event_index (int64) 
    node_name (object)
    CPUs_in_use (int64)
    GPUs_in_use (int64)
    memory_in_use (int64)
    total_CPUs (int64)
    total_GPUs (int64)
    total_memory (int64)
    CPU_utilisation (float64)
    GPU_utilisation (float64)
    memory_utilisation (float64)
    power_consumption (float64)



    METRICS TO COMPUTE:

    Total Energy (kWh)
    Average Power (time weighted) (kW)
    Energy Saving vs Baseline (%)

    Dropped Jobs (%)
    Dropped Work (%)

    Mean Active Nodes
    Peak Active Nodes
    Node-Hours Active
    
    CPU Utilisation (mean and p95)
    GPU Utilisation (mean and p95)
    Memory Utilisation (mean and p95)
    
    Mean free CPUs on active nodes
    Mean free GPUs on active nodes
    Mean free memory on active nodes
    
    Power over time
    Active Nodes Over Time
    Dropped Work Composition

    snapshots_df - to create graphs of metrics over time
    """
    
    nodes_df["is_active"] = nodes_df["power_consumption"] > 0

    # Group together node records with the same event index
    node_agg = nodes_df.groupby(level=0, sort=True).agg(
        total_cpus=("total_CPUs", "sum"),
        total_gpus=("total_GPUs", "sum"),
        total_mem=("total_memory", "sum"),
        cpus_used=("CPUs_in_use", "sum"),
        gpus_used=("GPUs_in_use", "sum"),
        mem_used=("memory_in_use", "sum"),
        cluster_power_W=("power_consumption", "sum"),
        active_nodes=("is_active", "sum"),   # bool sum = count of True
    )   

    # Make a df purely for active nodes
    active = nodes_df.loc[nodes_df["power_consumption"] > 0, ["total_CPUs","CPUs_in_use","total_GPUs","GPUs_in_use","total_memory","memory_in_use"]].copy()
    active["free_cpus"] = active["total_CPUs"] - active["CPUs_in_use"]
    active["free_gpus"] = active["total_GPUs"] - active["GPUs_in_use"]
    active["free_mem"]  = active["total_memory"] - active["memory_in_use"]

    # Group together active nodes with the same event index
    active_agg = active.groupby(level=0, sort=True).agg(
        free_cpus_sum=("free_cpus", "sum"),
        free_gpus_sum=("free_gpus", "sum"),
        free_mem_sum=("free_mem", "sum"),
        active_nodes_check=("free_cpus", "size"),
    )

    # Join event index to corresponding node records
    df = events_df.join(node_agg, how="inner")

    # Add active node columns aswell
    df = df.join(active_agg[["free_cpus_sum","free_gpus_sum","free_mem_sum"]], how="left")
    
    # Fill nan values with 0
    df[["free_cpus_sum","free_gpus_sum","free_mem_sum"]] = df[["free_cpus_sum","free_gpus_sum","free_mem_sum"]].fillna(0.0)

    df = df.sort_index()

    # dt (use dt>0 for time-weighted metrics and quantiles)
    df["dt_seconds"] = df["dt_seconds"].astype(float)
    dt_all = df["dt_seconds"].to_numpy()
    mask_dt = dt_all > 0.0

    # Add utilisation metrics
    df["cpu_util"] = np.where(df["total_cpus"] > 0, df["cpus_used"] / df["total_cpus"], 0.0)
    df["gpu_util"] = np.where(df["total_gpus"] > 0, df["gpus_used"] / df["total_gpus"], 0.0)
    df["mem_util"] = np.where(df["total_mem"] > 0, df["mem_used"] / df["total_mem"], 0.0)

    # Calculate energy usage (dt=0 contributes 0 automatically)
    df["energy_J"] = df["cluster_power_W"].to_numpy() * dt_all
    df["cumulative_energy_J"] = df["energy_J"].cumsum()

     # Active-node fragmentation: mean free resources per active node (per event)
    df["free_cpus_per_active_node"] = np.where(df["active_nodes"] > 0, df["free_cpus_sum"] / df["active_nodes"], 0.0)
    df["free_gpus_per_active_node"] = np.where(df["active_nodes"] > 0, df["free_gpus_sum"] / df["active_nodes"], 0.0)
    df["free_mem_per_active_node"] = np.where(df["active_nodes"] > 0, df["free_mem_sum"] / df["active_nodes"], 0.0)


    """
    DF NOW:
    
    - Event Columns (event_index, time, action, job_id, active_jobs, dt_seconds, success, failure_reason, limiting_resources, job_start_time, job_end_time, job_duration_seconds, job_cpu_seconds, job_gpu_seconds, job_mem_gb_seconds)
    - Aggregated Node Data (total_cpus, total_gpus, total_mem, cpus_used, gpus_used, mem_used, cluster_power_W, active_nodes)
    - Active Node Data (free_cpus_sum, free_gpus_sum, free_mem_sum)
    - Per-event Column (cpu_util, gpu_util, mem_util, energy_J, cumulative_energy_J, free_cpus_per_active_node, free_gpus_per_active_node, free_mem_per_active_node)
    """
   
    # Time-weighted totals/means (use only dt>0 - remove first entry)
    dt = dt_all[mask_dt]
    if dt.size == 0:
        return pd.DataFrame(), {}

    # Total metrics
    total_time_s = float(dt.sum())
    total_energy_J = float(df.loc[mask_dt, "energy_J"].sum())
    total_energy_kwh = total_energy_J / 3_600_000.0

    # Time weighted averages
    avg_power_kw = (total_energy_J / total_time_s) / 1000.0 if total_time_s > 0 else 0.0
    cpu_util_mean = float((df.loc[mask_dt, "cpu_util"].to_numpy() * dt).sum() / total_time_s)
    gpu_util_mean = float((df.loc[mask_dt, "gpu_util"].to_numpy() * dt).sum() / total_time_s)
    mem_util_mean = float((df.loc[mask_dt, "mem_util"].to_numpy() * dt).sum() / total_time_s)

    # 95th percentile values
    cpu_p95 = calculate_p95(df.loc[mask_dt, "cpu_util"].to_numpy(), dt)
    gpu_p95 = calculate_p95(df.loc[mask_dt, "gpu_util"].to_numpy(), dt)
    mem_p95 = calculate_p95(df.loc[mask_dt, "mem_util"].to_numpy(), dt)

    # Activity values - time weighted
    mean_active_nodes = float((df.loc[mask_dt, "active_nodes"].to_numpy() * dt).sum() / total_time_s)
    peak_active_nodes = int(df["active_nodes"].max()) if len(df) else 0
    node_hours_active = float(((df.loc[mask_dt, "active_nodes"].to_numpy() * dt) / 3600.0).sum())

    # Free space averages - time waeighted
    mean_free_cpus_active_nodes = float((df.loc[mask_dt, "free_cpus_per_active_node"].to_numpy() * dt).sum() / total_time_s)
    mean_free_gpus_active_nodes = float((df.loc[mask_dt, "free_gpus_per_active_node"].to_numpy() * dt).sum() / total_time_s)
    mean_free_memory_active_nodes = float((df.loc[mask_dt, "free_mem_per_active_node"].to_numpy() * dt).sum() / total_time_s)

    # Dropped job/work metrics (events_df is indexed, so reset for column filters)
    events_for_jobs = events_df.reset_index()
    starts = events_for_jobs[events_for_jobs["action"] == "start"].copy()
    total_jobs = len(starts)
    dropped = starts[starts["success"] == False]
    dropped_jobs = int(len(dropped))

    dropped_jobs_pct = (len(dropped) / total_jobs * 100.0) if total_jobs > 0 else 0.0

    total_work_s = starts["job_duration_seconds"].fillna(0.0).sum()
    dropped_work_s = dropped["job_duration_seconds"].fillna(0.0).sum()
    dropped_work_pct = (dropped_work_s / total_work_s * 100.0) if total_work_s > 0 else 0.0

    total_gpu_s = starts["job_gpu_seconds"].fillna(0.0).sum()
    dropped_gpu_s = dropped["job_gpu_seconds"].fillna(0.0).sum()
    dropped_gpu_seconds_pct = (dropped_gpu_s / total_gpu_s * 100.0) if total_gpu_s > 0 else 0.0

    total_cpu_s = starts["job_cpu_seconds"].fillna(0.0).sum()
    dropped_cpu_s = dropped["job_cpu_seconds"].fillna(0.0).sum()
    dropped_cpu_seconds_pct = (dropped_cpu_s / total_cpu_s * 100.0) if total_cpu_s > 0 else 0.0

    total_memory_s = starts["job_mem_gb_seconds"].fillna(0.0).sum()
    dropped_memory_s = dropped["job_mem_gb_seconds"].fillna(0.0).sum()
    dropped_memory_seconds_pct = (dropped_memory_s / total_memory_s * 100.0) if total_memory_s > 0 else 0.0

    # Energy saving vs baseline
    baseline_energy_kwh = config.get("baseline_total_energy_kwh")
    if baseline_energy_kwh is None or float(baseline_energy_kwh) <= 0.0:
        energy_saving_pct = None
    else:
        baseline_energy_kwh = float(baseline_energy_kwh)
        energy_saving_pct = 100.0 * (baseline_energy_kwh - total_energy_kwh) / baseline_energy_kwh

    # snapshots_df for plotting
    snapshots_df = df.reset_index()[[
        "event_index", "time", "dt_seconds",
        "cluster_power_W", "cpu_util", "gpu_util", "mem_util",
        "active_nodes", "cumulative_energy_J"
    ]].copy()

    results = {
        # energy
        "total_energy_kwh": float(total_energy_kwh),
        "average_power_kw": float(avg_power_kw),
        "energy_saving_pct": energy_saving_pct,

        # schedule preservation
        "total_jobs": total_jobs,
        "dropped_jobs": dropped_jobs,
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
        "cpu_util_mean": float(cpu_util_mean),
        "cpu_util_p95": float(cpu_p95),
        "gpu_util_mean": float(gpu_util_mean),
        "gpu_util_p95": float(gpu_p95),
        "mem_util_mean": float(mem_util_mean),
        "mem_util_p95": float(mem_p95),

        # fragmentation on active nodes
        "mean_free_cpus_active_nodes": float(mean_free_cpus_active_nodes),
        "mean_free_gpus_active_nodes": float(mean_free_gpus_active_nodes),
        "mean_free_memory_active_nodes": float(mean_free_memory_active_nodes),

        # run length
        "total_time_s": float(total_time_s),
    }

    return snapshots_df, results

def calculate_p95(metric_values, time_durations):
    """
    Time-weighted 95th percentile.
    """
    # Convert inputs to numpy arrays
    metric_values = np.asarray(metric_values, dtype=float)
    time_durations = np.asarray(time_durations, dtype=float)

    # Keep only valid samples that represent real time
    valid_mask = (
        np.isfinite(metric_values)
        & np.isfinite(time_durations)
        & (time_durations > 0)
    )

    metric_values = metric_values[valid_mask]
    time_durations = time_durations[valid_mask]

    if metric_values.size == 0:
        return float("nan")

    # Sort metric values from lowest to highest
    sort_order = np.argsort(metric_values)
    sorted_values = metric_values[sort_order]
    sorted_durations = time_durations[sort_order]

    # Compute cumulative time spent at or below each value
    cumulative_time = np.cumsum(sorted_durations)

    # Total simulated runtime
    total_time = cumulative_time[-1]

    # Find the time location corresponding to the 95th percentile
    percentile_time_threshold = 0.95 * total_time

    # Find first index where cumulative time exceeds threshold
    percentile_index = np.searchsorted(
        cumulative_time,
        percentile_time_threshold
    )

    return float(sorted_values[percentile_index])