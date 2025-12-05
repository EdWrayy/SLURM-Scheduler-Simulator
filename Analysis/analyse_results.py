import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from pathlib import Path


def load_config(config_file="config.txt"):
    """Load configuration from config file"""
    config = {}
    # Try to find config file in multiple locations
    config_paths = [
        config_file,  # Current directory
        Path(__file__).parent / config_file,  # Same directory as this script
    ]

    config_path = None
    for path in config_paths:
        if Path(path).exists():
            config_path = path
            break

    if config_path is None:
        raise FileNotFoundError(f"Config file '{config_file}' not found in any of: {config_paths}")

    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config


# Load configuration
config = load_config("config.txt")
input_events_directory = Path(config.get('input_events_directory'))
input_nodes_directory = Path(config.get('input_nodes_directory'))

# Visualization options
show_last_job_marker = config.get('show_last_job_submission_line', 'true').lower() == 'true'
last_job_marker_date = config.get('last_job_submission_date', '').strip()

# Find all parquet files in the directories
nodes_files = sorted(input_nodes_directory.glob("*.parquet"))
events_files = sorted(input_events_directory.glob("*.parquet"))

if not nodes_files or not events_files:
    raise FileNotFoundError(f"No parquet files found in:\n  Events: {input_events_directory}\n  Nodes: {input_nodes_directory}")

print(f"Found {len(nodes_files)} node file(s) and {len(events_files)} event file(s)")

# Read and combine all monthly files
print(f"\nReading {len(nodes_files)} node file(s) from: {input_nodes_directory}")
nodes_dfs = []
for nf in nodes_files:
    print(f"  - {nf.name}")
    nodes_dfs.append(pd.read_parquet(nf))
nodes_df = pd.concat(nodes_dfs, ignore_index=True)
print(f"Combined nodes shape: {nodes_df.shape}")

print(f"\nReading {len(events_files)} event file(s) from: {input_events_directory}")
events_dfs = []
for ef in events_files:
    print(f"  - {ef.name}")
    events_dfs.append(pd.read_parquet(ef))
events_df = pd.concat(events_dfs, ignore_index=True)
print(f"Combined events shape: {events_df.shape}")


print(f"Node data shape: {nodes_df.shape}")
print(nodes_df.head())
print(f"Event data shape: {events_df.shape}")
print(events_df.head())

# Merge nodes with events to get time and active_jobs information
df = nodes_df.merge(
    events_df[['event_index', 'time', 'active_jobs']].drop_duplicates('event_index'),
    on='event_index',
    how='left'
)

print(f"Merged data shape: {df.shape}")
print(df.head())

# Ensure time is datetime
df["time"] = pd.to_datetime(df["time"])
print(f"\nTime range: {df['time'].min()} -> {df['time'].max()}")

# Determine the last job submission marker date
if show_last_job_marker:
    if last_job_marker_date:
        # Use custom date from config (format: DD/MM/YYYY  HH:MM:SS)
        max_start_date = pd.to_datetime(last_job_marker_date, format='%d/%m/%Y  %H:%M:%S')
        print(f"Last job submission (custom): {max_start_date}")
    else:
        raise ValueError("last_job_submission_date is enabled but an invalid date is provided")
else:
    max_start_date = None
    print("Last job submission marker disabled")

# Calculate average utilization metrics up to last job submission
if max_start_date is not None:
    # Filter data up to the last job submission
    pre_submission_df = df[df["time"] <= max_start_date]

    if len(pre_submission_df) > 0:
        print("\n" + "="*60)
        print("AVERAGE UTILIZATION (until last job submission)")
        print("="*60)
        avg_cpu_util = pre_submission_df["CPU_utilisation"].mean() * 100
        avg_gpu_util = pre_submission_df["GPU_utilisation"].mean() * 100
        avg_memory_util = pre_submission_df["memory_utilisation"].mean() * 100

        print(f"Average CPU utilization:    {avg_cpu_util:.2f}%")
        print(f"Average GPU utilization:    {avg_gpu_util:.2f}%")
        print(f"Average Memory utilization: {avg_memory_util:.2f}%")
        print("="*60 + "\n")
    else:
        print("\nWarning: No data found before last job submission date\n")
else:
    print("\nNote: Average utilization not calculated (last job submission marker disabled)\n")

# ---------------------------
# 1. Cluster level over time
# ---------------------------
cluster = (
    df
    .groupby(["event_index", "time"], as_index=False)
    .agg({
        "CPUs_in_use": "sum",
        "GPUs_in_use": "sum",
        "memory_in_use": "sum",
        "total_CPUs": "sum",
        "total_GPUs": "sum",
        "total_memory": "sum",
        "CPU_utilisation": "mean",
        "GPU_utilisation": "mean",
        "memory_utilisation": "mean",
        "active_jobs": "first",
    })
    .sort_values("time")
)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: CPU and GPU usage over time
ax1 = axes[0, 0]
ax1.plot(cluster["time"], cluster["CPUs_in_use"], label="CPUs in use", linewidth=1.5)
ax1.plot(cluster["time"], cluster["GPUs_in_use"], label="GPUs in use", linewidth=1.5)
if show_last_job_marker and max_start_date is not None:
    ax1.axvline(max_start_date, color='red', linestyle='--', linewidth=2, label='Last job start', alpha=0.7)
ax1.set_xlabel("Time")
ax1.set_ylabel("CPUs / GPUs in use")
ax1.set_title("Cluster CPU and GPU usage over time")
ax1.grid(True, alpha=0.3)

ax1_sec = ax1.twinx()
ax1_sec.plot(cluster["time"], cluster["memory_in_use"], label="Memory in use (MB)", linewidth=1.5)
ax1_sec.set_ylabel("Memory in use (MB)")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_sec.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# Plot 2: utilisation percentages
axes[0, 1].plot(cluster["time"], cluster["CPU_utilisation"] * 100, label="CPU", linewidth=1.0)
axes[0, 1].plot(cluster["time"], cluster["GPU_utilisation"] * 100, label="GPU", linewidth=1.0)
axes[0, 1].plot(cluster["time"], cluster["memory_utilisation"] * 100, label="Memory", linewidth=1.0)
if show_last_job_marker and max_start_date is not None:
    axes[0, 1].axvline(max_start_date, color='red', linestyle='--', linewidth=2, label='Last job start', alpha=0.7)
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Utilisation (%)")
axes[0, 1].set_ylim(0, 105)
axes[0, 1].set_title("Cluster utilisation over time")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Plot 3: active jobs
axes[1, 0].plot(cluster["time"], cluster["active_jobs"], linewidth=1.5)
axes[1, 0].fill_between(cluster["time"], cluster["active_jobs"], alpha=0.3)
if show_last_job_marker and max_start_date is not None:
    axes[1, 0].axvline(max_start_date, color='red', linestyle='--', linewidth=2, label='Last job start', alpha=0.7)
axes[1, 0].set_xlabel("Time")
axes[1, 0].set_ylabel("Active jobs")
axes[1, 0].set_title("Active jobs over time")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Plot 4: usage vs capacity
ax4 = axes[1, 1]
ax4.plot(cluster["time"], cluster["CPUs_in_use"], label="CPUs in use", linewidth=1.0)
ax4.plot(cluster["time"], cluster["total_CPUs"], "--", label="Total CPUs", linewidth=1.0, alpha=0.7)
ax4.plot(cluster["time"], cluster["GPUs_in_use"], label="GPUs in use", linewidth=1.0)
ax4.plot(cluster["time"], cluster["total_GPUs"], "--", label="Total GPUs", linewidth=1.0, alpha=0.7)
if show_last_job_marker and max_start_date is not None:
    ax4.axvline(max_start_date, color='red', linestyle='--', linewidth=2, label='Last job start', alpha=0.7)
ax4.set_xlabel("Time")
ax4.set_ylabel("CPUs / GPUs")
ax4.set_title("CPU and GPU usage vs capacity")
ax4.grid(True, alpha=0.3)

ax4_sec = ax4.twinx()
ax4_sec.plot(cluster["time"], cluster["memory_in_use"], label="Memory in use", linewidth=1.0)
ax4_sec.plot(cluster["time"], cluster["total_memory"], "--", label="Total memory", linewidth=1.0, alpha=0.7)
ax4_sec.set_ylabel("Memory (MB)")

lines4, labels4 = ax4.get_legend_handles_labels()
lines4b, labels4b = ax4_sec.get_legend_handles_labels()
ax4.legend(lines4 + lines4b, labels4 + labels4b, loc="upper left")

plt.tight_layout()
plt.show()

