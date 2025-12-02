import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

df = pd.read_parquet("simulation_log.parquet")
print(df.columns)
print(df.head())
# Ensure time is datetime
df["time"] = pd.to_datetime(df["time"])

print(df["time"].min(), "->", df["time"].max())

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
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Utilisation (%)")
axes[0, 1].set_ylim(0, 105)
axes[0, 1].set_title("Cluster utilisation over time")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Plot 3: active jobs
axes[1, 0].plot(cluster["time"], cluster["active_jobs"], linewidth=1.5)
axes[1, 0].fill_between(cluster["time"], cluster["active_jobs"], alpha=0.3)
axes[1, 0].set_xlabel("Time")
axes[1, 0].set_ylabel("Active jobs")
axes[1, 0].set_title("Active jobs over time")
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: usage vs capacity
ax4 = axes[1, 1]
ax4.plot(cluster["time"], cluster["CPUs_in_use"], label="CPUs in use", linewidth=1.0)
ax4.plot(cluster["time"], cluster["total_CPUs"], "--", label="Total CPUs", linewidth=1.0, alpha=0.7)
ax4.plot(cluster["time"], cluster["GPUs_in_use"], label="GPUs in use", linewidth=1.0)
ax4.plot(cluster["time"], cluster["total_GPUs"], "--", label="Total GPUs", linewidth=1.0, alpha=0.7)
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

# --------------------------------------
# 2. Per node utilisation heatmap (scrollable)
# --------------------------------------

print("\n=== Available columns ===")
print(df.columns.tolist())

# Pick a node identifier column
node_col = None
for candidate in ["node_id", "node", "hostname", "node_name", "NodeList"]:
    if candidate in df.columns:
        node_col = candidate
        break

if node_col is None:
    print("\nWarning: no node identifier column found, cannot plot per node time series")
else:
    per_node = (
        df
        .groupby([node_col, "time"], as_index=False)
        .agg({"CPU_utilisation": "mean"})
    )

    cpu_pivot = (
        per_node
        .pivot(index="time", columns=node_col, values="CPU_utilisation")
        .sort_index()
    )

    # Optional: smooth in time
    cpu_pivot = cpu_pivot.resample("5T").mean()

    # Plotly heatmap (figure is tall so the notebook / page scrolls)
    fig = px.imshow(
        cpu_pivot.T.values * 100.0,
        x=cpu_pivot.index,
        y=cpu_pivot.columns,
        labels=dict(x="Time", y="Node", color="CPU utilisation (%)"),
        aspect="auto",
        origin="lower",
        zmin=0,
        zmax=100,
        color_continuous_scale="Viridis",
    )

    # Big height so you scroll to see all nodes
    fig.update_layout(
        title="Per node CPU utilisation over time",
        height=2000,
    )

    # Slightly cleaner x axis formatting
    fig.update_xaxes(
        tickformat="%H:%M",
        tickangle=0,
    )

    fig.show()
