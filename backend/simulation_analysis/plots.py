import pandas as pd
import matplotlib.pyplot as plt


def make_plots(snapshots_df, config, events_df):
    df = snapshots_df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values("time")

    plots = {}

    # Power over time
    fig, ax = plt.subplots()
    ax.plot(df["time"], df["cluster_power_W"])
    ax.set_title("Cluster power over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W)")
    fig.tight_layout()
    plots["power_over_time"] = fig

    # Active nodes over time
    fig, ax = plt.subplots()
    ax.plot(df["time"], df["active_nodes"])
    ax.set_title("Active nodes over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Active nodes")
    fig.tight_layout()
    plots["active_nodes_over_time"] = fig

    
    # Dropped job composition (by limiting resource)
    starts = events_df[events_df["action"] == "start"]
    dropped = starts[starts["success"] == False].copy()

    if not dropped.empty:
        dropped["limiting_resources"] = (
            dropped["limiting_resources"]
            .fillna("unknown")
            .astype(str)
        )

        breakdown = (
            dropped.groupby("limiting_resources")
            .size()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots()
        ax.bar(breakdown.index, breakdown.values)

        ax.set_title("Dropped jobs by limiting resource")
        ax.set_xlabel("Limiting resource")
        ax.set_ylabel("Number of dropped jobs")

        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()

        plots["dropped_job_composition"] = fig

    return plots
