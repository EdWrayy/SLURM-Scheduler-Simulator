import pandas as pd
import matplotlib.pyplot as plt

def make_plots(snapshots_df, config, events_df):
    df = snapshots_df.copy()
    show_last_job_submission = config['show_last_job_submission_line']

    # ---- infer last job START time if requested ----
    last_start_time = None
    if show_last_job_submission and events_df is not None:
        if "action" in events_df.columns and "time" in events_df.columns:
            starts = events_df.loc[events_df["action"] == "start", "time"]
            if not starts.empty:
                last_start_time = pd.to_datetime(starts, errors="coerce").dropna().max()

    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    df = df.sort_values("time")

    plots = {}

    def add_last_start_line(ax):
        if show_last_job_submission and last_start_time is not None:
            ax.axvline(
                last_start_time,
                linestyle="--",
                linewidth=1.5,
                label="Last job start"
            )

    # Power vs time
    fig, ax = plt.subplots()
    ax.plot(df["time"], df["cluster_power_W"])
    add_last_start_line(ax)
    ax.set_title("Cluster power vs time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cluster power (W)")
    if show_last_job_submission and last_start_time is not None:
        ax.legend()
    fig.tight_layout()
    plots["power_vs_time"] = fig

    # Cumulative energy
    fig, ax = plt.subplots()
    ax.plot(df["time"], df["cumulative_energy_J"])
    add_last_start_line(ax)
    ax.set_title("Cumulative energy vs time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative energy (J)")
    if show_last_job_submission and last_start_time is not None:
        ax.legend()
    fig.tight_layout()
    plots["cumulative_energy_vs_time"] = fig

    # Utilisation
    fig, ax = plt.subplots()
    ax.plot(df["time"], df["cpu_util"], label="CPU")
    ax.plot(df["time"], df["gpu_util"], label="GPU")
    ax.plot(df["time"], df["mem_util"], label="Memory")
    add_last_start_line(ax)
    ax.set_title("Utilisation vs time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Utilisation (fraction)")
    ax.legend()
    fig.tight_layout()
    plots["utilisation_vs_time"] = fig


    # Active-nodes utilisation (excludes powered-down nodes)
    fig, ax = plt.subplots()
    ax.plot(df["time"], df["active_cpu_util"], label="CPU (active)")
    ax.plot(df["time"], df["active_gpu_util"], label="GPU (active)")
    ax.plot(df["time"], df["active_mem_util"], label="Memory (active)")
    add_last_start_line(ax)
    ax.set_title("Active-nodes utilisation vs time (excludes powered-down nodes)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Utilisation (fraction)")
    ax.legend()
    fig.tight_layout()
    plots["active_utilisation_vs_time"] = fig

    # Rolling mean power
    window = 50
    fig, ax = plt.subplots()
    roll = df["cluster_power_W"].rolling(
        window=window,
        min_periods=max(2, window // 10)
    ).mean()
    ax.plot(df["time"], df["cluster_power_W"], alpha=0.4, label="Raw")
    ax.plot(df["time"], roll, label=f"Rolling mean (window={window})")
    add_last_start_line(ax)
    ax.set_title("Cluster power (raw and rolling mean)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cluster power (W)")
    ax.legend()
    fig.tight_layout()
    plots["power_rolling_mean"] = fig

    # Scatter: power vs CPU util (no time axis → no line)
    fig, ax = plt.subplots()
    ax.scatter(df["cpu_util"], df["cluster_power_W"], s=8)
    ax.set_title("Power vs CPU utilisation")
    ax.set_xlabel("CPU utilisation")
    ax.set_ylabel("Cluster power (W)")
    fig.tight_layout()
    plots["power_vs_cpu_scatter"] = fig

    return plots
