from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_pareto(df, plots_dir, pareto_x, pareto_y, exclude_names=None, filename="energy_savings_vs_dropped_work.png"):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_df = df[["name", pareto_x, pareto_y]].dropna(subset=[pareto_x, pareto_y]).copy()
    if exclude_names:
        plot_df = plot_df[~plot_df["name"].isin(exclude_names)]
    plot_df = plot_df.reset_index(drop=True)

    colors = cm.tab20(np.linspace(0, 1, len(plot_df)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in plot_df.iterrows():
        ax.scatter(r[pareto_x], r[pareto_y], color=colors[i], s=60, label=r["name"])

    ax.set_title("Energy Savings vs Dropped Work")
    ax.set_xlabel(pareto_x)
    ax.set_ylabel(pareto_y)

    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    fig.tight_layout()

    out_path = plots_dir / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_scatter(df, plots_dir, x_key, y_key, title, filename, exclude_names=None):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_df = df[["name", x_key, y_key]].dropna(subset=[x_key, y_key]).copy()
    if exclude_names:
        plot_df = plot_df[~plot_df["name"].isin(exclude_names)]
    plot_df = plot_df.reset_index(drop=True)

    colors = cm.tab20(np.linspace(0, 1, len(plot_df)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in plot_df.iterrows():
        ax.scatter(r[x_key], r[y_key], color=colors[i], s=60, label=r["name"])

    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)

    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    fig.tight_layout()

    out_path = plots_dir / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_comparison_table(df, plots_dir, baseline_name="Naive_Feasible", exclude_names=None, filename="comparison_table.png"):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        ("mean_free_gpus_active_nodes", "% Improved Free GPUs"),
        ("mean_free_cpus_active_nodes", "% Improved Free CPUs"),
        ("mean_active_nodes",           "% Reduced Active Nodes"),
        ("total_energy_kwh",            "% Reduced Total Energy"),
    ]
    metric_keys = [c[0] for c in columns]
    col_labels  = [c[1] for c in columns]

    needed = ["name"] + metric_keys
    plot_df = df[needed].dropna(subset=metric_keys).copy()

    baseline_rows = plot_df[plot_df["name"] == baseline_name]
    if baseline_rows.empty:
        raise ValueError(f"Baseline '{baseline_name}' not found in data")
    baseline = baseline_rows.iloc[0]

    plot_df = plot_df[plot_df["name"] != baseline_name].copy()
    if exclude_names:
        plot_df = plot_df[~plot_df["name"].isin(exclude_names)]
    plot_df = plot_df.reset_index(drop=True)

    cell_vals = []
    for _, row in plot_df.iterrows():
        row_vals = []
        for key in metric_keys:
            baseline_val = baseline[key]
            pct = (baseline_val - row[key]) / baseline_val * 100 if baseline_val != 0 else 0.0
            row_vals.append(pct)
        cell_vals.append(row_vals)

    row_labels = list(plot_df["name"])
    cell_text  = [[f"{v:+.2f}%" for v in row] for row in cell_vals]

    fig_h = max(3, 0.4 * len(row_labels) + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(len(col_labels))))

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0 or col_idx == -1:
            cell.set_facecolor("#d0d0d0")
            cell.set_text_props(fontweight="bold")
        else:
            val = cell_vals[row_idx - 1][col_idx]
            intensity = min(abs(val) / 20, 1.0)
            if val > 0:
                cell.set_facecolor((1 - intensity * 0.6, 1, 1 - intensity * 0.6))
            elif val < 0:
                cell.set_facecolor((1, 1 - intensity * 0.6, 1 - intensity * 0.6))

    ax.set_title(f"% Improvement vs {baseline_name}", fontsize=11, pad=12)
    fig.tight_layout()

    out_path = plots_dir / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path