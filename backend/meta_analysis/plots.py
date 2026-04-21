from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_pareto(df, plots_dir, pareto_x, pareto_y):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_df = df[["name", pareto_x, pareto_y]].dropna(subset=[pareto_x, pareto_y]).copy().reset_index(drop=True)

    colors = cm.tab20(np.linspace(0, 1, len(plot_df)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in plot_df.iterrows():
        ax.scatter(r[pareto_x], r[pareto_y], color=colors[i], s=60, label=r["name"])

    ax.set_title("Energy Savings vs Dropped Work")
    ax.set_xlabel(pareto_x)
    ax.set_ylabel(pareto_y)

    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    fig.tight_layout()

    out_path = plots_dir / "energy_savings_vs_dropped_work.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path