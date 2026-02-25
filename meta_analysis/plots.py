from pathlib import Path
import matplotlib.pyplot as plt


def plot_pareto(df, plots_dir, pareto_x, pareto_y):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_df = df[["name", pareto_x, pareto_y]].dropna(subset=[pareto_x, pareto_y]).copy()

    fig, ax = plt.subplots()
    ax.scatter(plot_df[pareto_x], plot_df[pareto_y])

    ax.set_title("Pareto Trade-Off")
    ax.set_xlabel(pareto_x)
    ax.set_ylabel(pareto_y)

    for _, r in plot_df.iterrows():
        ax.annotate(r["name"], (r[pareto_x], r[pareto_y]), fontsize=7)

    fig.tight_layout()

    out_path = plots_dir / "pareto_tradeoff.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path