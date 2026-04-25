from pathlib import Path
from .input import  load_reports, build_dataframe, parse_config_lists
from .metrics import make_rankings
from .plots import plot_pareto, plot_scatter, plot_comparison_table
from .report import save_rankings_json
from backend.common.config import load_config


def main():
    config = load_config(Path(__file__).with_name("config.json"))
    include_keys, higher_is_better_set, pareto_x, pareto_y = parse_config_lists(config)

    reports = load_reports(config["input_directory"])
    df = build_dataframe(reports, include_keys)

    rankings = make_rankings(df, include_keys, higher_is_better_set)

    output_directory = Path(config["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    rankings_path = output_directory / "rankings.json"
    save_rankings_json(rankings, rankings_path)

    plots_dir = output_directory / "plots"

    pareto_path = plot_pareto(df, plots_dir, pareto_x, pareto_y)

    free_resources_path = plot_scatter(
        df, plots_dir,
        x_key="mean_free_cpus_active_nodes",
        y_key="mean_free_gpus_active_nodes",
        title="Mean Free GPUs vs Mean Free CPUs (Active Nodes)",
        filename="mean_free_gpus_vs_cpus.png",
    )

    active_nodes_path = plot_scatter(
        df, plots_dir,
        x_key="dropped_work_pct",
        y_key="mean_active_nodes",
        title="Mean Active Nodes vs Dropped Work %",
        filename="active_nodes_vs_dropped_work.png",
    )

    table_path = plot_comparison_table(df, plots_dir)

    print("Saved rankings:", rankings_path)
    print("Saved energy savings vs dropped work plot:", pareto_path)
    print("Saved mean free GPUs vs CPUs plot:", free_resources_path)
    print("Saved active nodes vs dropped work plot:", active_nodes_path)
    print("Saved comparison table:", table_path)


if __name__ == "__main__":
    main()
