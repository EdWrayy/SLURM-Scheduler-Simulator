from pathlib import Path
from .input import  load_reports, build_dataframe, parse_config_lists
from .metrics import make_rankings
from .plots import plot_pareto
from .report import save_rankings_json
from common.config import load_config


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

    pareto_path = plot_pareto(df, output_directory / "plots", pareto_x, pareto_y)

    print("Saved rankings:", rankings_path)
    print("Saved pareto plot:", pareto_path)


if __name__ == "__main__":
    main()
