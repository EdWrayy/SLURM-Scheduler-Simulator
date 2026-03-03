from pathlib import Path

from common.config import load_config
from .input import load_simulation_output


def _load_analysis_config(config=None, config_path=None):
    if config is not None:
        return config

    path = Path(config_path) if config_path else Path(__file__).with_name("config.txt")
    return load_config(path)


def inspect_failed_jobs(config=None, config_path=None):
    """
    Load simulation events and print full details for each failed job start.

    Returns:
        pandas.DataFrame: filtered dataframe containing failed job-start events.
    """
    config = _load_analysis_config(config=config, config_path=config_path)
    events_df, _ = load_simulation_output(config)

    required_columns = {"action", "success"}
    missing = required_columns.difference(events_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"events dataframe is missing required column(s): {missing_str}")

    failed_jobs_df = events_df[
        (events_df["action"] == "start") & (events_df["success"] == False)
    ].copy()

    print(f"Loaded {len(events_df):,} event(s)")
    print(f"Found {len(failed_jobs_df):,} failed job(s)")

    if failed_jobs_df.empty:
        print("No failed jobs were found.")
        return failed_jobs_df

    failed_jobs_df = failed_jobs_df.sort_index().reset_index()

    for i, row in failed_jobs_df.iterrows():
        print(f"\nFailed job #{i + 1}")
        print("-" * 60)
        for key, value in row.items():
            print(f"{key}: {value}")

    return failed_jobs_df


def main():
    inspect_failed_jobs()


if __name__ == "__main__":
    main()
