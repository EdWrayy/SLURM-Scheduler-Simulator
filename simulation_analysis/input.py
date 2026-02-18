from pathlib import Path
import pandas as pd

def load_simulation_output(config):
    events_dir = Path(config["input_events_directory"])
    nodes_dir  = Path(config["input_nodes_directory"])

    return create_dataframes(events_dir, nodes_dir)

def create_dataframes(events_dir, nodes_dir):
    event_files = events_dir.glob("*.parquet")
    node_files = nodes_dir.glob("*.parquet")

    event_dfs = [pd.read_parquet(f) for f in event_files]
    events_df = pd.concat(event_dfs, ignore_index=True)
    events_df["time"] = pd.to_datetime(events_df["time"])
    events_df = events_df.sort_values("event_index").set_index("event_index", drop=False)

    node_dfs = [pd.read_parquet(f) for f in node_files]
    nodes_df = pd.concat(node_dfs, ignore_index=True)
    nodes_df = nodes_df.sort_values("event_index").set_index("event_index", drop=True)
    
    return events_df, nodes_df
