from pathlib import Path
import pandas as pd

def load_simulation_output(config):
    events_dir = Path(config["input_events_directory"])
    nodes_dir  = Path(config["input_nodes_directory"])

    return create_dataframes(events_dir, nodes_dir)

def create_dataframes(events_dir, nodes_dir):
    event_files = sorted(events_dir.glob("*.parquet"))
    node_files  = sorted(nodes_dir.glob("*.parquet"))

    events_df = pd.concat((pd.read_parquet(f) for f in event_files), ignore_index=True)
    events_df = events_df.set_index("event_index", drop=True)

    nodes_df = pd.concat((pd.read_parquet(f) for f in node_files), ignore_index=True)
    nodes_df = nodes_df.set_index("event_index", drop=True)
    
    return events_df, nodes_df