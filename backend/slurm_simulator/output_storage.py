from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def setup_output_paths(config):
    output_events_directory = Path(config.get("output_events_directory", "output/events"))
    output_nodes_directory = Path(config.get("output_nodes_directory", "output/nodes"))
    output_debug_directory = Path(config.get("output_debug_directory", "output/debug"))

    output_events_directory.mkdir(parents=True, exist_ok=True)
    output_nodes_directory.mkdir(parents=True, exist_ok=True)
    output_debug_directory.mkdir(parents=True, exist_ok=True)

    output_events_filename = config.get("output_events", "simulation_log_events.parquet")
    output_nodes_filename = config.get("output_nodes", "simulation_log_nodes.parquet")

    return (
        output_events_directory,
        output_nodes_directory,
        output_debug_directory,
        output_events_filename,
        output_nodes_filename,
    )


def save_monthly_data(
    month_str,
    event_recs,
    node_recs,
    output_events_directory,
    output_nodes_directory,
    output_events_filename,
    output_nodes_filename,
    months_processed,
):
    if not event_recs:
        return

    print(f"\n  Saving data for {month_str}...")

    events_path = Path(output_events_filename)
    nodes_path = Path(output_nodes_filename)

    events_monthly = output_events_directory / f"{events_path.stem}_{month_str}{events_path.suffix}"
    nodes_monthly = output_nodes_directory / f"{nodes_path.stem}_{month_str}{nodes_path.suffix}"

    events_table = pa.Table.from_pandas(pd.DataFrame(event_recs))
    nodes_table = pa.Table.from_pandas(pd.DataFrame(node_recs))

    pq.write_table(events_table, events_monthly)
    pq.write_table(nodes_table, nodes_monthly)

    print(f"  Saved {len(event_recs):,} events and {len(node_recs):,} node records")
    months_processed.append(month_str)
