import pandas as pd

from .slurm_simulator import SlurmSimulation
from .nodes_config import resolve_node_configs
from .strategy_factory import get_strategy_instance
from .node_topology import create_nodes, print_simulation_configuration
from .output_storage import setup_output_paths, save_monthly_data
from .debug_info import write_failed_placement_debug_file
from .event_processing import (
    build_job_event_from_row,
    compute_job_work_metadata,
    execute_event,
    append_records,
)


def run_simulation(config):
    node_configs = resolve_node_configs(config)
    nodes, node_type_by_name, island_groups = create_nodes(config, node_configs)
    print_simulation_configuration(config, node_configs, nodes, node_type_by_name)

    node_selection = get_strategy_instance(config["node_selection_strategy"], "node_selection_strategy")
    resource_distribution = get_strategy_instance(config["resource_distribution_strategy"], "resource_distribution_strategy")

    (
        output_events_directory,
        output_nodes_directory,
        output_debug_directory,
        output_events_filename,
        output_nodes_filename,
    ) = setup_output_paths(config)

    slurm_sim = SlurmSimulation(
        island_groups,
        node_selection,
        resource_distribution,
    )
    debug_outputs_enabled = bool(config.get("enable_debug_output", True))

    events_df = pd.read_parquet(config.get("input_events"))
    events_df["time"] = pd.to_datetime(events_df["time"])
    
    # Sort event order by: event time. action priority (finish before start at identical timestamps), stable tie-break columns when present.
    action_priority = {"finish": 0, "start": 1}
    events_df["action_priority"] = events_df["action"].map(action_priority).fillna(2).astype("int8")

    sort_columns = ["time", "action_priority"]
    tie_break_candidates = [
        "job_id",
        "start_time",
        "end_time",
        "nodes_required",
        "CPUs_required",
        "GPUs_required",
        "memory_required",
        "real_node_selection",
        "allowed_node_types",
    ]
    sort_columns.extend(col for col in tie_break_candidates if col in events_df.columns)

    events_df = events_df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    events_df = events_df.drop(columns=["action_priority"])
    events_df["year_month"] = events_df["time"].dt.to_period("M")

    event_records = []
    node_records = []
    months_processed = []
    current_month = None
    prev_time = None

    print(f"Starting simulation with {len(events_df):,} events...")

    for i, row in enumerate(events_df.itertuples(index=False), start=0):
        _, event = build_job_event_from_row(row)

        event_month = str(row.year_month)
        if current_month is None:
            current_month = event_month
            print(f"\nProcessing month: {current_month}")
        elif current_month != event_month:
            save_monthly_data(
                current_month,
                event_records,
                node_records,
                output_events_directory,
                output_nodes_directory,
                output_events_filename,
                output_nodes_filename,
                months_processed,
            )
            event_records = []
            node_records = []
            current_month = event_month
            print(f"\nProcessing month: {current_month}")

        if prev_time is None:
            dt_seconds = None
        else:
            dt_seconds = (event.time - prev_time).total_seconds()
        prev_time = event.time

        start_ts, end_ts, job_duration_seconds, job_cpu_seconds, job_gpu_seconds, job_mem_gb_seconds = compute_job_work_metadata(row)

        event_success, failure_reason = execute_event(slurm_sim, event)
        if debug_outputs_enabled and event.action == "start" and not event_success:
            write_failed_placement_debug_file(
                output_debug_directory,
                i,
                event,
                failure_reason,
                slurm_sim,
            )

        state = slurm_sim.get_current_state()

        append_records(
            event_records,
            node_records,
            i,
            event,
            state,
            dt_seconds,
            event_success,
            failure_reason,
            start_ts,
            end_ts,
            job_duration_seconds,
            job_cpu_seconds,
            job_gpu_seconds,
            job_mem_gb_seconds,
        )

    if event_records:
        save_monthly_data(
            current_month,
            event_records,
            node_records,
            output_events_directory,
            output_nodes_directory,
            output_events_filename,
            output_nodes_filename,
            months_processed,
        )

    print(f"\n\nSaved output for {len(months_processed)} months: {', '.join(months_processed)}")
