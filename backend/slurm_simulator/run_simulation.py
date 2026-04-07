from .slurm_simulator import (
    SlurmSimulation,
    Node,
    DefaultResourceDistribution,
    CopyRealNodeSelection,
    NaiveFirstFit,
    ActiveOnlyPowerModel,
    LinearWithIdlePowerModel,
    LinearWithSleepPowerModel,
)
from .power_constants import load_power_constants
from .nodes_config import resolve_node_configs
from backend.common.models import Job, JobEvent
import pandas as pd
from datetime import datetime
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import ast
import re

BYTES_PER_MIB = 1024 ** 2
BYTES_PER_GIB = 1024 ** 3




def get_strategy_instance(strategy_name, strategy_type):
    if strategy_type == "node_selection_strategy":
        if strategy_name == "CopyRealNodeSelection":
            return CopyRealNodeSelection()
        elif strategy_name == "NaiveFirstFit":
            return NaiveFirstFit()
        else:
            raise ValueError("Unknown Selection Strategy")
        
    
    elif strategy_type == "resource_distribution_strategy":
        if strategy_name == "DefaultResourceDistribution":
            return DefaultResourceDistribution()
        else:
            raise ValueError("Unknown Distribution Strategy")
    
    elif strategy_type == "power_model":
        if strategy_name == "ActiveOnlyPowerModel":
            return ActiveOnlyPowerModel()
        elif strategy_name == "LinearWithIdlePowerModel":
            return LinearWithIdlePowerModel()
        elif strategy_name == "LinearWithSleepPowerModel":
            return LinearWithSleepPowerModel()
        else:
            raise ValueError("Unknown Power Model")
    
    else:
        raise ValueError("Unknown Strategy Type")


def create_nodes(config, node_configs):
    nodes = []
    node_type_by_name = {}
    seen_node_names = set()
    cpu_power, gpu_power, ram_power, cpu_idle_fraction, gpu_idle_fraction = load_power_constants(
        config.get("power_constants_config")
    )

    power_model_name = config['power_model']
    if power_model_name == 'ActiveOnlyPowerModel':
        power_model = ActiveOnlyPowerModel()
    elif power_model_name == 'LinearWithIdlePowerModel':
        power_model = LinearWithIdlePowerModel(cpu_idle_fraction, gpu_idle_fraction)
    elif power_model_name == 'LinearWithSleepPowerModel':
        power_model = LinearWithSleepPowerModel(cpu_idle_fraction, gpu_idle_fraction)
    else:
        raise ValueError(f"Unknown power model: {power_model_name}")

    for node_config in node_configs:
        node_type = node_config["node_type"]
        config_order = int(node_config["config_order"])
        node_range = node_config["node_range"]
        num_nodes = int(node_config["num_nodes"])
        cpus = int(node_config["cpus_per_node"])
        physical_cpus = int(node_config["physical_cpus_per_node"])
        cores_per_physical_cpu = int(node_config["cores_per_physical_cpu"])
        gpus = int(node_config["gpus_per_node"])
        gpu_type = node_config["gpu_type"]
        memory_mb = int(node_config["memory_mb"])
        memory = memory_mb * BYTES_PER_MIB
        hardware = node_config.get("hardware")
        if not isinstance(hardware, dict):
            raise ValueError(f"Node '{node_type}' is missing hardware details ")
        cpu_hw = hardware.get("cpu")
        gpu_hw = hardware.get("gpu")
        ram_hw = hardware.get("ram")

        if cpu_hw not in cpu_power:
            raise ValueError(f"Node '{node_type}' references unknown CPU power key '{cpu_hw}'")
        cpu_p = cpu_power[cpu_hw]
        cpu_max_power  = cpu_p["max_W"]

        if gpu_hw is None:
            gpu_max_power = 0.0
        else:
            if gpu_hw not in gpu_power:
                raise ValueError(f"Node '{node_type}' references unknown GPU power key '{gpu_hw}'")
            gpu_p = gpu_power[gpu_hw]
            gpu_max_power = gpu_p["max_W"]

        dimms_per_node = int(node_config["dimms_per_node"])
        dimm_size_GB = int(hardware["dimm_size_GB"])
        if ram_hw not in ram_power:
            raise ValueError(f"Node '{node_type}' references unknown RAM power key '{ram_hw}'")
        dimm_size_key = str(dimm_size_GB)
        if dimm_size_key not in ram_power[ram_hw]:
            raise ValueError(f"Node '{node_type}' references unknown DIMM size '{dimm_size_GB}GB' for '{ram_hw}'")
        ram_fixed_power = dimms_per_node * ram_power[ram_hw][dimm_size_key]["W_per_dimm"]
                
        node_names_and_ids = expand_node_names_from_range(node_type, node_range, num_nodes)
        for node_name, node_id in node_names_and_ids:
            if node_name in seen_node_names:
                raise ValueError(f"Duplicate node name generated: '{node_name}'")
            seen_node_names.add(node_name)
            node_type_by_name[node_name] = node_type  

            node = Node(
                name=node_name,
                node_type=node_type,
                id = node_id, #Used for sorting nodes for selection
                list_position=config_order,
                physical_CPUs=physical_cpus,
                cores_per_physical_CPU=cores_per_physical_cpu,
                total_CPUs= cpus,
                total_GPUs=gpus,
                total_memory=memory,
                CPU_Max_Power = cpu_max_power,
                GPU_Max_Power = gpu_max_power,
                RAM_Power = ram_fixed_power,
                power_model = power_model
            )
            nodes.append(node)

    node_by_name = {node.name: node for node in nodes}

    island_groups = {}
    for node_config in node_configs:
        node_type = node_config["node_type"]
        for island_str in node_config.get("network_islands", []):
            island_names = expand_island_to_node_names(island_str)
            group = [node_by_name[n] for n in island_names if n in node_by_name]
            if group:
                island_groups.setdefault(node_type, []).append(group)

    nodes_in_islands = {node for groups in island_groups.values() for group in groups for node in group}
    for node in nodes:
        if node not in nodes_in_islands:
            island_groups.setdefault(node.node_type, []).append([node])

    return nodes, node_type_by_name, island_groups


def expand_island_to_node_names(island_str):
    match = re.fullmatch(r'([a-zA-Z]+\d*)\[([^\]]+)\]', island_str)
    if match is None:
        return {island_str}
    prefix = match.group(1)
    range_part = match.group(2)
    names = set()
    for token in range_part.split(','):
        token = token.strip()
        if '-' in token:
            start_str, end_str = token.split('-')
            width = max(len(start_str), len(end_str))
            for i in range(int(start_str), int(end_str) + 1):
                names.add(f"{prefix}{str(i).zfill(width)}")
        else:
            names.add(f"{prefix}{token}")
    return names


def expand_node_names_from_range(node_type, node_range, num_nodes):
    pattern = rf"{re.escape(node_type)}\[(\d+)(?:-(\d+))?\]"
    match = re.fullmatch(pattern, str(node_range))
    if match is None:
        raise ValueError(
            f"Invalid node_range '{node_range}' for node_type '{node_type}'. "
            "Expected format like 'ruby[001-074]' or 'coral[01]'."
        )

    start_str = match.group(1)
    end_str = match.group(2) if match.group(2) is not None else start_str
    width = max(len(start_str), len(end_str))
    start = int(start_str)
    end = int(end_str)

    if end < start:
        raise ValueError(f"Invalid node_range '{node_range}': end index is less than start index")

    expected_num_nodes = end - start + 1
    if expected_num_nodes != num_nodes:
        raise ValueError(
            f"node_range '{node_range}' implies {expected_num_nodes} nodes, "
            f"but num_nodes is {num_nodes} for node_type '{node_type}'"
        )

    return [(f"{node_type}{idx:0{width}d}", idx) for idx in range(start, end + 1)]

def print_simulation_configuration(config, node_configs, nodes, node_type_by_name):
    print("\n" + "="*70)
    print("SLURM SIMULATION CONFIGURATION")
    print("="*70)

    print("\nStrategies:")
    print(f"  Node selection: {config['node_selection_strategy']}")
    print(f"  Resource distribution: {config['resource_distribution_strategy']}")
    print(f"  Power model: {config['power_model']}")

    print("\nConfigured node groups:")
    for n in node_configs:
        node_type = n["node_type"]
        order = n["config_order"]
        node_range = n["node_range"]
        count = n["num_nodes"]
        cpus = n["cpus_per_node"]
        physical_cpus = n["physical_cpus_per_node"]
        cores_per_physical_cpu = n["cores_per_physical_cpu"]
        gpus = n["gpus_per_node"]
        mem = n["memory_mb"]

        print(
            f"  type={str(node_type):10s} "
            f"order={str(order):2s} "
            f"range={str(node_range):15s} "
            f"count={str(count):3s} "
            f"pCPUs={str(physical_cpus):2s} "
            f"cores/CPU={str(cores_per_physical_cpu):2s} "
            f"CPUs={str(cpus):4s} "
            f"GPUs={str(gpus):2s} "
            f"mem_mb={mem}"
        )

    print("\nNodes instantiated in simulation:")
    print(f"  Total nodes: {len(nodes)}")

    names = sorted(n.name for n in nodes)

    # print grouped by prefix
    from collections import defaultdict
    groups = defaultdict(list)

    for n in nodes:
        groups[node_type_by_name.get(n.name, "unknown")].append(n.name)

    for node_type, group in groups.items():
        group = sorted(group)
        print(f"\n  {node_type} ({len(group)} nodes)")
        print("   ", ", ".join(group[:10]))
        if len(group) > 10:
            print("    ...")

    print("="*70 + "\n")

def setup_output_paths(config):
    output_events_directory = Path(config.get('output_events_directory', 'output/events'))
    output_nodes_directory = Path(config.get('output_nodes_directory', 'output/nodes'))

    output_events_directory.mkdir(parents=True, exist_ok=True)
    output_nodes_directory.mkdir(parents=True, exist_ok=True)

    output_events_filename = config.get('output_events', 'simulation_log_events.parquet')
    output_nodes_filename = config.get('output_nodes', 'simulation_log_nodes.parquet')

    return output_events_directory, output_nodes_directory, output_events_filename, output_nodes_filename


def save_monthly_data(month_str, event_recs, node_recs, output_events_directory, output_nodes_directory, output_events_filename, output_nodes_filename, months_processed):
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


def build_job_event_from_row(row):
    job = Job(
        id=row.job_id,
        nodes_required=row.nodes_required,
        CPUs_required=row.CPUs_required,
        GPUs_required=row.GPUs_required,
        memory_required=row.memory_required,
        start_time=row.start_time,
        end_time=row.end_time,
        real_node_selection=ast.literal_eval(row.real_node_selection) if pd.notna(row.real_node_selection) else None,
        allowed_node_types=ast.literal_eval(row.allowed_node_types) if pd.notna(row.allowed_node_types) else None

    )
    event = JobEvent(job, row.action, row.time)
    return job, event


def compute_job_work_metadata(row):
    if row.action != 'start':
        return None, None, None, None, None, None

    start_ts = pd.to_datetime(row.start_time)
    end_ts = pd.to_datetime(row.end_time)

    job_duration_seconds = (end_ts - start_ts).total_seconds()
    
    if pd.isna(job_duration_seconds) or job_duration_seconds < 0:
        return start_ts, end_ts, None, None, None, None

    job_cpu_seconds = float(row.CPUs_required) * job_duration_seconds
    job_gpu_seconds = float(row.GPUs_required) * job_duration_seconds
    job_mem_gb_seconds = (float(row.memory_required) / BYTES_PER_GIB) * job_duration_seconds

    return start_ts, end_ts, job_duration_seconds, job_cpu_seconds, job_gpu_seconds, job_mem_gb_seconds


def execute_event(slurm_sim, event):
    event_success = None
    failure_reason = None

    if event.action == 'start':
        event_success, info = slurm_sim.place_job(event.job)
        if not event_success:
            failure_reason = info.get("reason")

    elif event.action == 'finish':
        release_success = slurm_sim.release_job(event.job.id)
        event_success = release_success
        if not release_success:
            failure_reason = "release_failed"
    else:
        raise ValueError("Unknown event action")

    return event_success, failure_reason


def append_records(event_records, node_records, i, event, state, dt_seconds, event_success, failure_reason, start_ts, end_ts, job_duration_seconds, job_cpu_seconds, job_gpu_seconds, job_mem_gb_seconds):
    event_records.append({
        'event_index': i,
        'time': event.time,
        'action': event.action,
        'job_id': event.job.id,
        'active_jobs': state['active_jobs'],
        'dt_seconds': dt_seconds,

        'success': event_success,
        'failure_reason': failure_reason,

        'job_start_time': start_ts,
        'job_end_time': end_ts,
        'job_duration_seconds': job_duration_seconds,
        'job_cpu_seconds': job_cpu_seconds,
        'job_gpu_seconds': job_gpu_seconds,
        'job_mem_gb_seconds': job_mem_gb_seconds,
    })

    for n_state in state['nodes']:
        node_records.append({
            'event_index': i,
            'node_name': n_state['name'],
            'physical_CPUs': n_state['physical_CPUs'],
            'cores_per_physical_CPU': n_state['cores_per_physical_CPU'],
            'CPUs_in_use': n_state['CPUs_in_use'],
            'GPUs_in_use': n_state['GPUs_in_use'],
            'memory_in_use': n_state['memory_in_use'],
            'total_CPUs': n_state['total_CPUs'],
            'total_GPUs': n_state['total_GPUs'],
            'total_memory': n_state['total_memory'],
            'CPU_utilisation': n_state['CPUs_in_use'] / n_state['total_CPUs'] if n_state['total_CPUs'] > 0 else 0,
            'GPU_utilisation': n_state['GPUs_in_use'] / n_state['total_GPUs'] if n_state['total_GPUs'] > 0 else 0,
            'memory_utilisation': n_state['memory_in_use'] / n_state['total_memory'] if n_state['total_memory'] > 0 else 0,
            'power_consumption': n_state['power_consumption']
        })




def run_simulation(config):
    node_configs = resolve_node_configs(config)
    nodes, node_type_by_name, island_groups = create_nodes(config, node_configs)
    print_simulation_configuration(config, node_configs, nodes, node_type_by_name)

    node_selection = get_strategy_instance(config['node_selection_strategy'], 'node_selection_strategy')
    resource_distribution = get_strategy_instance(config['resource_distribution_strategy'], 'resource_distribution_strategy')

    output_events_directory, output_nodes_directory, output_events_filename, output_nodes_filename = setup_output_paths(config)

    slurm_sim = SlurmSimulation(
        nodes,
        island_groups,
        node_selection,
        resource_distribution,
    )

    events_df = pd.read_parquet(config.get('input_events'))
    events_df['time'] = pd.to_datetime(events_df['time'])
    events_df['year_month'] = events_df['time'].dt.to_period('M')

    event_records = []
    node_records = []
    months_processed = []
    current_month = None
    prev_time = None

    print(f"Starting simulation with {len(events_df):,} events...")

    for i, row in enumerate(events_df.itertuples(index=False), start=0):
        job, event = build_job_event_from_row(row)

        event_month = str(row.year_month)
        if current_month is None:
            current_month = event_month
            print(f"\nProcessing month: {current_month}")
        elif current_month != event_month:
            save_monthly_data(current_month, event_records, node_records, output_events_directory, output_nodes_directory, output_events_filename, output_nodes_filename, months_processed)
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

        state = slurm_sim.get_current_state()

        append_records(event_records, node_records, i, event, state, dt_seconds, event_success, failure_reason, start_ts, end_ts, job_duration_seconds, job_cpu_seconds, job_gpu_seconds, job_mem_gb_seconds)

    if event_records:
        save_monthly_data(current_month, event_records, node_records, output_events_directory, output_nodes_directory, output_events_filename, output_nodes_filename, months_processed)

    print(f"\n\nSaved output for {len(months_processed)} months: {', '.join(months_processed)}")
