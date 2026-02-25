from .slurm_simulator import SlurmSimulation, Node, DefaultResourceDistribution, CopyRealNodeSelection, FirstFitNodeSelection, ActiveOnlyPowerModel, LinearWithIdlePowerModel, LinearWithSleepPowerModel
from .power_constants import NODE_HARDWARE, CPU_POWER, GPU_POWER, RAM_POWER
from common.models import Job, JobEvent
import pandas as pd
from datetime import datetime
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import ast




def get_strategy_instance(strategy_name, strategy_type):
    if strategy_type == "node_selection_strategy":
        if strategy_name == "CopyRealNodeSelection":
            return CopyRealNodeSelection()
        elif strategy_name == "FirstFitNodeSelection":
            return FirstFitNodeSelection()
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


def create_nodes(config):
    nodes = []
    node_type_by_name = {}

    power_model = get_strategy_instance(
        config['power_model'],
        'power_model'
    )

    for node_config in config["nodes"]:
        node_type = node_config[0]
        config_order = int(node_config[1])
        node_range = node_config[2]
        num_nodes = int(node_config[3])
        cpus = int(node_config[4])
        gpus = int(node_config[5])
        gpu_type = node_config[6]
        memory = int(node_config[7])

        power_data = NODE_HARDWARE[node_type]
        cpu_p = CPU_POWER[power_data["cpu"]]
        cpu_idle_power = cpu_p["idle_W"]
        cpu_max_power  = cpu_p["max_W"]

        gpu_type_hw = power_data["gpu"]  # can be None
        if gpu_type_hw is None:
            gpu_idle_power = 0.0
            gpu_max_power  = 0.0
        else:
            gpu_p = GPU_POWER[gpu_type_hw]
            gpu_idle_power = gpu_p["idle_W"]
            gpu_max_power  = gpu_p["max_W"]

        ram_type_hw = power_data["ram"]
        ram_p = RAM_POWER[ram_type_hw]
        ram_idle_power = ram_p["idle_W_per_GB"]
        ram_max_power  = ram_p["max_W_per_GB"]
                
        for i in range(1, num_nodes+1):
            node_name = get_real_node_name(node_type, i)
            node_type_by_name[node_name] = node_type  

            node = Node(
                name=node_name,
                id = i, #Used for sorting nodes for selection
                list_position=config_order,
                total_CPUs= cpus,
                total_GPUs=gpus,
                total_memory=memory,
                CPU_Max_Power = cpu_max_power, 
                GPU_Max_Power = gpu_max_power, 
                RAM_Max_Power = ram_max_power, 
                CPU_Idle_Power = cpu_idle_power, 
                GPU_Idle_Power = gpu_idle_power, 
                RAM_Idle_Power = ram_idle_power,
                power_model = power_model
            )
            nodes.append(node)

    return nodes, node_type_by_name


def get_real_node_name(node_type, id): 
    """
    Maps a node's type and id number to its name in the real logs for Iridis X.
    Examples:
      ruby, 1      -> ruby001
      swarma, 1    -> swarma1001
      swarmh, 12   -> swarmh1012
      other, 1     -> other01
    """

    if node_type == "ruby":
        # pad to 3 digits
        return f"{node_type}{id:03d}"

    elif node_type in {"swarmh", "swarma"}:
        return f"{node_type}{1000 + id:04d}"
    else:
        # pad to 2 digits
        return f"{node_type}{id:02d}"

def print_simulation_configuration(config, nodes, node_type_by_name):
    print("\n" + "="*70)
    print("SLURM SIMULATION CONFIGURATION")
    print("="*70)

    print("\nStrategies:")
    print(f"  Node selection: {config['node_selection_strategy']}")
    print(f"  Resource distribution: {config['resource_distribution_strategy']}")
    print(f"  Power model: {config['power_model']}")

    print("\nConfigured node groups:")
    for n in config["nodes"]:
        print(
            f"  type={n[0]:10s} "
            f"order={n[1]:2s} "
            f"range={n[2]:15s} "
            f"count={n[3]:3s} "
            f"CPUs={n[4]:4s} "
            f"GPUs={n[5]:2s} "
            f"mem={n[7]}"
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
        real_node_selection=ast.literal_eval(row.real_node_selection) if pd.notna(row.real_node_selection) else None
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
    job_mem_gb_seconds = (float(row.memory_required) / 1024.0) * job_duration_seconds

    return start_ts, end_ts, job_duration_seconds, job_cpu_seconds, job_gpu_seconds, job_mem_gb_seconds


def execute_event(slurm_sim, event):
    event_success = None
    failure_reason = None
    limiting_resources = None

    if event.action == 'start':
        event_success, info = slurm_sim.place_job(event.job)
        if not event_success:
            failure_reason = info.get("reason")
            lr = info.get("limiting_resources", [])
            if lr is None:
                limiting_resources = None
            elif isinstance(lr, str):
                limiting_resources = lr
            else:
                limiting_resources = ",".join(map(str, lr))

    elif event.action == 'finish':
        release_success = slurm_sim.release_job(event.job.id)
        event_success = release_success
        if not release_success:
            failure_reason = "release_failed"
            limiting_resources = "release_failed"
    else:
        raise ValueError("Unknown event action")

    return event_success, failure_reason, limiting_resources


def append_records(event_records, node_records, i, event, state, dt_seconds, event_success, failure_reason, limiting_resources, start_ts, end_ts, job_duration_seconds, job_cpu_seconds, job_gpu_seconds, job_mem_gb_seconds):
    event_records.append({
        'event_index': i,
        'time': event.time,
        'action': event.action,
        'job_id': event.job.id,
        'active_jobs': state['active_jobs'],
        'dt_seconds': dt_seconds,

        'success': event_success,
        'failure_reason': failure_reason,
        'limiting_resources': limiting_resources,

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
    nodes, node_type_by_name = create_nodes(config)
    print_simulation_configuration(config, nodes, node_type_by_name)

    node_selection = get_strategy_instance(config['node_selection_strategy'], 'node_selection_strategy')
    resource_distribution = get_strategy_instance(config['resource_distribution_strategy'], 'resource_distribution_strategy')

    output_events_directory, output_nodes_directory, output_events_filename, output_nodes_filename = setup_output_paths(config)

    slurm_sim = SlurmSimulation(
        nodes,
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

        event_success, failure_reason, limiting_resources = execute_event(slurm_sim, event)

        state = slurm_sim.get_current_state()

        append_records(event_records, node_records, i, event, state, dt_seconds, event_success, failure_reason, limiting_resources, start_ts, end_ts, job_duration_seconds, job_cpu_seconds, job_gpu_seconds, job_mem_gb_seconds)

    if event_records:
        save_monthly_data(current_month, event_records, node_records, output_events_directory, output_nodes_directory, output_events_filename, output_nodes_filename, months_processed)

    print(f"\n\nSaved output for {len(months_processed)} months: {', '.join(months_processed)}")