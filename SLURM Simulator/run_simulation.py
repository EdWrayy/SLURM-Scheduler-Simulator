from slurm_simulator import SlurmSimulation, Node, Job, DefaultResourceDistribution, CopyRealNodeSelection
import pandas as pd
from datetime import datetime
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa

class JobEvent():
    def __init__(self, job, action, time):
        self.job = job
        self.time = time
        self.action = action



def load_config(config_file="config.txt"):
    """Load configuration from config file"""
    config = {}
    node_list = []
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if key == "node":
                    _, rhs = line.split("=", 1)
                    parts = [p.strip() for p in rhs.split(",")]
                    node_list.append(parts)
                elif key.startswith("partition"):
                    pass #ignore for now
                else:
                    config[key.strip()] = value.strip()
    
    config["nodes"] = node_list
    return config


def get_strategy_instance(strategy_name, strategy_type):
    if strategy_type == "node_selection_strategy":
        if strategy_name == "CopyRealNodeSelection":
            return CopyRealNodeSelection()
        else:
            raise ValueError("Unknown Selection Strategy")
        
    
    elif strategy_type == "resource_distribution_strategy":
        if strategy_name == "DefaultResourceDistribution":
            return DefaultResourceDistribution()
        else:
            raise ValueError("Unknown Distribution Strategy")

    else:
        raise ValueError("Unknown Strategy Type")


def create_nodes(config):
    nodes = []
    for node_config in config["nodes"]:
        node_type = node_config[0]
        config_order = int(node_config[1])
        node_range = node_config[2]
        num_nodes = int(node_config[3])
        cpus = int(node_config[4])
        gpus = int(node_config[5])
        gpu_type = node_config[6]
        memory = int(node_config[7])
        
        for i in range(1, num_nodes+1):
            node = Node(
                name=get_real_node_name(node_type, i),
                id = i, #Used for sorting nodes for selection
                list_position=config_order,
                total_CPUs= cpus,
                total_GPUs=gpus,
                total_memory=memory,
                GPU_type=gpu_type,
            )
            nodes.append(node)

    return nodes


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
        # prefix 100 then append id with no padding
        return f"{node_type}100{id}"

    else:
        # pad to 2 digits
        return f"{node_type}{id:02d}"


def run_simulation(config):
    """Run simulation with the provided configuration"""

    nodes = create_nodes(config)

    # Create strategy instances from config
    node_selection = get_strategy_instance(
        config['node_selection_strategy'],
        'node_selection_strategy'
    )
    resource_distribution = get_strategy_instance(
        config['resource_distribution_strategy'],
        'resource_distribution_strategy'
    )

    # Setup output directory and file paths
    output_directory = config.get('output_directory', 'output')

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    output_events_filename = config.get('output_events', 'simulation_log_events.parquet')
    output_nodes_filename = config.get('output_nodes', 'simulation_log_nodes.parquet')
    output_log_filename = config.get('output_log', 'simulation.log')

    output_events = output_path / output_events_filename
    output_nodes = output_path / output_nodes_filename
    output_log = output_path / output_log_filename

    # Initialize simulation with logging
    slurm_sim = SlurmSimulation(
        config['cluster_name'],
        nodes,
        node_selection,
        resource_distribution,
        log_file=str(output_log)
    )

    input_events = config.get('input_events')
    events_df = pd.read_parquet(input_events)

    event_records = []
    node_records = []

    print(f"Starting simulation with {len(events_df):,} events...")
    for i, row in events_df.iterrows():
        job = Job(
        id=row['job_id'],
        nodes_required=row['nodes_required'],
        CPUs_required=row['CPUs_required'],
        GPUs_required=row['GPUs_required'],
        memory_required=row['memory_required'],
        start_time=row['start_time'],
        end_time=row['end_time'],
        real_node_selection=eval(row['real_node_selection']) if pd.notna(row['real_node_selection']) else None
        )
        event = JobEvent(job, row['action'], row['time'])

        if event.action == 'start':
            slurm_sim.place_job(event.job)
        elif event.action == 'finish':
            slurm_sim.release_job(event.job.id)
        else:
            raise ValueError("Unknown event action")

        state = slurm_sim.get_current_state()

        event_records.append({
            'event_index': i,
            'time': event.time,
            'action': event.action,
            'job_id': event.job.id,
            'active_jobs': state['active_jobs'],
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
            })

    events_table = pa.Table.from_pandas(pd.DataFrame(event_records))
    nodes_table = pa.Table.from_pandas(pd.DataFrame(node_records))

    pq.write_table(events_table, output_events)
    pq.write_table(nodes_table, output_nodes)

    # Print simulation statistics
    stats = slurm_sim.get_stats()
    print("\nSimulation complete:")
    for key, value in stats.items():
        print(f"{key}: {value:,}")

    return None


if __name__ == "__main__":
    config = load_config("config.txt")
    simulation = run_simulation(config)
    


