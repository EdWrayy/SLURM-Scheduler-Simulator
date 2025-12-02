from slurm_simulator import SlurmSimulation, Node, Job, DefaultResourceDistribution, CopyRealNodeSelection
import pandas as pd
from datetime import datetime
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
    



def parse_alloc_tres(alloc_tres):
    """
    Parse AllocTRES such as:
      billing=4,cpu=4,gres/gpu=1,mem=30720M,node=1

    Returns (cpus, gpus, memory_mb).
    """
    cpus = None
    gpus = 0
    mem_mb = None

    if pd.isna(alloc_tres):
        return cpus, gpus, mem_mb

    parts = str(alloc_tres).split(",")
    for part in parts:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip()

        if key == "cpu":
            cpus = int(val)
        elif key == "mem":
            mem_mb = int(val.rstrip("M"))
        elif key == "gpu" or key.startswith("gres/gpu"):
            gpus = int(val)

    return cpus, gpus, mem_mb

def expand_node_list(node_list):
    """
    Expand Slurm nodelist expressions such as:
      ruby[053-054,056,058]
    into:
      ['ruby053', 'ruby054', 'ruby056', 'ruby058']

    Also handles simple names such as 'ruby053' or multiple groups
    like 'ruby[001-003],swarma[010-011]'.
    """

    if not node_list or node_list == "":
        return None
    
    if '[' in node_list:
        name, values = node_list.split('[')
        values = values.rstrip(']')
        bounds = values.split(',')
        nodes = []
        for bound in bounds:
            if not '-' in bound:
                nodes.append(name + bound)
                continue
            lb, ub = bound.split('-')
            pad = len(lb)
            lower_bound, upper_bound = int(lb), int(ub)
            for i in range(upper_bound - lower_bound + 1):
                n = lower_bound + i
                nodes.append(name + str(n).zfill(pad))
        return nodes
    else:
        return [node_list] #Just a single numbered node

    



def load_job_events(file_path):
    df = pd.read_parquet(file_path)

    # 1) Filter out .batch and .extern jobs
    df["JobID"] = df["JobID"].astype(str)
    df = df[
        ~df["JobID"].str.endswith(".batch")
        & ~df["JobID"].str.endswith(".extern")
    ]

    # 2) Optional: filter out "None assigned" NodeList rows
    if "NodeList" in df.columns:
        df = df[df["NodeList"].notna()]
        df = df[df["NodeList"].astype(str).str.lower() != "none assigned"]

    for col in ["Start", "End"]:
        if col not in df.columns:
            raise KeyError(f"Column {col} not found in dataframe")

    def valid_time_series(series):
        return (
            series.notna()
            & (series.astype(str) != "0")
            & (series.astype(str).str.lower() != "unknown")
        )

    df = df[valid_time_series(df["Start"]) & valid_time_series(df["End"])]

    df["Start"] = pd.to_datetime(df["Start"], errors="coerce")
    df["End"] = pd.to_datetime(df["End"], errors="coerce")
    df = df[df["Start"].notna() & df["End"].notna()]

    events = []

    for _, row in df.iterrows():
        if "JobIDRaw" in df.columns:
            job_id = str(row["JobIDRaw"])
        else:
            job_id = str(row["JobID"])

        nodes_required = int(row["AllocNodes"]) if not pd.isna(row["AllocNodes"]) else 0
        alloc_cpus = int(row["AllocCPUS"]) if not pd.isna(row["AllocCPUS"]) else 0

        tres_cpus, gpus_required, memory_required = parse_alloc_tres(row.get("AllocTRES", None))
        CPUs_required = alloc_cpus if alloc_cpus > 0 else (tres_cpus or 0)

        node_list_value = row.get("NodeList", None)
        if pd.isna(node_list_value) or str(node_list_value).strip().lower() == "none assigned":
            real_selected_nodes = None
        else:
            real_selected_nodes = expand_node_list(str(node_list_value))

        start_time = row["Start"]
        end_time = row["End"]

        job = Job(
            id=job_id,
            nodes_required=nodes_required,
            CPUs_required=CPUs_required,
            GPUs_required=gpus_required,
            memory_required=memory_required,
            start_time=start_time,
            end_time=end_time,
            real_node_selection=real_selected_nodes,
        )

        events.append(JobEvent(job=job, action="start", time=start_time))
        events.append(JobEvent(job=job, action="finish", time=end_time))

    events.sort(key=lambda ev: ev.time)
    return events

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

    # Initialize simulation
    slurm_sim = SlurmSimulation(
        config['cluster_name'],
        nodes,
        node_selection,
        resource_distribution
    )

    events = load_job_events(config['input_jobs_file'])
    output_parquet = config.get('output_parquet', 'simulation_log.parquet')
    BATCH_SIZE = 10000000
    batch_number = 0
    batch_files = []

    records = []
    print(f"Starting simulation with {len(events):,} events...")
    for i, event in enumerate(events):
        if event.action == 'start':
            slurm_sim.place_job(event.job)
        elif event.action == 'finish':
            slurm_sim.release_job(event.job.id)
        else:
            raise ValueError("Unknown event action")

        state = slurm_sim.get_current_state()

        for n_state in state['nodes']:
            records.append({
                'event_index': i,
                'time': event.time,
                'action': event.action,
                'job_id': event.job.id,
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
                'active_jobs': state['active_jobs'],
            })

        if len(records) >= BATCH_SIZE:
            batch_file = f"simulation_batch_{batch_number}.parquet"
            df_batch = pd.DataFrame(records)
            df_batch.to_parquet(batch_file, index=False, engine='pyarrow')
            batch_files.append(batch_file)
            print(f"  Wrote batch {batch_number} with {len(records):,} records to {batch_file}")
            batch_number += 1
            records = []  # Clear records to free memory

    # Write final batch if any records remain
    if records:
        batch_file = f"simulation_batch_{batch_number}.parquet"
        df_batch = pd.DataFrame(records)
        df_batch.to_parquet(batch_file, index=False, engine='pyarrow')
        batch_files.append(batch_file)
        print(f"  Wrote final batch {batch_number} with {len(records):,} records to {batch_file}")

    # Combine all batch files into final output using ParquetWriter (memory efficient)
    print(f"\nCombining {len(batch_files)} batch files into {output_parquet}...")

    # Read schema from first batch
    first_table = pq.read_table(batch_files[0])
    writer = pq.ParquetWriter(output_parquet, first_table.schema)

    # Write each batch one at a time
    for i, batch_file in enumerate(batch_files):
        table = pq.read_table(batch_file)
        writer.write_table(table)
        print(f"  Combined batch {i+1}/{len(batch_files)}")

    writer.close()

    # Clean up batch files
    import os
    for batch_file in batch_files:
        os.remove(batch_file)

    print(f"  Deleted {len(batch_files)} temporary batch files")

    print(f"\n✓ Simulation complete! Output saved to {output_parquet}")

    return None


if __name__ == "__main__":
    config = load_config("config.txt")
    simulation = run_simulation(config)
    


