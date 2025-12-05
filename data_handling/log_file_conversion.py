import pandas as pd
from pathlib import Path
from common.models import Job, JobEvent 


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
    


def load_job_events(df):

    """
    Given a dataframe containing our accounting logs, we will convert it to events (start job and end job).
    We will also remove any unnecessary or invalid rows.
    """

    df["Start"] = pd.to_datetime(df["Start"], errors="coerce")
    df["End"] = pd.to_datetime(df["End"], errors="coerce")

    events = []

    for _, row in df.iterrows():
        job_id = str(row["JobID"])
        nodes_required = int(row["AllocNodes"]) if not pd.isna(row["AllocNodes"]) else 0
        cpus_required, gpus_required, memory_required = parse_alloc_tres(row.get("AllocTRES", None))

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
            CPUs_required=cpus_required,
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


def load_config(config_file="config.txt"):
    """Load configuration from config file"""
    config = {}
    # Try to find config file in multiple locations
    config_paths = [
        config_file,  # Current directory
        Path(__file__).parent / config_file,  # Same directory as this script
    ]

    config_path = None
    for path in config_paths:
        if Path(path).exists():
            config_path = path
            break

    if config_path is None:
        raise FileNotFoundError(f"Config file '{config_file}' not found in any of: {config_paths}")

    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config


def read_slurm_logs(input_directory):
    """Read all pipe-delimited .txt files from input directory into a single dataframe"""
    input_path = Path(input_directory)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_directory}")

    txt_files = list(input_path.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_directory}")

    print(f"Found {len(txt_files)} .txt files in {input_directory}")

    required_columns = ['AllocTRES', 'AllocNodes', 'JobID', 'Start', 'End', 'NodeList']

    dfs = []
    for txt_file in txt_files:
        print(f"Reading {txt_file.name}...")
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            df = pd.read_csv(f, sep='|', engine='python', usecols=required_columns, on_bad_lines='skip', quoting=3)

        # Drop any .batch, .extern or .number rows
        df = df[~df['JobID'].astype(str).str.contains('.', regex=False)]

        initial_row_count = len(df)

        # Drop rows with pandas NaN/null values (empty cells)
        for column in df.columns:
            mask = df[column].isna()
            dropped_count = mask.sum()
            if dropped_count > 0:
                dropped_rows = df[mask]
                print(f"  Dropped {dropped_count} row(s) from {txt_file.name} due to NaN/null value in column '{column}':")
                for _, row in dropped_rows.iterrows():
                    print(f"    JobID: {row['JobID']}")
                df = df[~mask]

        # Drop rows containing "Unknown", "NaN", or "None" as string values (case-insensitive)
        invalid_values = ['unknown', 'nan', 'none']
        for column in df.columns:
            for invalid_value in invalid_values:
                mask = df[column].astype(str).str.lower().str.strip() == invalid_value
                dropped_count = mask.sum()
                if dropped_count > 0:
                    dropped_rows = df[mask]
                    print(f"  Dropped {dropped_count} row(s) from {txt_file.name} due to '{invalid_value}' string value in column '{column}':")
                    for _, row in dropped_rows.iterrows():
                        print(f"    JobID: {row['JobID']}, {column}: {row[column]}")
                    df = df[~mask]

        final_row_count = len(df)
        total_dropped = initial_row_count - final_row_count
        if total_dropped > 0:
            print(f"  Total rows dropped from {txt_file.name}: {total_dropped} (from {initial_row_count} to {final_row_count})")

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Convert JobID to integer after all filtering is complete
    combined_df['JobID'] = combined_df['JobID'].astype(int)

    return combined_df


if __name__ == "__main__":
    config = load_config("config.txt")
    input_directory = config.get('input_directory')
    output_directory = config.get('output_directory', 'output')
    output_filename = config.get('output_filename', 'slurm_logs')

    df = read_slurm_logs(input_directory)

    print(f"\nDataFrame shape: {df.shape}")
    print(df.head())

    events = load_job_events(df)


    events_df = pd.DataFrame([
    {
        'job_id': event.job.id,
        'nodes_required': event.job.nodes_required,
        'CPUs_required': event.job.CPUs_required,
        'GPUs_required': event.job.GPUs_required,
        'memory_required': event.job.memory_required,
        'start_time': event.job.start_time,
        'end_time': event.job.end_time,
        'real_node_selection': str(event.job.real_node_selection) if event.job.real_node_selection else None,
        'action': event.action,
        'time': event.time
    }
    for event in events])

    print(f"\nEvents shape before deduplication: {events_df.shape}")

    # Remove duplicate events (same job_id and action)
    initial_count = len(events_df)
    events_df = events_df.drop_duplicates(subset=['job_id', 'action'], keep='first')
    duplicates_removed = initial_count - len(events_df)

    print(f"Removed {duplicates_removed} duplicate events")
    print(f"Events shape after deduplication: {events_df.shape}")
    print(f"\nFirst 5 rows:")
    print(events_df.head())

    # Save to parquet
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"{output_filename}.parquet"

    events_df.to_parquet(output_file, index=False, engine='pyarrow')
    print(f"\nEvents dataframe saved to: {output_file}")

