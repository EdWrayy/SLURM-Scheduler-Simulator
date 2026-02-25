import pandas as pd
import re
from pathlib import Path
from common.models import Job, JobEvent
from common.config import load_config


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


def expand_node_list(node_list: str):
    """
    Expand Slurm nodelist expressions into a list of concrete node names.

    Examples:
      ruby053
        -> ['ruby053']

      ruby[053-054,056,058]
        -> ['ruby053', 'ruby054', 'ruby056', 'ruby058']

      ruby[001-003],swarma[1001-1005],coral01
        -> ['ruby001','ruby002','ruby003','swarma1001',...,'coral01']
    """
    if node_list is None:
        return None

    s = str(node_list).strip()
    if s == "" or s.lower() == "none assigned":
        return None

    # Split by commas that are NOT inside [...]
    parts = re.split(r",(?![^\[]*\])", s)

    out = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Match "prefix[inside]"
        m = re.match(r"^([^\[]+)\[(.+)\]$", part)
        if not m:
            # Plain hostname
            out.append(part)
            continue

        prefix = m.group(1)
        inside = m.group(2)

        for token in inside.split(","):
            token = token.strip()
            if not token:
                continue

            if "-" in token:
                lb, ub = token.split("-", 1)
                pad = len(lb)
                for n in range(int(lb), int(ub) + 1):
                    out.append(prefix + str(n).zfill(pad))
            else:
                out.append(prefix + token)

    return out or None
    


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


def convert_logs(config):
    """
    Convert SLURM accounting logs to parquet format.
    
    Args:
        config: Configuration dictionary with keys:
            - input_directory: Directory containing .txt log files
            - output_directory: Directory to save output (default: 'output')
            - output_filename: Name of output file without extension (default: 'slurm_logs')
    """
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
    
    return output_file


if __name__ == "__main__":
    config = load_config(Path(__file__).with_name("config.txt"))
    convert_logs(config)

