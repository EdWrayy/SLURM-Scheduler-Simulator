from datetime import datetime
from pathlib import Path
import re


def _sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned or "unknown"


def _unique_nodes_for_type(island_groups, node_type):
    unique = {}
    for group in island_groups.get(node_type, []):
        for node in group:
            unique[node.name] = node
    return sorted(unique.values(), key=lambda n: (n.list_position, n.id, n.name))


def _job_allowed_types(job, island_groups):
    if job.allowed_node_types:
        return list(dict.fromkeys(job.allowed_node_types))
    return sorted(island_groups.keys())


def _format_bytes_as_gib(value):
    try:
        return f"{float(value) / (1024 ** 3):.3f} GiB"
    except (TypeError, ValueError):
        return "n/a"


def write_failed_placement_debug_file(debug_directory, event_index, event, failure_reason, slurm_sim):
    """
    Write one debug file per failed start placement.
    Includes:
      - job details and failure reason
      - full capacities/usage/free values for every node in every allowed family
    """
    if event.action != "start":
        return None

    debug_directory = Path(debug_directory)
    debug_directory.mkdir(parents=True, exist_ok=True)

    job = event.job
    allowed_types = _job_allowed_types(job, slurm_sim.island_groups)

    safe_job_id = _sanitize_filename(job.id)
    safe_reason = _sanitize_filename(failure_reason or "unknown_failure")
    filename = f"failed_place_event_{int(event_index):07d}_job_{safe_job_id}_{safe_reason}.txt"
    output_path = debug_directory / filename

    lines = []
    lines.append("SLURM SIMULATOR FAILED PLACEMENT DEBUG")
    lines.append("=" * 72)
    lines.append(f"generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"event_index: {event_index}")
    lines.append(f"event_time: {event.time}")
    lines.append(f"failure_reason: {failure_reason}")
    lines.append("")
    lines.append("JOB")
    lines.append("-" * 72)
    lines.append(f"job_id: {job.id}")
    lines.append(f"nodes_required: {job.nodes_required}")
    lines.append(f"CPUs_required: {job.CPUs_required}")
    lines.append(f"GPUs_required: {job.GPUs_required}")
    lines.append(f"memory_required_bytes: {job.memory_required}")
    lines.append(f"memory_required_gib: {_format_bytes_as_gib(job.memory_required)}")
    lines.append(f"start_time: {job.start_time}")
    lines.append(f"end_time: {job.end_time}")
    lines.append(f"allowed_node_types: {job.allowed_node_types}")
    lines.append(f"real_node_selection: {job.real_node_selection}")
    lines.append(f"active_jobs_currently_tracked: {len(slurm_sim.job_tracker)}")
    lines.append("")
    lines.append("ELIGIBLE NODE FAMILIES AND NODE CAPACITIES")
    lines.append("-" * 72)

    for node_type in allowed_types:
        nodes = _unique_nodes_for_type(slurm_sim.island_groups, node_type)
        lines.append(f"family: {node_type}")
        if not nodes:
            lines.append("  no nodes found for this family in simulator island groups")
            lines.append("")
            continue

        family_total_cpu = sum(n.total_CPUs for n in nodes)
        family_total_gpu = sum(n.total_GPUs for n in nodes)
        family_total_mem = sum(n.total_memory for n in nodes)
        family_used_cpu = sum(n.CPUs_in_use for n in nodes)
        family_used_gpu = sum(n.GPUs_in_use for n in nodes)
        family_used_mem = sum(n.memory_in_use for n in nodes)

        lines.append(f"  node_count: {len(nodes)}")
        lines.append(
            f"  family_cpu_used_total_free: {family_used_cpu}/{family_total_cpu}/{family_total_cpu - family_used_cpu}"
        )
        lines.append(
            f"  family_gpu_used_total_free: {family_used_gpu}/{family_total_gpu}/{family_total_gpu - family_used_gpu}"
        )
        lines.append(
            f"  family_mem_used_total_free_bytes: {family_used_mem}/{family_total_mem}/{family_total_mem - family_used_mem}"
        )
        lines.append(
            f"  family_mem_used_total_free_gib: "
            f"{_format_bytes_as_gib(family_used_mem)}/"
            f"{_format_bytes_as_gib(family_total_mem)}/"
            f"{_format_bytes_as_gib(family_total_mem - family_used_mem)}"
        )
        lines.append("  nodes:")

        for node in nodes:
            free_cpu = node.total_CPUs - node.CPUs_in_use
            free_gpu = node.total_GPUs - node.GPUs_in_use
            free_mem = node.total_memory - node.memory_in_use
            lines.append(
                "    "
                f"name={node.name} id={node.id} list_position={node.list_position} "
                f"cpu_used/total/free={node.CPUs_in_use}/{node.total_CPUs}/{free_cpu} "
                f"gpu_used/total/free={node.GPUs_in_use}/{node.total_GPUs}/{free_gpu} "
                f"mem_used/total/free_bytes={node.memory_in_use}/{node.total_memory}/{free_mem} "
                f"mem_free_gib={_format_bytes_as_gib(free_mem)} "
                f"power_W={node.current_power_consumption}"
            )
        lines.append("")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    return output_path
