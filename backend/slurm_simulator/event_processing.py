import ast

import pandas as pd

from backend.common.models import Job, JobEvent

BYTES_PER_GIB = 1024 ** 3


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
        allowed_node_types=ast.literal_eval(row.allowed_node_types) if pd.notna(row.allowed_node_types) else None,
    )
    event = JobEvent(job, row.action, row.time)
    return job, event


def compute_job_work_metadata(row):
    if row.action != "start":
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

    if event.action == "start":
        event_success, info = slurm_sim.place_job(event.job)
        if not event_success:
            failure_reason = info.get("reason")

    elif event.action == "finish":
        release_success = slurm_sim.release_job(event.job.id)
        event_success = release_success
        if not release_success:
            failure_reason = "release_failed"
    else:
        raise ValueError("Unknown event action")

    return event_success, failure_reason


def append_records(
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
):
    event_records.append(
        {
            "event_index": i,
            "time": event.time,
            "action": event.action,
            "job_id": event.job.id,
            "active_jobs": state["active_jobs"],
            "dt_seconds": dt_seconds,
            "success": event_success,
            "failure_reason": failure_reason,
            "job_start_time": start_ts,
            "job_end_time": end_ts,
            "job_duration_seconds": job_duration_seconds,
            "job_cpu_seconds": job_cpu_seconds,
            "job_gpu_seconds": job_gpu_seconds,
            "job_mem_gb_seconds": job_mem_gb_seconds,
        }
    )

    # Write one aggregated node snapshot per event to reduce IO and memory pressure.
    nodes = state["nodes"]
    active_nodes = [n for n in nodes if n["power_consumption"] > 0]
    gpu_active_nodes = [n for n in active_nodes if n["total_GPUs"] > 0]

    node_records.append(
        {
            "event_index": i,
            "total_cpus": int(sum(n["total_CPUs"] for n in nodes)),
            "total_gpus": int(sum(n["total_GPUs"] for n in nodes)),
            "total_mem": int(sum(n["total_memory"] for n in nodes)),
            "cpus_used": int(sum(n["CPUs_in_use"] for n in nodes)),
            "gpus_used": int(sum(n["GPUs_in_use"] for n in nodes)),
            "mem_used": int(sum(n["memory_in_use"] for n in nodes)),
            "cluster_power_W": float(sum(n["power_consumption"] for n in nodes)),
            "active_nodes": int(len(active_nodes)),
            "free_cpus_sum": int(sum(n["total_CPUs"] - n["CPUs_in_use"] for n in active_nodes)),
            "free_gpus_sum": int(
                sum(
                    (n["total_GPUs"] - n["GPUs_in_use"]) if n["total_GPUs"] > 0 else 0
                    for n in active_nodes
                )
            ),
            "free_mem_sum": int(sum(n["total_memory"] - n["memory_in_use"] for n in active_nodes)),
            "gpu_active_nodes": int(len(gpu_active_nodes)),
        }
    )
