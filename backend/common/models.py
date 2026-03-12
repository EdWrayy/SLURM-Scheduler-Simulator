"""
Shared data models for SLURM Scheduler Simulator.

This module contains core data classes used across the simulation and data handling components.
"""


class JobEvent:
    """Represents an event in a job's lifecycle (submit, start, end, etc.)."""

    def __init__(self, job, action, time):
        self.job = job
        self.time = time
        self.action = action


class Job:
    """Represents a SLURM job with resource requirements and scheduling information."""

    def __init__(self, id, nodes_required, CPUs_required, GPUs_required, memory_required, start_time, end_time, real_node_selection=None):
        self.id = id
        self.nodes_required = nodes_required
        self.CPUs_required = CPUs_required
        self.GPUs_required = GPUs_required
        self.memory_required = memory_required
        self.start_time = start_time
        self.end_time = end_time
        self.node_utilisation = {}
        self.real_node_selection = real_node_selection  # Exclusively for copy real node selection simulation, can be ignored otherwise
