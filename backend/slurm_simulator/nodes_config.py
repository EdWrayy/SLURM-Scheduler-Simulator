import json
from pathlib import Path


DEFAULT_NODES_CONFIG_PATH = Path(__file__).with_name("nodes.json")


def _validate_nodes(nodes, source):
    # Checks each node has the required hardware and CPU fields before we try to use them
    for i, node in enumerate(nodes):
        node_type = node.get("node_type", f"index_{i}")
        hardware = node.get("hardware")
        if not isinstance(hardware, dict):
            raise ValueError(f"Node '{node_type}' in {source} must include a 'hardware' object")
        if "cpu" not in hardware or "ram" not in hardware:
            raise ValueError(f"Node '{node_type}' in {source} hardware must include 'cpu' and 'ram'")
        physical_cpus = node.get("physical_cpus_per_node")
        if not isinstance(physical_cpus, int) or physical_cpus <= 0:
            raise ValueError(
                f"Node '{node_type}' in {source} must include positive integer 'physical_cpus_per_node'"
            )
        cores_per_cpu = node.get("cores_per_physical_cpu")
        if not isinstance(cores_per_cpu, int) or cores_per_cpu <= 0:
            raise ValueError(
                f"Node '{node_type}' in {source} must include positive integer 'cores_per_physical_cpu'"
            )


def load_nodes_config(path):
    # Reads a nodes JSON file from disk and returns the validated list of node configs
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    nodes = data["nodes"]
    _validate_nodes(nodes, config_path)
    return nodes
