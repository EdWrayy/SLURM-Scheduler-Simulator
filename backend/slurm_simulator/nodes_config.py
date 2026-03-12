import json
from pathlib import Path


DEFAULT_NODES_CONFIG_PATH = Path(__file__).with_name("nodes.json")


def _extract_nodes(data, source):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        nodes = data.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError(f"'nodes' must be a list in {source}")
        return nodes
    raise ValueError(f"nodes config root must be a list or object in {source}")


def _validate_nodes(nodes, source):
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
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
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = _extract_nodes(data, config_path)
    _validate_nodes(nodes, config_path)
    return nodes


def resolve_node_configs(main_config):
    nodes_path = main_config.get("nodes_config")
    if nodes_path:
        return load_nodes_config(nodes_path)

    inline_nodes = main_config.get("nodes", [])
    if not isinstance(inline_nodes, list):
        raise ValueError("Config key 'nodes' must be a list when provided")
    _validate_nodes(inline_nodes, "main config")
    return inline_nodes
