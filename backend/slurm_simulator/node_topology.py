import re

from .slurm_simulator import (
    Node,
    ActiveOnlyPowerModel,
    LinearWithIdlePowerModel,
    LinearWithSleepPowerModel,
)
from .power_constants import load_power_constants

BYTES_PER_MIB = 1024 ** 2


def create_nodes(config, node_configs):
    nodes = []
    node_type_by_name = {}
    seen_node_names = set()
    cpu_power, gpu_power, ram_power, cpu_idle_fraction, gpu_idle_fraction = load_power_constants(
        config.get("power_constants_config")
    )

    power_model_name = config["power_model"]
    if power_model_name == "ActiveOnlyPowerModel":
        power_model = ActiveOnlyPowerModel()
    elif power_model_name == "LinearWithIdlePowerModel":
        power_model = LinearWithIdlePowerModel(cpu_idle_fraction, gpu_idle_fraction)
    elif power_model_name == "LinearWithSleepPowerModel":
        power_model = LinearWithSleepPowerModel(cpu_idle_fraction, gpu_idle_fraction)
    else:
        raise ValueError(f"Unknown power model: {power_model_name}")

    for node_config in node_configs:
        node_type = node_config["node_type"]
        config_order = int(node_config["config_order"])
        node_range = node_config["node_range"]
        num_nodes = int(node_config["num_nodes"])
        cpus = int(node_config["cpus_per_node"])
        physical_cpus = int(node_config["physical_cpus_per_node"])
        cores_per_physical_cpu = int(node_config["cores_per_physical_cpu"])
        gpus = int(node_config["gpus_per_node"])
        gpu_type = node_config["gpu_type"]
        memory_mb = int(node_config["memory_mb"])
        memory = memory_mb * BYTES_PER_MIB
        hardware = node_config.get("hardware")
        if not isinstance(hardware, dict):
            raise ValueError(f"Node '{node_type}' is missing hardware details ")
        cpu_hw = hardware.get("cpu")
        gpu_hw = hardware.get("gpu")
        ram_hw = hardware.get("ram")

        if cpu_hw not in cpu_power:
            raise ValueError(f"Node '{node_type}' references unknown CPU power key '{cpu_hw}'")
        cpu_p = cpu_power[cpu_hw]
        cpu_max_power = cpu_p["max_W"]

        if gpu_hw is None:
            gpu_max_power = 0.0
        else:
            if gpu_hw not in gpu_power:
                raise ValueError(f"Node '{node_type}' references unknown GPU power key '{gpu_hw}'")
            gpu_p = gpu_power[gpu_hw]
            gpu_max_power = gpu_p["max_W"]

        dimms_per_node = int(node_config["dimms_per_node"])
        dimm_size_GB = int(hardware["dimm_size_GB"])
        if ram_hw not in ram_power:
            raise ValueError(f"Node '{node_type}' references unknown RAM power key '{ram_hw}'")
        dimm_size_key = str(dimm_size_GB)
        if dimm_size_key not in ram_power[ram_hw]:
            raise ValueError(f"Node '{node_type}' references unknown DIMM size '{dimm_size_GB}GB' for '{ram_hw}'")
        ram_fixed_power = dimms_per_node * ram_power[ram_hw][dimm_size_key]["W_per_dimm"]

        node_names_and_ids = expand_node_names_from_range(node_type, node_range, num_nodes)
        for node_name, node_id in node_names_and_ids:
            if node_name in seen_node_names:
                raise ValueError(f"Duplicate node name generated: '{node_name}'")
            seen_node_names.add(node_name)
            node_type_by_name[node_name] = node_type

            node = Node(
                name=node_name,
                node_type=node_type,
                id=node_id,
                list_position=config_order,
                physical_CPUs=physical_cpus,
                cores_per_physical_CPU=cores_per_physical_cpu,
                total_CPUs=cpus,
                total_GPUs=gpus,
                total_memory=memory,
                CPU_Max_Power=cpu_max_power,
                GPU_Max_Power=gpu_max_power,
                RAM_Power=ram_fixed_power,
                power_model=power_model,
            )
            nodes.append(node)

    node_by_name = {node.name: node for node in nodes}

    island_groups = {}
    for node_config in node_configs:
        node_type = node_config["node_type"]
        for island_str in node_config.get("network_islands", []):
            island_names = expand_island_to_node_names(island_str)
            group = [node_by_name[n] for n in island_names if n in node_by_name]
            if group:
                island_groups.setdefault(node_type, []).append(group)

    nodes_in_islands = {node for groups in island_groups.values() for group in groups for node in group}
    for node in nodes:
        if node not in nodes_in_islands:
            island_groups.setdefault(node.node_type, []).append([node])

    return nodes, node_type_by_name, island_groups


def expand_island_to_node_names(island_str):
    """
    Parse island node list notation into node names
    Can handle ranges, lists and single values
    """
    match = re.fullmatch(r"([a-zA-Z]+\d*)\[([^\]]+)\]", island_str)
    if match is None:
        return {island_str}
    prefix = match.group(1)
    range_part = match.group(2)
    names = set()
    for token in range_part.split(","):
        token = token.strip()
        if "-" in token:
            start_str, end_str = token.split("-")
            width = max(len(start_str), len(end_str))
            for i in range(int(start_str), int(end_str) + 1):
                names.add(f"{prefix}{str(i).zfill(width)}")
        else:
            names.add(f"{prefix}{token}")
    return names


def expand_node_names_from_range(node_type, node_range, num_nodes):
    """
    Parse range notation into start and end values
    Create a of given node type for each number in the range
    Handle any invalid configs
    """
    pattern = r"\[(\d+)(?:-(\d+))?\]"
    match = re.fullmatch(pattern, str(node_range))
    if match is None:
        raise ValueError(
            f"Invalid node_range '{node_range}' for node_type '{node_type}'. "
            "Expected format like '[001-074]' or '[01]'."
        )

    start_str = match.group(1)
    end_str = match.group(2) if match.group(2) is not None else start_str
    width = max(len(start_str), len(end_str))
    start = int(start_str)
    end = int(end_str)

    if end < start:
        raise ValueError(f"Invalid node_range '{node_range}': end index is less than start index")

    expected_num_nodes = end - start + 1
    if expected_num_nodes != num_nodes:
        raise ValueError(
            f"node_range '{node_range}' implies {expected_num_nodes} nodes, "
            f"but num_nodes is {num_nodes} for node_type '{node_type}'"
        )

    return [(f"{node_type}{idx:0{width}d}", idx) for idx in range(start, end + 1)]


def print_simulation_configuration(config, node_configs, nodes, node_type_by_name):
    print("\n" + "=" * 70)
    print("SLURM SIMULATION CONFIGURATION")
    print("=" * 70)

    print("\nStrategies:")
    print(f"  Node selection: {config['node_selection_strategy']}")
    print(f"  Resource distribution: {config['resource_distribution_strategy']}")
    print(f"  Power model: {config['power_model']}")

    print("\nConfigured node groups:")
    for n in node_configs:
        node_type = n["node_type"]
        order = n["config_order"]
        node_range = n["node_range"]
        count = n["num_nodes"]
        cpus = n["cpus_per_node"]
        physical_cpus = n["physical_cpus_per_node"]
        cores_per_physical_cpu = n["cores_per_physical_cpu"]
        gpus = n["gpus_per_node"]
        mem = n["memory_mb"]

        print(
            f"  type={str(node_type):10s} "
            f"order={str(order):2s} "
            f"range={str(node_range):15s} "
            f"count={str(count):3s} "
            f"pCPUs={str(physical_cpus):2s} "
            f"cores/CPU={str(cores_per_physical_cpu):2s} "
            f"CPUs={str(cpus):4s} "
            f"GPUs={str(gpus):2s} "
            f"mem_mb={mem}"
        )

    print("\nNodes instantiated in simulation:")
    print(f"  Total nodes: {len(nodes)}")

    names = sorted(n.name for n in nodes)

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

    print("=" * 70 + "\n")
