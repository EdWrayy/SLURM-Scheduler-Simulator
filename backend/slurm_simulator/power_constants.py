import json
from pathlib import Path


POWER_CONSTANTS_PATH = Path(__file__).with_name("power_constants.json")


def _require_dict(data, key):
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"power constants config key '{key}' must be an object")
    return value


def load_power_constants(path=None):
    config_path = Path(path) if path is not None else POWER_CONSTANTS_PATH

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("power constants config root must be an object")

    cpu_power = _require_dict(data, "cpu_power")
    gpu_power = _require_dict(data, "gpu_power")
    ram_power = _require_dict(data, "ram_power")
    return cpu_power, gpu_power, ram_power


CPU_POWER, GPU_POWER, RAM_POWER = load_power_constants()
