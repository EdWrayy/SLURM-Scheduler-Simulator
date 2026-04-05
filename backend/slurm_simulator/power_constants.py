import json
from pathlib import Path


POWER_CONSTANTS_PATH = Path(__file__).with_name("power_constants.json")


def _safe_lookup(data, key):
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"power constants config key '{key}' invalid")
    return value


def load_power_constants(path=None):
    config_path = Path(path) if path is not None else POWER_CONSTANTS_PATH

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("power constants config root must be an object")

    cpu_power = _safe_lookup(data, "cpu_power")
    gpu_power = _safe_lookup(data, "gpu_power")
    ram_power = _safe_lookup(data, "ram_power")
    cpu_idle_fraction = data.get("cpu_idle_fraction")
    gpu_idle_fraction = data.get("gpu_idle_fraction")
    if not isinstance(cpu_idle_fraction, (int, float)):
        raise ValueError("power constants 'cpu_idle_fraction' must be a number")
    if not isinstance(gpu_idle_fraction, (int, float)):
        raise ValueError("power constants 'gpu_idle_fraction' must be a number")
    return cpu_power, gpu_power, ram_power, cpu_idle_fraction, gpu_idle_fraction


