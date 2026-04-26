import json
from pathlib import Path

PATH_KEYS = {
    "input_events",
    "nodes_config",
    "power_constants_config",
    "input_directory",
    "output_directory",
    "input_events_directory",
    "input_nodes_directory",
    "output_events_directory",
    "output_nodes_directory",
    "output_debug_directory",
}


def _resolve_config_paths(config, config_dir):
    for key, value in list(config.items()):
        if key in PATH_KEYS and isinstance(value, str) and value.strip():
            path_value = Path(value)
            if not path_value.is_absolute():
                config[key] = str((config_dir / path_value).resolve())


def load_config(config_file):
    """Load configuration from a JSON file."""
    path = Path(config_file).resolve()
    config_dir = path.parent

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    _resolve_config_paths(config, config_dir)
    return config

