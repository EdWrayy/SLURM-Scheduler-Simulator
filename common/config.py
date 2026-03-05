import json
from pathlib import Path


def load_config(config_file):
    """Load configuration from a JSON file."""
    path = Path(config_file).resolve()

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")

    nodes = config.get("nodes")
    if nodes is None:
        config["nodes"] = []
    elif not isinstance(nodes, list):
        raise ValueError("Config key 'nodes' must be a list when provided")

    return config

