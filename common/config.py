from pathlib import Path

def load_config(config_file="config.txt"):
    """Load configuration from config file"""
    here = Path(__file__).parent
    path = here / config_file
    
    config = {}
    node_list = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if key == "node":
                    _, rhs = line.split("=", 1)
                    parts = [p.strip() for p in rhs.split(",")]
                    node_list.append(parts)
                elif key.startswith("partition"):
                    pass #ignore for now
                else:
                    config[key.strip()] = value.strip()
    
    config["nodes"] = node_list
    return config

