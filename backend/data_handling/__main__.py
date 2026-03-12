from .log_file_conversion import convert_logs
from pathlib import Path
from backend.common.config import load_config

if __name__ == "__main__":
    config = load_config(Path(__file__).with_name("config.json"))
    convert_logs(config)
