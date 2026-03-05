from .run_simulation import run_simulation
from pathlib import Path
from common.config import load_config

if __name__ == "__main__":
    config = load_config(Path(__file__).with_name("config.json"))
    simulation = run_simulation(config)
