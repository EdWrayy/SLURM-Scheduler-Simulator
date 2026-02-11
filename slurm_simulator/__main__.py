from .run_simulation import load_config, run_simulation

if __name__ == "__main__":
    config = load_config("config.txt")
    simulation = run_simulation(config)