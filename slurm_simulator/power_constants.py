NODE_HARDWARE = {
    "ruby": {
        "cpu": "epyc_7452",
        "gpu": None,
        "ram": "ddr4",
        "num_gpus": 0,
    },
    "rose": {
        "cpu": "xeon_6336y",
        "gpu": "a100",
        "ram": "ddr4",
        "num_gpus": 2,
    },
    "swarma": {
        "cpu": "epyc_7413",
        "gpu": "a100",
        "ram": "ddr4",
        "num_gpus": 4,
    },
    "swarmh": {
        "cpu": "xeon_8468",
        "gpu": "h100",
        "ram": "ddr4",
        "num_gpus": 8,
    },
    "blossom": {
        "cpu": "epyc_9255",
        "gpu": "h200",
        "ram": "ddr4",
        "num_gpus": 4,
    },
    "flamingo": {
        "cpu": "epyc_9275f",
        "gpu": "h200",
        "ram": "ddr4",
        "num_gpus": 2,
    },
    "cotton": {
        "cpu": "epyc_9255",
        "gpu": "l4",
        "ram": "ddr4",
        "num_gpus": 8,
    },
    "coral": {
        "cpu": "epyc_9255",
        "gpu": "l40s",
        "ram": "ddr4",
        "num_gpus": 8,
    },
    "cherry": {
        "cpu": "epyc_9534",
        "gpu": "mi300x",
        "ram": "ddr4",
        "num_gpus": 8,
    },
    "swarml4": {
        "cpu": "epyc_9534",
        "gpu": "l4",
        "ram": "ddr4",
        "num_gpus": 8,
    },
    "ecsai": {
        "cpu": "epyc_9534",
        "gpu": "mi300x",
        "ram": "ddr4",
        "num_gpus": 8,
    },
}

CPU_POWER = {
    "epyc_7452":   {"idle_W": 70,  "max_W": 180},
    "epyc_7413":   {"idle_W": 75,  "max_W": 180},
    "epyc_9255":   {"idle_W": 90,  "max_W": 240},
    "epyc_9275f":  {"idle_W": 100, "max_W": 280},
    "epyc_9534":   {"idle_W": 120, "max_W": 320},
    "xeon_6336y":  {"idle_W": 80,  "max_W": 200},
    "xeon_8468":   {"idle_W": 120, "max_W": 350},
}


GPU_POWER = {
    "a100":   {"idle_W": 50,  "max_W": 400},
    "h100":   {"idle_W": 60,  "max_W": 700},
    "h200":   {"idle_W": 65,  "max_W": 700},
    "l4":     {"idle_W": 15,  "max_W": 72},
    "l40s":   {"idle_W": 40,  "max_W": 300},
    "mi300x": {"idle_W": 80,  "max_W": 750},
}


RAM_POWER = {
    "ddr4": {"idle_W_per_GB": 0.25, "max_W_per_GB": 0.5,}
}
