from pathlib import Path
import json

from .slurm_simulator import (
    GreedyBlockFill,
    EvenSplit,
    CopyRealNodeSelection,
    NaiveFeasible,
    Active_Node_Reuse,
    LoadSpreading,
    CPU_Best_Fit,
    GPU_Best_Fit,
    Manhattan_Slack_Best_Fit,
    Dominant_Resource_Best_Fit,
    Workload_Aware_Weighted_Manhattan_Slack,
    Workload_Aware_Weighted_Dominant_Resource,
    ActiveOnlyPowerModel,
    LinearWithIdlePowerModel,
    LinearWithSleepPowerModel,
)


WORKLOAD_AWARE_STRATEGIES = {
    "Workload_Aware_Weighted_Manhattan_Slack": Workload_Aware_Weighted_Manhattan_Slack,
    "Workload_Aware_Weighted_Dominant_Resource": Workload_Aware_Weighted_Dominant_Resource,
}


def _load_family_weights() -> dict:
    weights_file = Path(__file__).with_name("resource_weights.json")
    if not weights_file.exists():
        raise FileNotFoundError(
            f"resource_weights.json not found at {weights_file}. "
            "Run the Resource Weights tool in the Data Handling tab first."
        )
    with open(weights_file) as f:
        return json.load(f)


def get_strategy_instance(strategy_name, strategy_type):
    if strategy_type == "node_selection_strategy":
        if strategy_name == "CopyRealNodeSelection":
            return CopyRealNodeSelection()
        elif strategy_name == "NaiveFeasible":
            return NaiveFeasible()
        elif strategy_name == "Active_Node_Reuse":
            return Active_Node_Reuse()
        elif strategy_name == "LoadSpreading":
            return LoadSpreading()
        elif strategy_name == "CPU_Best_Fit":
            return CPU_Best_Fit()
        elif strategy_name == "GPU_Best_Fit":
            return GPU_Best_Fit()
        elif strategy_name == "Manhattan_Slack_Best_Fit":
            return Manhattan_Slack_Best_Fit()
        elif strategy_name == "Dominant_Resource_Best_Fit":
            return Dominant_Resource_Best_Fit()
        elif strategy_name in WORKLOAD_AWARE_STRATEGIES:
            family_weights = _load_family_weights()
            return WORKLOAD_AWARE_STRATEGIES[strategy_name](family_weights)
        else:
            raise ValueError(f"Unknown Selection Strategy: {strategy_name}")

    elif strategy_type == "resource_distribution_strategy":
        if strategy_name == "GreedyBlockFill":
            return GreedyBlockFill()
        elif strategy_name == "EvenSplit":
            return EvenSplit()
        else:
            raise ValueError(f"Unknown Distribution Strategy: {strategy_name}")

    elif strategy_type == "power_model":
        if strategy_name == "ActiveOnlyPowerModel":
            return ActiveOnlyPowerModel()
        elif strategy_name == "LinearWithIdlePowerModel":
            return LinearWithIdlePowerModel()
        elif strategy_name == "LinearWithSleepPowerModel":
            return LinearWithSleepPowerModel()
        else:
            raise ValueError(f"Unknown Power Model: {strategy_name}")

    else:
        raise ValueError(f"Unknown Strategy Type: {strategy_type}")
