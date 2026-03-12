from backend.common.config import load_config
from .input import load_simulation_output
from .metrics import calculate_all_metrics
from .plots import make_plots
from .report import write_report
from pathlib import Path

if __name__ == "__main__":
    print("Loading config...")
    config = load_config(Path(__file__).with_name("config.json"))
    
    print("Loading simulation output (events + nodes)...")
    events_df, nodes_df = load_simulation_output(config)
    print(f"  Loaded {len(events_df)} event(s) and {len(nodes_df)} node(s)")
    
    print("Calculating metrics...")
    snapshots_df, overall_metrics = calculate_all_metrics(events_df, nodes_df, config)
    print(f"  Calculated {len(snapshots_df)} snapshot(s)")

    print("Making plots...")
    plots = make_plots(snapshots_df, config, events_df)
    print(f"  Created {len(plots)} plot(s)")
    
    print("Writing report...")
    report_path = write_report(config, plots, overall_metrics)
    print(f"Done. Report saved to: {report_path}")
    
