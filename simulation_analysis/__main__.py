from common.config import load_config
from .input import load_simulation_output
from .metrics import calculate_all_metrics
from .plots import make_plots
from .report import write_pdf_report
from pathlib import Path

if __name__ == "__main__":
    print("Loading config...")
    config = load_config(Path(__file__).with_name("config.txt"))
    
    print("Loading simulation output (events + nodes)...")
    events_df, nodes_df = load_simulation_output(config)
    print(f"  Loaded {len(events_df)} event(s) and {len(nodes_df)} node(s)")
    
    print("Calculating metrics...")
    snapshots, overall_metrics = calculate_all_metrics(events_df, nodes_df)
    print(f"  Calculated {len(snapshots)} snapshot(s)")

    print("Making plots...")
    plots = make_plots(snapshots, config, events_df)
    print(f"  Created {len(plots)} plot(s)")
    
    print("Writing PDF report...")
    pdf_path = write_pdf_report(config, plots, overall_metrics)
    print(f"Done. Report saved to: {pdf_path}")
    
