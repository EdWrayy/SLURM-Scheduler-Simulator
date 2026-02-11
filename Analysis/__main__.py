from common.config import load_config
from .input import load_simulation_output
from .metrics import calculate_all_metrics
from .plots import make_plots
from .report import write_pdf_report

if __name__ == "__main__":
    config = load_config("config.txt")
    events_df, nodes_df = load_simulation_output(config)
    snapshots, overall_metrics = calculate_all_metrics(events_df, nodes_df)
    plots = make_plots(snapshots, config, events_df)
    write_pdf_report(config, plots, overall_metrics)
    
