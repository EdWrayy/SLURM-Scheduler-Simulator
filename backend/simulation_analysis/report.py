from pathlib import Path
from datetime import datetime
import json


def write_report(config, plots, overall_metrics):
    out_dir = Path(config["output_report_directory"])
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = config.get("run_id")
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_dir = out_dir / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save plots as PNGs
    plot_files = {}
    for name, fig in plots.items():
        fname = f"{name}.png"
        fpath = plots_dir / fname
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        plot_files[name] = str(fpath)

    # Write summary.json (flat dict for meta-analysis)
    summary = dict(overall_metrics)
    summary["run_id"] = run_id
    summary["plot_files"] = plot_files  # optional, handy for debugging

    summary_path = report_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary_path
