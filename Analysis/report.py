from pathlib import Path
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def write_pdf_report(config, plots, overall_metrics):

    out_dir = Path(config["output_report_directory"])
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = out_dir / f"simulation_report_{timestamp}.pdf"

    title = f"Simulation Report ({timestamp})"

    with PdfPages(pdf_path) as pdf:

        # Summary page as a figure with text
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
        ax.axis("off")
        ax.set_title(title, pad=20)

        lines = []
        for k, v in overall_metrics.items():
            if isinstance(v, float):
                lines.append(f"{k}: {v:.6g}")
            else:
                lines.append(f"{k}: {v}")

        ax.text(
            0.02, 0.95,
            "\n".join(lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=10
        )

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Plot pages
        for _, fig in plots.items():
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path
