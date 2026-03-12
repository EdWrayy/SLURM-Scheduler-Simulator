import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from backend.common.config import load_config
from backend.simulation_analysis.input import load_simulation_output
from backend.simulation_analysis.metrics import calculate_all_metrics
from backend.simulation_analysis.plots import make_plots
from backend.simulation_analysis.report import write_report
from frontend.ui.settings_store import get_output_root_directory
from frontend.ui.terminal_panel import TerminalPanel, TerminalStream


class SimulationAnalysisTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.analysis_config_path = (
            Path(__file__).resolve().parents[2] / "backend" / "simulation_analysis" / "config.json"
        )
        self.analysis_config = self._load_analysis_config()

        layout = QVBoxLayout()

        title = QLabel("Simulation Analysis")
        layout.addWidget(title)

        config_group = QGroupBox("Analysis Configuration")
        config_layout = QFormLayout()

        self.baseline_usage_input = QLineEdit(
            str(self.analysis_config.get("baseline_total_energy_kwh", ""))
        )
        self.baseline_usage_input.setPlaceholderText("e.g. 4079354.072")
        self.baseline_usage_input.editingFinished.connect(self._save_baseline_usage)
        config_layout.addRow("Baseline Usage (kWh):", self.baseline_usage_input)

        input_paths = self._load_analysis_input_paths()

        self.events_folder_path = QLineEdit(str(input_paths["events"]))
        self.events_folder_browse = QPushButton("Browse...")
        self.events_folder_browse.clicked.connect(self._browse_events_folder)
        self.events_folder_path.editingFinished.connect(self._save_events_directory)
        events_row = QHBoxLayout()
        events_row.addWidget(self.events_folder_path)
        events_row.addWidget(self.events_folder_browse)
        config_layout.addRow("Events Folder:", events_row)

        self.nodes_folder_path = QLineEdit(str(input_paths["nodes"]))
        self.nodes_folder_browse = QPushButton("Browse...")
        self.nodes_folder_browse.clicked.connect(self._browse_nodes_folder)
        self.nodes_folder_path.editingFinished.connect(self._save_nodes_directory)
        nodes_row = QHBoxLayout()
        nodes_row.addWidget(self.nodes_folder_path)
        nodes_row.addWidget(self.nodes_folder_browse)
        config_layout.addRow("Nodes Folder:", nodes_row)

        self.output_name_input = QLineEdit(
            str(self.analysis_config.get("ui_output_name", self._default_output_name_from_input()))
        )
        self.output_name_input.editingFinished.connect(self._save_output_name)
        config_layout.addRow("Output Name:", self.output_name_input)

        self.output_report_directory_path = QLineEdit("")
        self.output_report_directory_path.setReadOnly(True)
        config_layout.addRow("Output Report Directory:", self.output_report_directory_path)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        actions_row = QHBoxLayout()
        self.run_simulation_analysis_button = QPushButton("Run Simulation Analysis")
        self.run_simulation_analysis_button.clicked.connect(self._run_simulation_analysis)
        actions_row.addWidget(self.run_simulation_analysis_button)
        layout.addLayout(actions_row)

        self.terminal_panel = TerminalPanel("Simulation Analysis Terminal")
        layout.addWidget(self.terminal_panel)
        self.setLayout(layout)
        self._refresh_output_report_directory()

    def _load_analysis_input_paths(self) -> dict[str, Path]:
        return {
            "events": Path(self.analysis_config.get("input_events_directory", "")),
            "nodes": Path(self.analysis_config.get("input_nodes_directory", "")),
        }

    def _browse_events_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Events Folder")
        if folder:
            self.events_folder_path.setText(folder)
            self._save_events_directory()

    def _browse_nodes_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Nodes Folder")
        if folder:
            self.nodes_folder_path.setText(folder)
            self._save_nodes_directory()

    def _save_events_directory(self) -> None:
        self._set_config_value("input_events_directory", self.events_folder_path.text().strip())
        if not self.analysis_config.get("ui_output_name"):
            self.output_name_input.setText(self._default_output_name_from_input())
        self._refresh_output_report_directory()

    def _save_nodes_directory(self) -> None:
        self._set_config_value("input_nodes_directory", self.nodes_folder_path.text().strip())

    def _save_baseline_usage(self) -> None:
        value = self.baseline_usage_input.text().strip()
        if not value:
            self._set_config_value("baseline_total_energy_kwh", "")
            return
        try:
            self._set_config_value("baseline_total_energy_kwh", float(value))
        except ValueError:
            QMessageBox.warning(self, "Invalid Baseline", "Baseline Usage must be a valid number.")

    def _save_output_name(self) -> None:
        name = self.output_name_input.text().strip() or self._default_output_name_from_input()
        self.output_name_input.setText(name)
        self._set_config_value("ui_output_name", name)
        self._refresh_output_report_directory()

    def _run_simulation_analysis(self) -> None:
        self._save_events_directory()
        self._save_nodes_directory()
        self._save_baseline_usage()
        self._save_output_name()
        self._refresh_output_report_directory()

        self.run_simulation_analysis_button.setEnabled(False)
        terminal_stream = TerminalStream(self.terminal_panel)
        self.terminal_panel.append_text("\n[simulation-analysis] Starting run...\n")

        try:
            with redirect_stdout(terminal_stream), redirect_stderr(terminal_stream):
                print("Loading config...")
                config = load_config(self.analysis_config_path)

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
        except Exception as exc:
            self.terminal_panel.append_text(f"\n[simulation-analysis] Failed: {exc}\n")
            QMessageBox.critical(self, "Simulation Analysis Failed", str(exc))
            self.run_simulation_analysis_button.setEnabled(True)
            return

        self.terminal_panel.append_text("\n[simulation-analysis] Completed successfully.\n")
        QMessageBox.information(self, "Simulation Analysis Complete", "Simulation analysis finished.")
        self.run_simulation_analysis_button.setEnabled(True)

    def _load_analysis_config(self) -> dict:
        try:
            with self.analysis_config_path.open("r", encoding="utf-8") as file:
                config = json.load(file)
            if isinstance(config, dict):
                return config
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _set_config_value(self, key: str, value: str | float) -> None:
        self.analysis_config[key] = value
        self._save_analysis_config()

    def _save_analysis_config(self) -> None:
        with self.analysis_config_path.open("w", encoding="utf-8") as file:
            json.dump(self.analysis_config, file, indent=2)
            file.write("\n")

    def _default_output_name_from_input(self) -> str:
        events_path = Path(str(self.analysis_config.get("input_events_directory", "")).strip())
        if not events_path.parts:
            return "analysis_run"

        parts = list(events_path.parts)
        if parts and parts[-1].lower() == "events":
            parts = parts[:-1]

        if len(parts) >= 2:
            return str(Path(parts[-2].replace(" ", "_")) / parts[-1].replace(" ", "_"))
        return parts[-1].replace(" ", "_")

    def _refresh_output_report_directory(self) -> None:
        root = get_output_root_directory().strip()
        if not root:
            self.output_report_directory_path.setText("<Set output root in General Settings>")
            return

        output_name = self.output_name_input.text().strip() or "analysis_run"
        relative_output_path = self._normalize_relative_output_name(output_name)
        output_dir = Path(root) / "simulation_analysis" / relative_output_path
        output_dir_str = str(output_dir)

        self.output_report_directory_path.setText(output_dir_str)
        self._set_config_value("output_report_directory", output_dir_str)

    def _normalize_relative_output_name(self, value: str) -> Path:
        cleaned = value.replace("/", "\\").strip("\\")
        parts = [part for part in cleaned.split("\\") if part]
        if not parts:
            return Path("analysis_run")
        return Path(*parts)
