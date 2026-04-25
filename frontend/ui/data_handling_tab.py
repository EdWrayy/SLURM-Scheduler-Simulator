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
from backend.data_handling.convert_to_csv import convert_to_csv
from backend.data_handling.filter_node_family import filter_to_node_family, filter_to_single_node
from backend.data_handling.log_file_conversion import convert_logs
from backend.data_handling.resource_weights import compute_resource_weights, save_weights
from backend.data_handling.workload_breakdown import workload_breakdown
from frontend.ui.terminal_panel import TerminalPanel, TerminalStream


class DataHandlingTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.data_handling_config_path = (
            Path(__file__).resolve().parents[2] / "backend" / "data_handling" / "config.json"
        )

        config = self._load_conversion_config()

        layout = QVBoxLayout()

        title = QLabel("Data Handling")
        layout.addWidget(title)

        parquet_group = QGroupBox("Batch Parquet Conversion")
        parquet_layout = QFormLayout()

        self.parquet_input_directory_path = QLineEdit(config["input_directory"])
        self.parquet_input_directory_browse = QPushButton("Browse...")
        self.parquet_input_directory_browse.clicked.connect(self._browse_parquet_input_directory)
        self.parquet_input_directory_path.editingFinished.connect(self._save_parquet_input_directory)
        parquet_input_row = QHBoxLayout()
        parquet_input_row.addWidget(self.parquet_input_directory_path)
        parquet_input_row.addWidget(self.parquet_input_directory_browse)
        parquet_layout.addRow("Input Logs Folder:", parquet_input_row)

        self.parquet_output_directory_path = QLineEdit(config["output_directory"])
        self.parquet_output_directory_browse = QPushButton("Browse...")
        self.parquet_output_directory_browse.clicked.connect(self._browse_parquet_output_directory)
        self.parquet_output_directory_path.editingFinished.connect(self._save_parquet_output_directory)
        parquet_output_row = QHBoxLayout()
        parquet_output_row.addWidget(self.parquet_output_directory_path)
        parquet_output_row.addWidget(self.parquet_output_directory_browse)
        parquet_layout.addRow("Output Parquet Folder:", parquet_output_row)

        self.parquet_output_filename_input = QLineEdit(config["output_filename"])
        self.parquet_output_filename_input.setPlaceholderText("e.g. entire-history-logs")
        self.parquet_output_filename_input.editingFinished.connect(self._save_parquet_output_filename)
        parquet_layout.addRow("Output Filename:", self.parquet_output_filename_input)

        parquet_group.setLayout(parquet_layout)
        layout.addWidget(parquet_group)

        self.run_parquet_conversion_button = QPushButton("Run Parquet Conversion")
        self.run_parquet_conversion_button.clicked.connect(self._run_parquet_conversion)
        layout.addWidget(self.run_parquet_conversion_button)

        csv_group = QGroupBox("Single File CSV Conversion")
        csv_layout = QFormLayout()

        self.csv_input_file_path = QLineEdit(config["csv_input_file"])
        self.csv_input_file_browse = QPushButton("Browse...")
        self.csv_input_file_browse.clicked.connect(self._browse_csv_input_file)
        self.csv_input_file_path.editingFinished.connect(self._save_csv_input_file)
        csv_input_row = QHBoxLayout()
        csv_input_row.addWidget(self.csv_input_file_path)
        csv_input_row.addWidget(self.csv_input_file_browse)
        csv_layout.addRow("Input File:", csv_input_row)

        default_csv_output = config["csv_output_file"] or self._default_csv_output_path(config)
        self.csv_output_file_path = QLineEdit(default_csv_output)
        self.csv_output_file_browse = QPushButton("Browse...")
        self.csv_output_file_browse.clicked.connect(self._browse_csv_output_file)
        self.csv_output_file_path.editingFinished.connect(self._save_csv_output_file)
        csv_output_row = QHBoxLayout()
        csv_output_row.addWidget(self.csv_output_file_path)
        csv_output_row.addWidget(self.csv_output_file_browse)
        csv_layout.addRow("Output CSV File:", csv_output_row)

        csv_group.setLayout(csv_layout)
        layout.addWidget(csv_group)

        self.run_csv_conversion_button = QPushButton("Convert File to CSV")
        self.run_csv_conversion_button.clicked.connect(self._run_csv_conversion)
        layout.addWidget(self.run_csv_conversion_button)

        breakdown_group = QGroupBox("Workload Breakdown")
        breakdown_layout = QFormLayout()

        self.breakdown_parquet_file_path = QLineEdit(config["breakdown_parquet_file"])
        self.breakdown_parquet_file_browse = QPushButton("Browse...")
        self.breakdown_parquet_file_browse.clicked.connect(self._browse_breakdown_parquet_file)
        self.breakdown_parquet_file_path.editingFinished.connect(self._save_breakdown_parquet_file)
        breakdown_input_row = QHBoxLayout()
        breakdown_input_row.addWidget(self.breakdown_parquet_file_path)
        breakdown_input_row.addWidget(self.breakdown_parquet_file_browse)
        breakdown_layout.addRow("Parquet File:", breakdown_input_row)

        breakdown_group.setLayout(breakdown_layout)
        layout.addWidget(breakdown_group)

        self.run_breakdown_button = QPushButton("Run Workload Breakdown")
        self.run_breakdown_button.clicked.connect(self._run_workload_breakdown)
        layout.addWidget(self.run_breakdown_button)

        filter_group = QGroupBox("Filter to Node Family")
        filter_layout = QFormLayout()

        self.filter_input_parquet_path = QLineEdit(config["filter_input_parquet_file"])
        self.filter_input_parquet_browse = QPushButton("Browse...")
        self.filter_input_parquet_browse.clicked.connect(self._browse_filter_input_parquet)
        self.filter_input_parquet_path.editingFinished.connect(self._save_filter_input_parquet)
        filter_input_row = QHBoxLayout()
        filter_input_row.addWidget(self.filter_input_parquet_path)
        filter_input_row.addWidget(self.filter_input_parquet_browse)
        filter_layout.addRow("Input Parquet File:", filter_input_row)

        self.filter_node_family_input = QLineEdit(config["filter_node_family"])
        self.filter_node_family_input.setPlaceholderText("e.g. ruby")
        self.filter_node_family_input.editingFinished.connect(self._save_filter_node_family)
        filter_layout.addRow("Node Family:", self.filter_node_family_input)

        self.filter_output_directory_path = QLineEdit(config["filter_output_directory"])
        self.filter_output_directory_browse = QPushButton("Browse...")
        self.filter_output_directory_browse.clicked.connect(self._browse_filter_output_directory)
        self.filter_output_directory_path.editingFinished.connect(self._save_filter_output_directory)
        filter_output_dir_row = QHBoxLayout()
        filter_output_dir_row.addWidget(self.filter_output_directory_path)
        filter_output_dir_row.addWidget(self.filter_output_directory_browse)
        filter_layout.addRow("Output Folder:", filter_output_dir_row)

        self.filter_output_filename_input = QLineEdit(config["filter_output_filename"])
        self.filter_output_filename_input.setPlaceholderText("e.g. ruby-only-jobs")
        self.filter_output_filename_input.editingFinished.connect(self._save_filter_output_filename)
        filter_layout.addRow("Output Filename:", self.filter_output_filename_input)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        self.run_filter_button = QPushButton("Filter to Node Family")
        self.run_filter_button.clicked.connect(self._run_filter_node_family)
        layout.addWidget(self.run_filter_button)

        single_node_group = QGroupBox("Filter to Single Node Jobs")
        single_node_layout = QFormLayout()

        self.single_node_input_parquet_path = QLineEdit(config["single_node_input_parquet_file"])
        self.single_node_input_parquet_browse = QPushButton("Browse...")
        self.single_node_input_parquet_browse.clicked.connect(self._browse_single_node_input_parquet)
        self.single_node_input_parquet_path.editingFinished.connect(self._save_single_node_input_parquet)
        single_node_input_row = QHBoxLayout()
        single_node_input_row.addWidget(self.single_node_input_parquet_path)
        single_node_input_row.addWidget(self.single_node_input_parquet_browse)
        single_node_layout.addRow("Input Parquet File:", single_node_input_row)

        self.single_node_output_directory_path = QLineEdit(config["single_node_output_directory"])
        self.single_node_output_directory_browse = QPushButton("Browse...")
        self.single_node_output_directory_browse.clicked.connect(self._browse_single_node_output_directory)
        self.single_node_output_directory_path.editingFinished.connect(self._save_single_node_output_directory)
        single_node_output_dir_row = QHBoxLayout()
        single_node_output_dir_row.addWidget(self.single_node_output_directory_path)
        single_node_output_dir_row.addWidget(self.single_node_output_directory_browse)
        single_node_layout.addRow("Output Folder:", single_node_output_dir_row)

        self.single_node_output_filename_input = QLineEdit(config["single_node_output_filename"])
        self.single_node_output_filename_input.setPlaceholderText("e.g. single-node-jobs")
        self.single_node_output_filename_input.editingFinished.connect(self._save_single_node_output_filename)
        single_node_layout.addRow("Output Filename:", self.single_node_output_filename_input)

        single_node_group.setLayout(single_node_layout)
        layout.addWidget(single_node_group)

        self.run_single_node_filter_button = QPushButton("Filter to Single Node Jobs")
        self.run_single_node_filter_button.clicked.connect(self._run_filter_single_node)
        layout.addWidget(self.run_single_node_filter_button)

        weights_group = QGroupBox("Resource Weights")
        weights_layout = QFormLayout()

        self.weights_parquet_file_path = QLineEdit(config["weights_parquet_file"])
        self.weights_parquet_file_browse = QPushButton("Browse...")
        self.weights_parquet_file_browse.clicked.connect(self._browse_weights_parquet_file)
        self.weights_parquet_file_path.editingFinished.connect(self._save_weights_parquet_file)
        weights_parquet_row = QHBoxLayout()
        weights_parquet_row.addWidget(self.weights_parquet_file_path)
        weights_parquet_row.addWidget(self.weights_parquet_file_browse)
        weights_layout.addRow("Input Parquet File:", weights_parquet_row)

        self.weights_output_directory_path = QLineEdit(config["weights_output_directory"])
        self.weights_output_directory_browse = QPushButton("Browse...")
        self.weights_output_directory_browse.clicked.connect(self._browse_weights_output_directory)
        self.weights_output_directory_path.editingFinished.connect(self._save_weights_output_directory)
        weights_output_row = QHBoxLayout()
        weights_output_row.addWidget(self.weights_output_directory_path)
        weights_output_row.addWidget(self.weights_output_directory_browse)
        weights_layout.addRow("Output Folder:", weights_output_row)

        weights_group.setLayout(weights_layout)
        layout.addWidget(weights_group)

        self.run_weights_button = QPushButton("Compute Resource Weights")
        self.run_weights_button.clicked.connect(self._run_resource_weights)
        layout.addWidget(self.run_weights_button)

        self.terminal_panel = TerminalPanel("Data Handling Terminal")
        layout.addWidget(self.terminal_panel)
        self.setLayout(layout)

    def _load_conversion_config(self) -> dict[str, str]:
        config_path = Path(__file__).resolve().parents[2] / "backend" / "data_handling" / "config.json"
        default_config = {
            "input_directory": "",
            "output_directory": "",
            "output_filename": "slurm_logs",
            "csv_input_file": "",
            "csv_output_file": "",
            "breakdown_parquet_file": "",
            "filter_input_parquet_file": "",
            "filter_node_family": "",
            "filter_output_directory": "",
            "filter_output_filename": "",
            "weights_parquet_file": "",
            "weights_output_directory": "",
            "single_node_input_parquet_file": "",
            "single_node_output_directory": "",
            "single_node_output_filename": "",
        }

        try:
            with config_path.open("r", encoding="utf-8") as file:
                config = json.load(file)
            return {
                "input_directory": str(config.get("input_directory", "")),
                "output_directory": str(config.get("output_directory", "")),
                "output_filename": str(config.get("output_filename", "slurm_logs")),
                "csv_input_file": str(config.get("csv_input_file", "")),
                "csv_output_file": str(config.get("csv_output_file", "")),
                "breakdown_parquet_file": str(config.get("breakdown_parquet_file", "")),
                "filter_input_parquet_file": str(config.get("filter_input_parquet_file", "")),
                "filter_node_family": str(config.get("filter_node_family", "")),
                "filter_output_directory": str(config.get("filter_output_directory", "")),
                "filter_output_filename": str(config.get("filter_output_filename", "")),
                "weights_parquet_file": str(config.get("weights_parquet_file", "")),
                "weights_output_directory": str(config.get("weights_output_directory", "")),
                "single_node_input_parquet_file": str(config.get("single_node_input_parquet_file", "")),
                "single_node_output_directory": str(config.get("single_node_output_directory", "")),
                "single_node_output_filename": str(config.get("single_node_output_filename", "")),
            }
        except (OSError, json.JSONDecodeError):
            return default_config

    def _default_csv_output_path(self, config: dict[str, str]) -> str:
        output_directory = config.get("output_directory", "")
        output_filename = config.get("output_filename", "slurm_logs")

        if not output_directory:
            return f"{output_filename}.csv"
        return str(Path(output_directory) / f"{output_filename}.csv")

    def _browse_parquet_input_directory(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Input Logs Folder")
        if folder:
            self.parquet_input_directory_path.setText(folder)
            self._save_parquet_input_directory()

    def _browse_parquet_output_directory(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Parquet Folder")
        if folder:
            self.parquet_output_directory_path.setText(folder)
            self._save_parquet_output_directory()

    def _browse_csv_input_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Log File",
            "",
            "Text Files (*.txt);;All Files (*)",
        )
        if file_path:
            self.csv_input_file_path.setText(file_path)
            self._save_csv_input_file()

    def _browse_csv_output_file(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output CSV File",
            self.csv_output_file_path.text() or "output.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if file_path:
            self.csv_output_file_path.setText(file_path)
            self._save_csv_output_file()

    def _save_parquet_input_directory(self) -> None:
        self._set_config_value("input_directory", self.parquet_input_directory_path.text().strip())

    def _save_parquet_output_directory(self) -> None:
        self._set_config_value("output_directory", self.parquet_output_directory_path.text().strip())

    def _save_parquet_output_filename(self) -> None:
        self._set_config_value("output_filename", self.parquet_output_filename_input.text().strip())

    def _save_csv_input_file(self) -> None:
        self._set_config_value("csv_input_file", self.csv_input_file_path.text().strip())

    def _save_csv_output_file(self) -> None:
        self._set_config_value("csv_output_file", self.csv_output_file_path.text().strip())

    def _browse_breakdown_parquet_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Parquet File",
            "",
            "Parquet Files (*.parquet);;All Files (*)",
        )
        if file_path:
            self.breakdown_parquet_file_path.setText(file_path)
            self._save_breakdown_parquet_file()

    def _save_breakdown_parquet_file(self) -> None:
        self._set_config_value("breakdown_parquet_file", self.breakdown_parquet_file_path.text().strip())

    def _run_parquet_conversion(self) -> None:
        self._save_parquet_input_directory()
        self._save_parquet_output_directory()
        self._save_parquet_output_filename()

        self.run_parquet_conversion_button.setEnabled(False)
        terminal_stream = TerminalStream(self.terminal_panel)
        self.terminal_panel.append_text("\n[data-handling] Starting parquet conversion...\n")

        try:
            with redirect_stdout(terminal_stream), redirect_stderr(terminal_stream):
                config = load_config(self.data_handling_config_path)
                output_file = convert_logs(config)
                print(f"Done. Wrote parquet file: {output_file}")
        except Exception as exc:
            self.terminal_panel.append_text(f"\n[data-handling] Parquet conversion failed: {exc}\n")
            QMessageBox.critical(self, "Parquet Conversion Failed", str(exc))
            self.run_parquet_conversion_button.setEnabled(True)
            return

        self.terminal_panel.append_text("\n[data-handling] Parquet conversion completed successfully.\n")
        QMessageBox.information(self, "Parquet Conversion Complete", "Parquet conversion finished.")
        self.run_parquet_conversion_button.setEnabled(True)

    def _run_csv_conversion(self) -> None:
        self._save_csv_input_file()
        self._save_csv_output_file()

        input_file = self.csv_input_file_path.text().strip()
        output_file = self.csv_output_file_path.text().strip()
        if not input_file or not output_file:
            QMessageBox.warning(
                self, "Missing Fields", "Please provide both input file and output CSV path."
            )
            return

        self.run_csv_conversion_button.setEnabled(False)
        terminal_stream = TerminalStream(self.terminal_panel)
        self.terminal_panel.append_text("\n[data-handling] Starting CSV conversion...\n")

        try:
            with redirect_stdout(terminal_stream), redirect_stderr(terminal_stream):
                convert_to_csv(input_file, output_file)
        except Exception as exc:
            self.terminal_panel.append_text(f"\n[data-handling] CSV conversion failed: {exc}\n")
            QMessageBox.critical(self, "CSV Conversion Failed", str(exc))
            self.run_csv_conversion_button.setEnabled(True)
            return

        self.terminal_panel.append_text("\n[data-handling] CSV conversion completed successfully.\n")
        QMessageBox.information(self, "CSV Conversion Complete", "CSV conversion finished.")
        self.run_csv_conversion_button.setEnabled(True)

    def _run_workload_breakdown(self) -> None:
        self._save_breakdown_parquet_file()

        parquet_file = self.breakdown_parquet_file_path.text().strip()
        if not parquet_file:
            QMessageBox.warning(self, "Missing File", "Please provide a parquet file path.")
            return

        nodes_json = Path(__file__).resolve().parents[2] / "backend" / "data_handling" / "nodes.json"
        nodes_config_path = str(nodes_json) if nodes_json.exists() else None

        self.run_breakdown_button.setEnabled(False)
        terminal_stream = TerminalStream(self.terminal_panel)
        self.terminal_panel.append_text("\n[data-handling] Running workload breakdown...\n")

        try:
            with redirect_stdout(terminal_stream), redirect_stderr(terminal_stream):
                workload_breakdown(parquet_file, nodes_config_path)
        except Exception as exc:
            self.terminal_panel.append_text(f"\n[data-handling] Workload breakdown failed: {exc}\n")
            QMessageBox.critical(self, "Workload Breakdown Failed", str(exc))
            self.run_breakdown_button.setEnabled(True)
            return

        self.terminal_panel.append_text("\n[data-handling] Workload breakdown complete.\n")
        self.run_breakdown_button.setEnabled(True)

    def _browse_weights_parquet_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Parquet File",
            "",
            "Parquet Files (*.parquet);;All Files (*)",
        )
        if file_path:
            self.weights_parquet_file_path.setText(file_path)
            self._save_weights_parquet_file()

    def _browse_weights_output_directory(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.weights_output_directory_path.setText(folder)
            self._save_weights_output_directory()

    def _save_weights_parquet_file(self) -> None:
        self._set_config_value("weights_parquet_file", self.weights_parquet_file_path.text().strip())

    def _save_weights_output_directory(self) -> None:
        self._set_config_value("weights_output_directory", self.weights_output_directory_path.text().strip())

    def _run_resource_weights(self) -> None:
        self._save_weights_parquet_file()
        self._save_weights_output_directory()

        parquet_file = self.weights_parquet_file_path.text().strip()
        output_dir = self.weights_output_directory_path.text().strip()
        if not parquet_file or not output_dir:
            QMessageBox.warning(
                self, "Missing Fields", "Please provide both an input parquet file and an output folder."
            )
            return

        config = self._load_conversion_config()
        nodes_config_path = config.get("nodes_config_file", "")
        if not nodes_config_path:
            nodes_config_path = str(
                Path(__file__).resolve().parents[2] / "backend" / "slurm_simulator" / "nodes.json"
            )

        output_file = str(Path(output_dir) / "resource_weights.json")

        self.run_weights_button.setEnabled(False)
        terminal_stream = TerminalStream(self.terminal_panel)
        self.terminal_panel.append_text("\n[data-handling] Computing resource weights...\n")

        try:
            with redirect_stdout(terminal_stream), redirect_stderr(terminal_stream):
                weights = compute_resource_weights(parquet_file, nodes_config_path)
                save_weights(weights, output_file)
        except Exception as exc:
            self.terminal_panel.append_text(f"\n[data-handling] Resource weights failed: {exc}\n")
            QMessageBox.critical(self, "Resource Weights Failed", str(exc))
            self.run_weights_button.setEnabled(True)
            return

        self.terminal_panel.append_text("\n[data-handling] Resource weights computed successfully.\n")
        QMessageBox.information(self, "Resource Weights Complete", f"Weights saved to:\n{output_file}")
        self.run_weights_button.setEnabled(True)

    def _browse_filter_input_parquet(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Parquet File",
            "",
            "Parquet Files (*.parquet);;All Files (*)",
        )
        if file_path:
            self.filter_input_parquet_path.setText(file_path)
            self._save_filter_input_parquet()

    def _browse_filter_output_directory(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.filter_output_directory_path.setText(folder)
            self._save_filter_output_directory()

    def _save_filter_input_parquet(self) -> None:
        self._set_config_value("filter_input_parquet_file", self.filter_input_parquet_path.text().strip())

    def _save_filter_node_family(self) -> None:
        self._set_config_value("filter_node_family", self.filter_node_family_input.text().strip())

    def _save_filter_output_directory(self) -> None:
        self._set_config_value("filter_output_directory", self.filter_output_directory_path.text().strip())

    def _save_filter_output_filename(self) -> None:
        self._set_config_value("filter_output_filename", self.filter_output_filename_input.text().strip())

    def _run_filter_node_family(self) -> None:
        self._save_filter_input_parquet()
        self._save_filter_node_family()
        self._save_filter_output_directory()
        self._save_filter_output_filename()

        parquet_file = self.filter_input_parquet_path.text().strip()
        family = self.filter_node_family_input.text().strip()
        output_dir = self.filter_output_directory_path.text().strip()
        output_filename = self.filter_output_filename_input.text().strip()

        if not parquet_file or not family or not output_dir or not output_filename:
            QMessageBox.warning(
                self, "Missing Fields", "Please fill in all fields: input file, node family, output folder, and filename."
            )
            return

        self.run_filter_button.setEnabled(False)
        terminal_stream = TerminalStream(self.terminal_panel)
        self.terminal_panel.append_text(f"\n[data-handling] Filtering to node family '{family}'...\n")

        try:
            with redirect_stdout(terminal_stream), redirect_stderr(terminal_stream):
                out_path = filter_to_node_family(parquet_file, family, output_dir, output_filename)
        except Exception as exc:
            self.terminal_panel.append_text(f"\n[data-handling] Filter failed: {exc}\n")
            QMessageBox.critical(self, "Filter Failed", str(exc))
            self.run_filter_button.setEnabled(True)
            return

        self.terminal_panel.append_text("\n[data-handling] Filter complete.\n")
        QMessageBox.information(self, "Filter Complete", f"Filtered parquet saved to:\n{out_path}")
        self.run_filter_button.setEnabled(True)

    def _browse_single_node_input_parquet(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Parquet File", "", "Parquet Files (*.parquet);;All Files (*)"
        )
        if file_path:
            self.single_node_input_parquet_path.setText(file_path)
            self._save_single_node_input_parquet()

    def _browse_single_node_output_directory(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.single_node_output_directory_path.setText(folder)
            self._save_single_node_output_directory()

    def _save_single_node_input_parquet(self) -> None:
        self._set_config_value("single_node_input_parquet_file", self.single_node_input_parquet_path.text().strip())

    def _save_single_node_output_directory(self) -> None:
        self._set_config_value("single_node_output_directory", self.single_node_output_directory_path.text().strip())

    def _save_single_node_output_filename(self) -> None:
        self._set_config_value("single_node_output_filename", self.single_node_output_filename_input.text().strip())

    def _run_filter_single_node(self) -> None:
        self._save_single_node_input_parquet()
        self._save_single_node_output_directory()
        self._save_single_node_output_filename()

        parquet_file = self.single_node_input_parquet_path.text().strip()
        output_dir = self.single_node_output_directory_path.text().strip()
        output_filename = self.single_node_output_filename_input.text().strip()

        if not parquet_file or not output_dir or not output_filename:
            QMessageBox.warning(
                self, "Missing Fields", "Please fill in all fields: input file, output folder, and filename."
            )
            return

        self.run_single_node_filter_button.setEnabled(False)
        terminal_stream = TerminalStream(self.terminal_panel)
        self.terminal_panel.append_text("\n[data-handling] Filtering to single node jobs...\n")

        try:
            with redirect_stdout(terminal_stream), redirect_stderr(terminal_stream):
                out_path = filter_to_single_node(parquet_file, output_dir, output_filename)
        except Exception as exc:
            self.terminal_panel.append_text(f"\n[data-handling] Filter failed: {exc}\n")
            QMessageBox.critical(self, "Filter Failed", str(exc))
            self.run_single_node_filter_button.setEnabled(True)
            return

        self.terminal_panel.append_text("\n[data-handling] Filter complete.\n")
        QMessageBox.information(self, "Filter Complete", f"Filtered parquet saved to:\n{out_path}")
        self.run_single_node_filter_button.setEnabled(True)

    def _set_config_value(self, key: str, value: str) -> None:
        config = self._load_conversion_config()
        config[key] = value
        with self.data_handling_config_path.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
            file.write("\n")
