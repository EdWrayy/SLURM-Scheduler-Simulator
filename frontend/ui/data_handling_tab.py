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
from backend.data_handling.log_file_conversion import convert_logs
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

    def _set_config_value(self, key: str, value: str) -> None:
        config = self._load_conversion_config()
        config[key] = value
        with self.data_handling_config_path.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
            file.write("\n")
