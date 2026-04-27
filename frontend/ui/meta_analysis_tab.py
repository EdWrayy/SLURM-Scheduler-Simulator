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
from backend.meta_analysis.__main__ import main as run_meta_analysis
from frontend.ui.settings_store import get_output_root_directory
from frontend.ui.terminal_panel import TerminalPanel, TerminalStream


class MetaAnalysisTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.meta_config_path = Path(__file__).resolve().parents[2] / "backend" / "meta_analysis" / "config.json"
        self.meta_config = self._load_meta_config()

        layout = QVBoxLayout()

        title = QLabel("Meta Analysis")
        layout.addWidget(title)

        config_group = QGroupBox("Meta Analysis Configuration")
        config_layout = QFormLayout()

        input_directory = self._load_input_directory()
        self.meta_analysis_input_path = QLineEdit(str(input_directory))
        self.meta_analysis_input_browse = QPushButton("Browse...")
        self.meta_analysis_input_browse.clicked.connect(self._browse_input_directory)
        self.meta_analysis_input_path.editingFinished.connect(self._save_input_directory)

        input_row = QHBoxLayout()
        input_row.addWidget(self.meta_analysis_input_path)
        input_row.addWidget(self.meta_analysis_input_browse)
        config_layout.addRow("Input JSON Folder:", input_row)

        self.output_name_input = QLineEdit(
            str(self.meta_config.get("ui_output_name", self._default_output_name_from_input()))
        )
        self.output_name_input.editingFinished.connect(self._save_output_name)
        config_layout.addRow("Output Name:", self.output_name_input)

        self.baseline_name_input = QLineEdit(
            str(self.meta_config.get("baseline_name", "Naive_Feasible"))
        )
        self.baseline_name_input.setPlaceholderText("e.g. Naive_Feasible")
        self.baseline_name_input.editingFinished.connect(self._save_baseline_name)
        config_layout.addRow("Baseline Filename:", self.baseline_name_input)

        self.meta_analysis_output_directory = QLineEdit("")
        self.meta_analysis_output_directory.setReadOnly(True)
        config_layout.addRow("Output Directory:", self.meta_analysis_output_directory)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        self.run_meta_analysis_button = QPushButton("Run Meta Analysis")
        self.run_meta_analysis_button.clicked.connect(self._run_meta_analysis)
        layout.addWidget(self.run_meta_analysis_button)

        self.terminal_panel = TerminalPanel("Meta Analysis Terminal")
        layout.addWidget(self.terminal_panel)
        self.setLayout(layout)
        self._refresh_output_directory()

    def _load_input_directory(self) -> Path:
        return Path(self.meta_config.get("input_directory", ""))

    def _browse_input_directory(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Input JSON Folder")
        if folder:
            self.meta_analysis_input_path.setText(folder)
            self._save_input_directory()

    def _save_input_directory(self) -> None:
        self._set_config_value("input_directory", self.meta_analysis_input_path.text().strip())
        if not self.meta_config.get("ui_output_name"):
            self.output_name_input.setText(self._default_output_name_from_input())
        self._refresh_output_directory()

    def _save_output_name(self) -> None:
        name = self.output_name_input.text().strip() or self._default_output_name_from_input()
        self.output_name_input.setText(name)
        self._set_config_value("ui_output_name", name)
        self._refresh_output_directory()

    def _save_baseline_name(self) -> None:
        self._set_config_value("baseline_name", self.baseline_name_input.text().strip())

    def _run_meta_analysis(self) -> None:
        self._save_input_directory()
        self._save_output_name()
        self._refresh_output_directory()

        self.run_meta_analysis_button.setEnabled(False)
        terminal_stream = TerminalStream(self.terminal_panel)
        self.terminal_panel.append_text("\n[meta-analysis] Starting run...\n")

        try:
            with redirect_stdout(terminal_stream), redirect_stderr(terminal_stream):
                # Load once to fail fast with the same behavior as other tabs.
                load_config(self.meta_config_path)
                run_meta_analysis()
        except Exception as exc:
            self.terminal_panel.append_text(f"\n[meta-analysis] Failed: {exc}\n")
            QMessageBox.critical(self, "Meta Analysis Failed", str(exc))
            self.run_meta_analysis_button.setEnabled(True)
            return

        self.terminal_panel.append_text("\n[meta-analysis] Completed successfully.\n")
        QMessageBox.information(self, "Meta Analysis Complete", "Meta analysis finished.")
        self.run_meta_analysis_button.setEnabled(True)

    def _load_meta_config(self) -> dict:
        try:
            with self.meta_config_path.open("r", encoding="utf-8") as file:
                config = json.load(file)
            if isinstance(config, dict):
                return config
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _set_config_value(self, key: str, value: str) -> None:
        self.meta_config[key] = value
        self._save_meta_config()

    def _save_meta_config(self) -> None:
        try:
            with self.meta_config_path.open("r", encoding="utf-8") as file:
                on_disk = json.load(file)
        except (OSError, json.JSONDecodeError):
            on_disk = {}
        on_disk.update(self.meta_config)
        self.meta_config = on_disk
        with self.meta_config_path.open("w", encoding="utf-8") as file:
            json.dump(self.meta_config, file, indent=2)
            file.write("\n")

    def _default_output_name_from_input(self) -> str:
        input_path = Path(str(self.meta_config.get("input_directory", "")).strip())
        if not input_path.parts:
            return "meta_run"

        parts = list(input_path.parts)
        if parts and parts[-1].lower() in {"inputs", "input", "json"}:
            parts = parts[:-1]

        if len(parts) >= 2:
            return f"{parts[-2]}_{parts[-1]}".replace(" ", "_")
        return parts[-1].replace(" ", "_")

    def _refresh_output_directory(self) -> None:
        root = get_output_root_directory().strip()
        if not root:
            self.meta_analysis_output_directory.setText("<Set output root in General Settings>")
            return

        output_name = self.output_name_input.text().strip() or "meta_run"
        output_dir = Path(root) / "meta_analysis" / output_name
        output_dir_str = str(output_dir)
        self.meta_analysis_output_directory.setText(output_dir_str)
        self._set_config_value("output_directory", output_dir_str)
