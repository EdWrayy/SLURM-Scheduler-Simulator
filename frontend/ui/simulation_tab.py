import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
from backend.slurm_simulator.run_simulation import run_simulation
from frontend.ui.settings_store import get_output_root_directory
from frontend.ui.terminal_panel import TerminalPanel


class SimulationWorker(QThread):
    output = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, config):
        super().__init__()
        self._config = config

    def run(self):
        stream = _WorkerStream(self.output)
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                run_simulation(self._config)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class _WorkerStream:
    def __init__(self, signal):
        self._signal = signal

    def write(self, text):
        if text:
            self._signal.emit(text)
        return len(text) if text else 0

    def flush(self):
        pass


class SimulationTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.simulator_dir = Path(__file__).resolve().parents[2] / "backend" / "slurm_simulator"
        self.simulator_config_path = self.simulator_dir / "config.json"
        self.nodes_json_target = self.simulator_dir / "nodes.json"
        self.power_json_target = self.simulator_dir / "power_constants.json"
        self.weights_json_target = self.simulator_dir / "resource_weights.json"
        self.simulation_config = self._load_simulation_config()

        layout = QVBoxLayout()

        title = QLabel("Simulation Configuration")
        layout.addWidget(title)

        files_group = QGroupBox("Input Files")
        files_layout = QFormLayout()

        self.events_parquet_path = QLineEdit(
            self.simulation_config.get("input_events", "")
        )
        self.events_parquet_browse = QPushButton("Browse...")
        self.events_parquet_browse.clicked.connect(self._browse_events_parquet)
        self.events_parquet_path.editingFinished.connect(self._apply_events_parquet_from_textbox)
        events_parquet_row = QHBoxLayout()
        events_parquet_row.addWidget(self.events_parquet_path)
        events_parquet_row.addWidget(self.events_parquet_browse)
        files_layout.addRow("Input Events Parquet:", events_parquet_row)

        self.nodes_json_path = QLineEdit(
            self.simulation_config.get("ui_nodes_json_source", str(self.nodes_json_target))
        )
        self.nodes_json_browse = QPushButton("Browse...")
        self.nodes_json_browse.clicked.connect(self._browse_nodes_json)
        self.nodes_json_path.editingFinished.connect(self._apply_nodes_json_from_textbox)
        nodes_row = QHBoxLayout()
        nodes_row.addWidget(self.nodes_json_path)
        nodes_row.addWidget(self.nodes_json_browse)
        files_layout.addRow("Nodes JSON:", nodes_row)

        self.power_json_path = QLineEdit(
            self.simulation_config.get("ui_power_json_source", str(self.power_json_target))
        )
        self.power_json_browse = QPushButton("Browse...")
        self.power_json_browse.clicked.connect(self._browse_power_json)
        self.power_json_path.editingFinished.connect(self._apply_power_json_from_textbox)
        power_row = QHBoxLayout()
        power_row.addWidget(self.power_json_path)
        power_row.addWidget(self.power_json_browse)
        files_layout.addRow("Power JSON:", power_row)

        self.weights_json_path = QLineEdit(
            self.simulation_config.get("ui_weights_json_source", str(self.weights_json_target))
        )
        self.weights_json_browse = QPushButton("Browse...")
        self.weights_json_browse.clicked.connect(self._browse_weights_json)
        self.weights_json_path.editingFinished.connect(self._apply_weights_json_from_textbox)
        weights_row = QHBoxLayout()
        weights_row.addWidget(self.weights_json_path)
        weights_row.addWidget(self.weights_json_browse)
        files_layout.addRow("Resource Weights JSON:", weights_row)

        files_group.setLayout(files_layout)
        layout.addWidget(files_group)

        input_group = QGroupBox("Output Folders (Fixed by General Settings)")
        input_layout = QFormLayout()

        self.events_input_path = QLineEdit(self.simulation_config.get("output_events_directory", ""))
        self.events_input_path.setReadOnly(True)
        self.events_input_browse = QPushButton("Browse...")
        self.events_input_browse.setEnabled(False)
        events_row = QHBoxLayout()
        events_row.addWidget(self.events_input_path)
        events_row.addWidget(self.events_input_browse)
        input_layout.addRow("Output Events Folder:", events_row)

        self.nodes_input_path = QLineEdit(self.simulation_config.get("output_nodes_directory", ""))
        self.nodes_input_path.setReadOnly(True)
        self.nodes_input_browse = QPushButton("Browse...")
        self.nodes_input_browse.setEnabled(False)
        nodes_row = QHBoxLayout()
        nodes_row.addWidget(self.nodes_input_path)
        nodes_row.addWidget(self.nodes_input_browse)
        input_layout.addRow("Output Nodes Folder:", nodes_row)

        self.debug_input_path = QLineEdit(self.simulation_config.get("output_debug_directory", ""))
        self.debug_input_path.setReadOnly(True)
        self.debug_input_browse = QPushButton("Browse...")
        self.debug_input_browse.setEnabled(False)
        debug_row = QHBoxLayout()
        debug_row.addWidget(self.debug_input_path)
        debug_row.addWidget(self.debug_input_browse)
        input_layout.addRow("Output Debug Folder:", debug_row)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        config_group = QGroupBox("Simulation Parameters")
        config_layout = QFormLayout()

        self.node_selection_combo = QComboBox()
        self.node_selection_combo.addItems(
            [
                "NaiveFeasible",
                "CopyRealNodeSelection",
                "LoadSpreading",
                "CPU_Best_Fit",
                "GPU_Best_Fit",
                "Manhattan_Slack_Best_Fit",
                "Dominant_Resource_Best_Fit",
                "Workload_Aware_Weighted_Manhattan_Slack",
                "Workload_Aware_Weighted_Dominant_Resource",
            ]
        )
        self._set_combo_selection(
            self.node_selection_combo,
            self.simulation_config.get("node_selection_strategy", "NaiveFeasible"),
        )
        self.node_selection_combo.currentTextChanged.connect(self._save_node_selection_strategy)
        self.node_selection_combo.currentTextChanged.connect(self._refresh_fixed_output_directories)
        config_layout.addRow("Node Selection Strategy:", self.node_selection_combo)

        self.resource_distribution_combo = QComboBox()
        self.resource_distribution_combo.addItems(["GreedyBlockFill", "EvenSplit"])
        self._set_combo_selection(
            self.resource_distribution_combo,
            self.simulation_config.get("resource_distribution_strategy", "GreedyBlockFill"),
        )
        self.resource_distribution_combo.currentTextChanged.connect(
            self._save_resource_distribution_strategy
        )
        config_layout.addRow("Resource Distribution Strategy:", self.resource_distribution_combo)

        self.power_model_combo = QComboBox()
        self.power_model_combo.addItems(
            ["LinearWithSleepPowerModel", "LinearWithIdlePowerModel", "ActiveOnlyPowerModel"]
        )
        self._set_combo_selection(
            self.power_model_combo,
            self.simulation_config.get("power_model", "LinearWithSleepPowerModel"),
        )
        self.power_model_combo.currentTextChanged.connect(self._save_power_model)
        self.power_model_combo.currentTextChanged.connect(self._refresh_fixed_output_directories)
        config_layout.addRow("Power Model:", self.power_model_combo)

        self.debug_output_checkbox = QCheckBox("Write failed-placement debug files")
        self.debug_output_checkbox.setChecked(self._config_bool("enable_debug_output", True))
        self.debug_output_checkbox.toggled.connect(self._save_enable_debug_output)
        config_layout.addRow("Enable Debug Output:", self.debug_output_checkbox)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        self.run_simulation_button = QPushButton("Run Simulation")
        self.run_simulation_button.clicked.connect(self._run_simulation)
        layout.addWidget(self.run_simulation_button)

        self.terminal_panel = TerminalPanel("Simulation Terminal")
        layout.addWidget(self.terminal_panel)
        self.setLayout(layout)
        self._refresh_fixed_output_directories()

    def _browse_events_parquet(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Events Parquet",
            self.events_parquet_path.text(),
            "Parquet Files (*.parquet);;All Files (*)",
        )
        if not file_path:
            return
        self.events_parquet_path.setText(file_path)
        self._set_config_value("input_events", file_path)

    def _apply_events_parquet_from_textbox(self) -> None:
        selected = self.events_parquet_path.text().strip()
        if selected:
            self._set_config_value("input_events", selected)

    def _browse_nodes_json(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Nodes JSON",
            self.nodes_json_path.text(),
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return
        self.nodes_json_path.setText(file_path)
        self._apply_selected_json_to_target(file_path, self.nodes_json_target, "ui_nodes_json_source")

    def _browse_power_json(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Power Constants JSON",
            self.power_json_path.text(),
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return
        self.power_json_path.setText(file_path)
        self._apply_selected_json_to_target(file_path, self.power_json_target, "ui_power_json_source")

    def _apply_nodes_json_from_textbox(self) -> None:
        selected = self.nodes_json_path.text().strip()
        if selected:
            self._apply_selected_json_to_target(selected, self.nodes_json_target, "ui_nodes_json_source")

    def _apply_power_json_from_textbox(self) -> None:
        selected = self.power_json_path.text().strip()
        if selected:
            self._apply_selected_json_to_target(selected, self.power_json_target, "ui_power_json_source")

    def _browse_weights_json(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Resource Weights JSON",
            self.weights_json_path.text(),
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return
        self.weights_json_path.setText(file_path)
        self._apply_selected_json_to_target(file_path, self.weights_json_target, "ui_weights_json_source")

    def _apply_weights_json_from_textbox(self) -> None:
        selected = self.weights_json_path.text().strip()
        if selected:
            self._apply_selected_json_to_target(selected, self.weights_json_target, "ui_weights_json_source")

    def _save_events_output_directory(self) -> None:
        directory = self.events_input_path.text().strip()
        self._set_config_value("output_events_directory", directory)

    def _save_nodes_output_directory(self) -> None:
        directory = self.nodes_input_path.text().strip()
        self._set_config_value("output_nodes_directory", directory)

    def _save_debug_output_directory(self) -> None:
        directory = self.debug_input_path.text().strip()
        self._set_config_value("output_debug_directory", directory)

    def _save_node_selection_strategy(self, value: str) -> None:
        self._set_config_value("node_selection_strategy", value)

    def _save_resource_distribution_strategy(self, value: str) -> None:
        self._set_config_value("resource_distribution_strategy", value)

    def _save_power_model(self, value: str) -> None:
        self._set_config_value("power_model", value)

    def _save_enable_debug_output(self, enabled: bool) -> None:
        self._set_config_value("enable_debug_output", bool(enabled))

    def _run_simulation(self) -> None:
        # Ensure manual path edits are persisted before running.
        self._refresh_fixed_output_directories()
        self._apply_nodes_json_from_textbox()
        self._apply_power_json_from_textbox()
        self._apply_weights_json_from_textbox()
        self._save_node_selection_strategy(self.node_selection_combo.currentText())
        self._save_resource_distribution_strategy(self.resource_distribution_combo.currentText())
        self._save_power_model(self.power_model_combo.currentText())
        self._save_enable_debug_output(self.debug_output_checkbox.isChecked())

        self.run_simulation_button.setEnabled(False)
        self.terminal_panel.append_text("\n[simulation] Starting run...\n")

        config = load_config(self.simulator_config_path)
        self._worker = SimulationWorker(config)
        self._simulation_failed = False
        self._worker.output.connect(self.terminal_panel.append_text)
        self._worker.error.connect(self._on_simulation_error)
        self._worker.finished.connect(self._on_simulation_finished)
        self._worker.start()

    def _on_simulation_error(self, message: str) -> None:
        self._simulation_failed = True
        self.terminal_panel.append_text(f"\n[simulation] Failed: {message}\n")
        QMessageBox.critical(self, "Simulation Failed", message)

    def _on_simulation_finished(self) -> None:
        self.run_simulation_button.setEnabled(True)
        if not self._simulation_failed:
            self.terminal_panel.append_text("\n[simulation] Completed successfully.\n")
            QMessageBox.information(self, "Simulation Complete", "Simulation finished successfully.")

    def _apply_selected_json_to_target(
        self, selected_path: str, target_path: Path, ui_source_key: str
    ) -> None:
        source = Path(selected_path)
        if not source.exists() or not source.is_file():
            QMessageBox.warning(self, "Invalid File", f"JSON file not found:\n{source}")
            return

        try:
            with source.open("r", encoding="utf-8-sig") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Invalid JSON", f"Could not read JSON file:\n{source}\n\n{exc}")
            return

        try:
            with target_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)
                file.write("\n")
        except OSError as exc:
            QMessageBox.warning(
                self, "Write Failed", f"Could not update simulator JSON:\n{target_path}\n\n{exc}"
            )
            return

        # Keep simulator config pointed at the local copied files.
        self._set_config_value(ui_source_key, str(source))
        if target_path == self.nodes_json_target:
            self._set_config_value("nodes_config", target_path.name)
        elif target_path == self.power_json_target:
            self._set_config_value("power_constants_config", target_path.name)

    def _set_combo_selection(self, combo: QComboBox, value: str) -> None:
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _load_simulation_config(self) -> dict:
        try:
            with self.simulator_config_path.open("r", encoding="utf-8") as file:
                config = json.load(file)
            if isinstance(config, dict):
                return config
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _set_config_value(self, key: str, value) -> None:
        self.simulation_config[key] = value
        self._save_simulation_config()

    def _config_bool(self, key: str, default: bool) -> bool:
        value = self.simulation_config.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _save_simulation_config(self) -> None:
        with self.simulator_config_path.open("w", encoding="utf-8") as file:
            json.dump(self.simulation_config, file, indent=2)
            file.write("\n")

    def _refresh_fixed_output_directories(self) -> None:
        root = get_output_root_directory().strip()
        if not root:
            return

        power_model = self.power_model_combo.currentText().strip() or str(
            self.simulation_config.get("power_model", "")
        )
        node_strategy = self.node_selection_combo.currentText().strip() or str(
            self.simulation_config.get("node_selection_strategy", "")
        )

        base = Path(root) / "simulation" / power_model / node_strategy
        events_dir = str(base / "events")
        nodes_dir = str(base / "nodes")
        debug_dir = str(base / "debug")

        self.events_input_path.setText(events_dir)
        self.nodes_input_path.setText(nodes_dir)
        self.debug_input_path.setText(debug_dir)
        self._save_events_output_directory()
        self._save_nodes_output_directory()
        self._save_debug_output_directory()
