import json
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from frontend.ui.settings_store import get_output_root_directory, set_output_root_directory


class GeneralSettingsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.project_root = Path(__file__).resolve().parents[2]
        self.sim_config_path = self.project_root / "backend" / "slurm_simulator" / "config.json"
        self.sim_analysis_config_path = self.project_root / "backend" / "simulation_analysis" / "config.json"
        self.meta_config_path = self.project_root / "backend" / "meta_analysis" / "config.json"

        layout = QVBoxLayout()
        layout.addWidget(QLabel("General Settings"))

        root_group = QGroupBox("Output Root Directory")
        root_layout = QFormLayout()

        self.output_root_path = QLineEdit(get_output_root_directory())
        self.output_root_path.editingFinished.connect(self._apply_output_root_from_text)
        self.output_root_browse = QPushButton("Browse...")
        self.output_root_browse.clicked.connect(self._browse_output_root)
        root_row = QHBoxLayout()
        root_row.addWidget(self.output_root_path)
        root_row.addWidget(self.output_root_browse)
        root_layout.addRow("Root Output Folder:", root_row)

        root_group.setLayout(root_layout)
        layout.addWidget(root_group)

        preview_group = QGroupBox("Fixed Output Paths (Within Root)")
        preview_layout = QFormLayout()

        self.simulation_output_preview = QLineEdit("")
        self.simulation_output_preview.setReadOnly(True)
        preview_layout.addRow("Simulation:", self.simulation_output_preview)

        self.sim_analysis_output_preview = QLineEdit("")
        self.sim_analysis_output_preview.setReadOnly(True)
        preview_layout.addRow("Simulation Analysis:", self.sim_analysis_output_preview)

        self.meta_analysis_output_preview = QLineEdit("")
        self.meta_analysis_output_preview.setReadOnly(True)
        preview_layout.addRow("Meta Analysis:", self.meta_analysis_output_preview)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        note = QLabel(
            "Simulation, Simulation Analysis, and Meta Analysis outputs are fixed under this root."
        )
        layout.addWidget(note)

        layout.addStretch(1)
        self.setLayout(layout)

        self._refresh_previews(self.output_root_path.text().strip())

    def _browse_output_root(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Root Output Folder")
        if folder:
            self.output_root_path.setText(folder)
            self._apply_output_root(folder)

    def _apply_output_root_from_text(self) -> None:
        self._apply_output_root(self.output_root_path.text().strip())

    def _apply_output_root(self, root_folder: str) -> None:
        set_output_root_directory(root_folder)
        self._refresh_previews(root_folder)
        self._sync_backend_output_paths(root_folder)

    def _refresh_previews(self, root_folder: str) -> None:
        if not root_folder:
            self.simulation_output_preview.setText("<root>/simulation/<power_model>/<node_selection>")
            self.sim_analysis_output_preview.setText("<root>/simulation_analysis/<name>")
            self.meta_analysis_output_preview.setText("<root>/meta_analysis/<name>")
            return

        root = Path(root_folder)
        self.simulation_output_preview.setText(
            str(root / "simulation" / "<power_model>" / "<node_selection>")
        )
        self.sim_analysis_output_preview.setText(str(root / "simulation_analysis" / "<name>"))
        self.meta_analysis_output_preview.setText(str(root / "meta_analysis" / "<name>"))

    def _sync_backend_output_paths(self, root_folder: str) -> None:
        if not root_folder:
            return

        root = Path(root_folder)

        sim_config = self._load_json_config(self.sim_config_path)
        power_model = str(sim_config.get("power_model", "LinearWithSleepPowerModel"))
        node_selection = str(sim_config.get("node_selection_strategy", "NaiveFirstFit"))
        sim_base = root / "simulation" / power_model / node_selection
        sim_config["output_events_directory"] = str(sim_base / "events")
        sim_config["output_nodes_directory"] = str(sim_base / "nodes")
        sim_config["output_debug_directory"] = str(sim_base / "debug")
        self._save_json_config(self.sim_config_path, sim_config)

        sim_analysis_config = self._load_json_config(self.sim_analysis_config_path)
        sim_analysis_leaf = self._current_leaf_name(
            sim_analysis_config.get("output_report_directory", ""),
            "analysis_run",
        )
        sim_analysis_config["output_report_directory"] = str(
            root / "simulation_analysis" / sim_analysis_leaf
        )
        self._save_json_config(self.sim_analysis_config_path, sim_analysis_config)

        meta_config = self._load_json_config(self.meta_config_path)
        meta_leaf = self._current_leaf_name(meta_config.get("output_directory", ""), "meta_run")
        meta_config["output_directory"] = str(root / "meta_analysis" / meta_leaf)
        self._save_json_config(self.meta_config_path, meta_config)

    def _current_leaf_name(self, output_path: str, fallback: str) -> str:
        if not output_path:
            return fallback
        try:
            leaf = Path(output_path).name
        except (TypeError, ValueError):
            return fallback
        if not leaf:
            return fallback
        if leaf in {"simulation_analysis", "meta_analysis"}:
            return fallback
        return leaf

    def _load_json_config(self, path: Path) -> dict:
        try:
            with path.open("r", encoding="utf-8") as file:
                config = json.load(file)
            if isinstance(config, dict):
                return config
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _save_json_config(self, path: Path, config: dict) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
            file.write("\n")
