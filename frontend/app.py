from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from frontend.ui.simulation_tab import SimulationTab
from frontend.ui.data_handling_tab import DataHandlingTab
from frontend.ui.simulation_analysis_tab import SimulationAnalysisTab
from frontend.ui.meta_analysis_tab import MetaAnalysisTab
from frontend.ui.general_settings_tab import GeneralSettingsTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SLURM Simulator Tools")
        self.resize(1000, 700)

        tabs = QTabWidget()
    
        tabs.addTab(SimulationTab(), "Run Simulation")
        tabs.addTab(SimulationAnalysisTab(), "Simulation Analysis")
        tabs.addTab(MetaAnalysisTab(), "Meta Analysis")
        tabs.addTab(DataHandlingTab(), "Data Handling")
        tabs.addTab(GeneralSettingsTab(), "General Settings")

        self.setCentralWidget(tabs)


def main() -> int:
    app = QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
