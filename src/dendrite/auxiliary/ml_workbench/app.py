"""
Offline ML Workbench - Main Application

Unified interface for offline machine learning workflows:
- Study selection
- Model training
- Model evaluation
- Benchmarking
"""

import datetime
import sys
from typing import Any

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.design_tokens import BG_PANEL
from dendrite.gui.styles.widget_styles import LAYOUT, apply_app_styles
from dendrite.gui.widgets.common import PillNavigation
from dendrite.utils.logger_central import get_logger

logger = get_logger(__name__)


class AppState(QtCore.QObject):
    """Shared application state across all tabs."""

    study_changed = QtCore.pyqtSignal(object)
    model_trained = QtCore.pyqtSignal(str, object)
    benchmark_results_changed = QtCore.pyqtSignal(dict)
    validation_data_ready = QtCore.pyqtSignal(object)  # (continuous, times, labels) tuple

    def __init__(self):
        super().__init__()
        self._current_study = None
        self._trained_models: dict[str, Any] = {}
        self._benchmark_results: dict[str, dict[str, Any]] = {}
        self._validation_data = None  # Holdout data for evaluation

    @property
    def current_study(self):
        """Get currently selected study (study dict)."""
        return self._current_study

    def set_current_study(self, data):
        """Set current study and notify listeners."""
        self._current_study = data
        self.study_changed.emit(data)

    @property
    def trained_models(self) -> dict[str, Any]:
        return self._trained_models

    def add_trained_model(self, name: str, model):
        self._trained_models[name] = model
        self.model_trained.emit(name, model)

    @property
    def benchmark_results(self) -> dict[str, dict[str, Any]]:
        return self._benchmark_results

    def set_benchmark_result(self, model: str, best_params: dict[str, Any]):
        self._benchmark_results[model] = best_params
        self.benchmark_results_changed.emit(self._benchmark_results)

    def set_validation_data(self, data):
        """Set holdout validation data (continuous, event_times, event_labels)."""
        self._validation_data = data
        self.validation_data_ready.emit(data)

    @property
    def validation_data(self):
        """Get holdout validation data."""
        return self._validation_data


class TrainerApp(QtWidgets.QMainWindow):
    """Main application window for Offline ML Workbench."""

    def __init__(self):
        super().__init__()
        self._state = AppState()
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        self.setWindowTitle("Offline ML Workbench")
        self.setMinimumSize(1200, 800)
        self.resize(1600, 1000)

        central = QtWidgets.QWidget()
        central.setStyleSheet(f"""
            background: {BG_PANEL};
            QLabel {{ background: transparent; }}
        """)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(LAYOUT["spacing_md"], LAYOUT["spacing_md"], LAYOUT["spacing_md"], 0)
        layout.setSpacing(LAYOUT["spacing_md"])

        # Navigation tabs (PillNavigation like main window)
        self._tabs_nav = PillNavigation(
            tabs=[
                ("data", "Data"),
                ("training", "Training"),
                ("evaluation", "Evaluation"),
                ("benchmark", "Benchmark"),
            ],
            size="large",
            parent=self,
        )
        self._tabs_nav.section_changed.connect(self._on_tab_changed)
        layout.addWidget(self._tabs_nav)

        # Stacked widget for tab content
        self._tabs_stack = QtWidgets.QStackedWidget()
        layout.addWidget(self._tabs_stack, 1)

        self._create_tabs()
        self._status_bar = self.statusBar()
        self._status_bar.showMessage("Ready")

    def _create_tabs(self):
        from dendrite.auxiliary.ml_workbench.tabs import (
            BenchmarkTab,
            DataTab,
            EvaluationTab,
            TrainingTab,
        )

        self._data_tab = DataTab()
        self._tabs_stack.addWidget(self._data_tab)

        self._training_tab = TrainingTab()
        self._training_tab.set_app_state(self._state)
        self._tabs_stack.addWidget(self._training_tab)

        self._evaluation_tab = EvaluationTab()
        self._evaluation_tab.set_app_state(self._state)
        self._tabs_stack.addWidget(self._evaluation_tab)

        self._benchmark_tab = BenchmarkTab()
        self._benchmark_tab.set_app_state(self._state)
        self._tabs_stack.addWidget(self._benchmark_tab)

    def _on_tab_changed(self, section: str):
        """Handle navigation tab change."""
        idx = {"data": 0, "training": 1, "evaluation": 2, "benchmark": 3}
        self._tabs_stack.setCurrentIndex(idx.get(section, 0))

    def _connect_signals(self):
        self._data_tab.study_changed.connect(self._on_study_changed)
        self._training_tab.training_finished.connect(self._on_training_finished)
        self._state.study_changed.connect(self._on_study_changed_status)

    def _on_study_changed(self, study_data: dict):
        """Handle study selection from data tab."""
        self._state.set_current_study(study_data)

    def _on_training_finished(self, result):
        decoder = result.decoder
        model_type = getattr(decoder, "model_type", "model")
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        name = f"{model_type}_{timestamp}"
        self._state.add_trained_model(name, decoder)
        self._status_bar.showMessage(f"Model '{name}' trained and ready for evaluation")

    def _on_study_changed_status(self, data):
        """Update status bar when study changes."""
        if data is None:
            self._status_bar.showMessage("No study selected")
            return
        if isinstance(data, dict) and "config" in data:
            config = data["config"]
            self._status_bar.showMessage(
                f"Study: {config.name} ({len(config.subjects)} subjects)"
            )

    @property
    def state(self) -> AppState:
        return self._state


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_app_styles(app)

    window = TrainerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
