"""Training results visualization panel.

Self-contained widget that owns all results visualization: training curves,
confusion matrix, per-class metrics, Optuna search progress, and metric cards.
"""

from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from sklearn.metrics import cohen_kappa_score, f1_score

from dendrite.auxiliary.ml_workbench import TrainResult
from dendrite.auxiliary.ml_workbench.utils import create_styled_plot
from dendrite.gui.styles.design_tokens import (
    ACCENT,
    BG_PANEL,
    FONT_SIZE,
    STATUS_SUCCESS,
    STATUS_WARNING_ALT,
    TEXT_MAIN,
    TEXT_MUTED,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles


class TrainingResultsPanel(QtWidgets.QWidget):
    """Panel displaying all training result visualizations.

    Includes loss/accuracy training curves, Optuna search progress scatter plot,
    confusion matrix heatmap, per-class precision/recall/F1 bars, metric summary
    cards, and Optuna search summary.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cm_labels: list[pg.TextItem] = []
        self._cv_items: list = []
        self._optuna_trial_data: list[tuple[int, float, bool]] = []
        self._class_names: dict[int, str] | None = None
        self._live_train_losses: list[float] = []
        self._live_train_accs: list[float] = []
        self._live_val_losses: list[float] = []
        self._live_val_accs: list[float] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LAYOUT["spacing_md"])

        plots_container = QtWidgets.QFrame()
        plots_container.setObjectName("plotsContainer")
        plots_container.setStyleSheet(WidgetStyles.container(bg="main", radius=0))
        plots_layout = QtWidgets.QVBoxLayout(plots_container)
        plots_layout.setContentsMargins(
            LAYOUT["spacing_xl"], LAYOUT["spacing_xl"],
            LAYOUT["spacing_xl"], LAYOUT["spacing_xl"],
        )
        plots_layout.setSpacing(LAYOUT["spacing_md"])

        # Row 1: Training curves (Loss + Accuracy)
        plots_layout.addLayout(self._create_training_curves())

        # Row 2: Optuna search progress (hidden by default)
        self._create_optuna_plot()
        self._optuna_plot.hide()
        plots_layout.addWidget(self._optuna_plot)

        # Row 3: Analysis plots (Confusion matrix + Per-class metrics)
        plots_layout.addLayout(self._create_analysis_plots())

        # Row 4: Metrics summary cards
        plots_layout.addLayout(self._create_metric_cards())

        # Optuna summary (shown after Optuna search)
        self._create_optuna_summary()
        self._optuna_summary_frame.hide()
        plots_layout.addWidget(self._optuna_summary_frame)

        layout.addWidget(plots_container, stretch=1)

    def _create_training_curves(self) -> QtWidgets.QHBoxLayout:
        curves_layout = QtWidgets.QHBoxLayout()
        curves_layout.setSpacing(LAYOUT["spacing_lg"])

        # Loss plot
        self._loss_plot = create_styled_plot("Loss")
        self._loss_plot.addLegend(offset=(-10, 10))
        self._train_loss_curve = self._loss_plot.plot(
            pen=pg.mkPen(ACCENT, width=2.5), name="Train", antialias=True
        )
        self._val_loss_curve = self._loss_plot.plot(
            pen=pg.mkPen(STATUS_WARNING_ALT, width=2.5, style=QtCore.Qt.PenStyle.DashLine),
            name="Val",
            antialias=True,
        )
        curves_layout.addWidget(self._loss_plot)

        # Accuracy plot
        self._acc_plot = create_styled_plot("Accuracy", y_range=(0, 1))
        self._acc_plot.addLegend(offset=(-10, 10))
        self._train_acc_curve = self._acc_plot.plot(
            pen=pg.mkPen(ACCENT, width=2.5), name="Train", antialias=True
        )
        self._val_acc_curve = self._acc_plot.plot(
            pen=pg.mkPen(STATUS_WARNING_ALT, width=2.5, style=QtCore.Qt.PenStyle.DashLine),
            name="Val",
            antialias=True,
        )
        # Chance level line (updated when training completes)
        self._chance_line = pg.InfiniteLine(
            pos=0.5,
            angle=0,
            pen=pg.mkPen(color=TEXT_MUTED, width=1.5, style=QtCore.Qt.PenStyle.DotLine),
            label="chance",
            labelOpts={"position": 0.95, "color": TEXT_MUTED},
        )
        self._acc_plot.addItem(self._chance_line)
        curves_layout.addWidget(self._acc_plot)

        return curves_layout

    def _create_optuna_plot(self):
        self._optuna_plot = create_styled_plot("Search Progress", height=160, y_range=(0, 1))
        self._optuna_plot.setLabel("bottom", "Trial")
        self._optuna_plot.setLabel("left", "Validation Accuracy")

        # Regular trial scatter points
        self._optuna_trials_scatter = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen(ACCENT, width=1), brush=pg.mkBrush(ACCENT), symbol="o"
        )
        self._optuna_plot.addItem(self._optuna_trials_scatter)

        # Best trial scatter points (highlighted)
        self._optuna_best_scatter = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen(STATUS_SUCCESS, width=2),
            brush=pg.mkBrush(STATUS_SUCCESS),
            symbol="star",
        )
        self._optuna_plot.addItem(self._optuna_best_scatter)

        # Best accuracy line
        self._optuna_best_line = pg.InfiniteLine(
            pos=0,
            angle=0,
            pen=pg.mkPen(color=STATUS_SUCCESS, width=1.5, style=QtCore.Qt.PenStyle.DashLine),
        )
        self._optuna_plot.addItem(self._optuna_best_line)

        # Text item for best accuracy
        self._optuna_best_text = pg.TextItem("", anchor=(1, 0), color=STATUS_SUCCESS)
        self._optuna_best_text.setFont(QtGui.QFont("sans-serif", 10, QtGui.QFont.Weight.Bold))
        self._optuna_plot.addItem(self._optuna_best_text)

    def _create_analysis_plots(self) -> QtWidgets.QHBoxLayout:
        analysis_layout = QtWidgets.QHBoxLayout()
        analysis_layout.setSpacing(LAYOUT["spacing_lg"])

        # Confusion matrix heatmap
        self._cm_plot = create_styled_plot("Confusion Matrix", height=180, show_grid=False)
        self._cm_plot.setAspectLocked(True)
        self._cm_image = pg.ImageItem()
        self._cm_plot.addItem(self._cm_image)
        analysis_layout.addWidget(self._cm_plot)

        # Per-class metrics bar chart
        self._class_plot = create_styled_plot("Per-Class Metrics", height=180, y_range=(0, 1))
        self._class_plot.addLegend(offset=(-10, 10))
        analysis_layout.addWidget(self._class_plot)

        return analysis_layout

    def _create_metric_cards(self) -> QtWidgets.QHBoxLayout:
        metrics_layout = QtWidgets.QHBoxLayout()
        metrics_layout.setSpacing(LAYOUT["spacing_md"])

        acc_card, self._acc_label = self._create_metric_card("Accuracy", primary=True)
        metrics_layout.addWidget(acc_card)

        val_acc_card, self._val_acc_label = self._create_metric_card("Val Acc")
        metrics_layout.addWidget(val_acc_card)

        f1_card, self._f1_label = self._create_metric_card("F1 Score")
        metrics_layout.addWidget(f1_card)

        kappa_card, self._kappa_label = self._create_metric_card("Kappa")
        metrics_layout.addWidget(kappa_card)

        metrics_layout.addStretch()
        return metrics_layout

    def _create_metric_card(
        self, title: str, primary: bool = False
    ) -> tuple[QtWidgets.QFrame, QtWidgets.QLabel]:
        """Create a styled metric card widget and return (card, value_label)."""
        card = QtWidgets.QFrame()
        card.setStyleSheet(WidgetStyles.metric_card())

        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(
            LAYOUT["spacing_lg"], LAYOUT["spacing_md"],
            LAYOUT["spacing_lg"], LAYOUT["spacing_md"],
        )
        layout.setSpacing(LAYOUT["spacing_xs"])

        title_lbl = QtWidgets.QLabel(title.upper())
        title_lbl.setStyleSheet(WidgetStyles.metric_card_title())
        layout.addWidget(title_lbl)

        value_lbl = QtWidgets.QLabel("--")
        size = FONT_SIZE["xl"] if primary else FONT_SIZE["lg"]
        weight = "bold" if primary else "600"
        value_lbl.setStyleSheet(WidgetStyles.metric_card_value(size=size, weight=weight))
        layout.addWidget(value_lbl)

        return card, value_lbl

    def _create_optuna_summary(self):
        self._optuna_summary_frame = QtWidgets.QFrame()
        self._optuna_summary_frame.setStyleSheet(WidgetStyles.metric_card())
        summary_layout = QtWidgets.QVBoxLayout(self._optuna_summary_frame)
        summary_layout.setContentsMargins(
            LAYOUT["spacing_lg"], LAYOUT["spacing_md"],
            LAYOUT["spacing_lg"], LAYOUT["spacing_md"],
        )
        summary_layout.setSpacing(LAYOUT["spacing_xs"])

        optuna_title = QtWidgets.QLabel("SEARCH RESULTS")
        optuna_title.setStyleSheet(WidgetStyles.metric_card_title())
        summary_layout.addWidget(optuna_title)

        self._optuna_summary_label = QtWidgets.QLabel("")
        self._optuna_summary_label.setWordWrap(True)
        self._optuna_summary_label.setStyleSheet(WidgetStyles.inline_label(color=TEXT_MAIN))
        summary_layout.addWidget(self._optuna_summary_label)

    # --- Public API ---

    def clear(self):
        """Reset all visualizations to initial state."""
        self._acc_label.setText("--")
        self._val_acc_label.setText("--")
        self._f1_label.setText("--")
        self._kappa_label.setText("--")
        self._train_loss_curve.setData([], [])
        self._val_loss_curve.setData([], [])
        self._train_acc_curve.setData([], [])
        self._val_acc_curve.setData([], [])
        self._update_confusion_matrix(None)
        self._class_plot.clear()
        self._update_cv_visualization(None, 0)
        self._optuna_summary_frame.hide()
        self._optuna_plot.hide()
        self._optuna_trial_data = []
        self._optuna_trials_scatter.setData([], [])
        self._optuna_best_scatter.setData([], [])
        self._optuna_best_line.setValue(0)
        self._optuna_best_text.setText("")

    def show_optuna_plot(self):
        """Show the Optuna search progress plot."""
        self._optuna_plot.show()

    def update_epoch(
        self,
        epoch: int,
        total: int,
        train_loss: float,
        train_acc: float,
        val_loss: float | None,
        val_acc: float | None,
    ):
        """Update loss/accuracy curves after each epoch (live updates)."""
        self._live_train_losses.append(train_loss)
        self._live_train_accs.append(train_acc)

        epochs = list(range(1, len(self._live_train_losses) + 1))

        self._train_loss_curve.setData(epochs, self._live_train_losses)
        self._train_acc_curve.setData(epochs, self._live_train_accs)

        if val_loss is not None:
            self._live_val_losses.append(val_loss)
            self._val_loss_curve.setData(epochs, self._live_val_losses)
        if val_acc is not None:
            self._live_val_accs.append(val_acc)
            self._val_acc_curve.setData(epochs, self._live_val_accs)

    def reset_live_tracking(self):
        """Reset live epoch tracking lists for a new training run."""
        self._live_train_losses = []
        self._live_train_accs = []
        self._live_val_losses = []
        self._live_val_accs = []

    def update_trial(self, trial_num: int, accuracy: float, is_best: bool):
        """Update Optuna progress plot with new trial result."""
        self._optuna_trial_data.append((trial_num, accuracy, is_best))

        all_trials = [(t[0], t[1]) for t in self._optuna_trial_data]
        best_trials = [(t[0], t[1]) for t in self._optuna_trial_data if t[2]]

        if all_trials:
            x_all, y_all = zip(*all_trials, strict=False)
            self._optuna_trials_scatter.setData(x_all, y_all)

        if best_trials:
            x_best, y_best = zip(*best_trials, strict=False)
            self._optuna_best_scatter.setData(x_best, y_best)

            current_best = max(y_best)
            self._optuna_best_line.setValue(current_best)
            self._optuna_best_text.setText(f"Best: {current_best * 100:.1f}%")
            self._optuna_best_text.setPos(trial_num, current_best + 0.05)

        self._optuna_plot.setXRange(0, trial_num + 1, padding=0.05)

    def update_from_result(self, result: TrainResult, class_names: dict[int, str] | None):
        """Update all visualizations from a completed training result.

        Args:
            result: Training result containing metrics, confusion matrix, history
            class_names: Mapping of class index to display name
        """
        self._class_names = class_names

        history = result.train_history
        if history:
            epochs = list(range(1, len(history.get("loss", [])) + 1))
            if epochs:
                self._train_loss_curve.setData(epochs, history.get("loss", []))
                self._val_loss_curve.setData(epochs, history.get("val_loss", []))
                self._train_acc_curve.setData(epochs, history.get("accuracy", []))
                self._val_acc_curve.setData(epochs, history.get("val_accuracy", []))

        if result.confusion_matrix is not None and result.confusion_matrix.size > 0:
            n_classes = result.confusion_matrix.shape[0]
            chance_level = 1.0 / n_classes
            self._chance_line.setValue(chance_level)

        self._update_confusion_matrix(result.confusion_matrix)
        self._update_class_metrics(result.confusion_matrix)
        self._update_cv_visualization(
            result.cv_results, len(history.get("accuracy", [])) if history else 0
        )
        self._update_metric_labels(result)

    def update_optuna_summary(self, optuna_results: dict[str, Any] | None):
        """Update optuna summary display."""
        if not optuna_results:
            self._optuna_summary_frame.hide()
            return

        best_params = optuna_results.get("best_params", {})
        best_value = optuna_results.get("best_value", 0)
        n_completed = optuna_results.get("n_completed", 0)
        n_pruned = optuna_results.get("n_pruned", 0)
        n_failed = optuna_results.get("n_failed", 0)

        param_lines = []
        for k, v in best_params.items():
            if isinstance(v, float):
                param_lines.append(f"  {k}: {v:.4g}")
            else:
                param_lines.append(f"  {k}: {v}")
        params_str = "\n".join(param_lines) if param_lines else "  (none)"

        summary = (
            f"Trials: {n_completed} completed, {n_pruned} pruned, {n_failed} failed\n"
            f"Best accuracy: {best_value * 100:.1f}%\n\n"
            f"Best Parameters:\n{params_str}"
        )

        self._optuna_summary_label.setText(summary)
        self._optuna_summary_frame.show()

    # --- Private visualization helpers ---

    def _update_metric_labels(self, result: TrainResult) -> None:
        """Update the metric summary labels with training results."""
        self._acc_label.setText(f"{result.accuracy * 100:.1f}%")
        self._val_acc_label.setText(f"{result.val_accuracy * 100:.1f}%")

        cm = result.confusion_matrix
        if cm is not None and cm.size > 0:
            y_true, y_pred = [], []
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    y_true.extend([i] * cm[i, j])
                    y_pred.extend([j] * cm[i, j])

            if y_true:
                f1 = f1_score(y_true, y_pred, average="weighted")
                kappa = cohen_kappa_score(y_true, y_pred)
                self._f1_label.setText(f"{f1:.3f}")
                self._kappa_label.setText(f"{kappa:.3f}")

    def _create_confusion_matrix_colormap(self) -> np.ndarray:
        """Create colormap lookup table for confusion matrix.

        Returns:
            256x4 RGBA lookup table array
        """
        colors = np.array(
            [
                [21, 21, 21, 255],
                [0, 80, 160, 255],
                [0, 123, 255, 255],
                [0, 180, 140, 255],
                [0, 212, 170, 255],
            ],
            dtype=np.uint8,
        )

        lut = np.zeros((256, 4), dtype=np.uint8)
        n_colors = len(colors)
        for i in range(256):
            t = i / 255.0 * (n_colors - 1)
            idx = min(int(t), n_colors - 2)
            frac = t - idx
            lut[i] = colors[idx] * (1 - frac) + colors[idx + 1] * frac
        return lut

    def _update_confusion_matrix(self, cm: np.ndarray | None) -> None:
        """Update confusion matrix heatmap visualization."""
        for label in self._cm_labels:
            self._cm_plot.removeItem(label)
        self._cm_labels.clear()

        if cm is None or cm.size == 0:
            self._cm_image.clear()
            return

        n_classes = cm.shape[0]

        cm_normalized = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_normalized = cm_normalized / row_sums

        self._cm_image.setImage(cm_normalized.T)
        self._cm_image.setLookupTable(self._create_confusion_matrix_colormap())

        self._cm_image.setRect(0, 0, n_classes, n_classes)
        self._cm_plot.setXRange(-0.5, n_classes - 0.5, padding=0.1)
        self._cm_plot.setYRange(-0.5, n_classes - 0.5, padding=0.1)

        class_names = self._class_names or {i: str(i) for i in range(n_classes)}
        ax = self._cm_plot.getAxis("bottom")
        ax.setTicks([[(i, class_names.get(i, str(i))) for i in range(n_classes)]])
        ax.setLabel("Predicted")
        ay = self._cm_plot.getAxis("left")
        ay.setTicks([[(i, class_names.get(i, str(i))) for i in range(n_classes)]])
        ay.setLabel("Actual")

        for i in range(n_classes):
            for j in range(n_classes):
                value = cm[i, j]
                normalized = cm_normalized[i, j]
                pct = normalized * 100
                text_color = TEXT_MAIN if normalized < 0.6 else BG_PANEL
                label = f"{value}\n({pct:.0f}%)"
                text = pg.TextItem(label, anchor=(0.5, 0.5), color=text_color)
                text.setFont(QtGui.QFont("sans-serif", 9, QtGui.QFont.Weight.Bold))
                text.setPos(j + 0.5, i + 0.5)
                self._cm_plot.addItem(text)
                self._cm_labels.append(text)

    def _update_class_metrics(self, cm: np.ndarray | None) -> None:
        """Update per-class precision/recall/F1 bar chart."""
        self._class_plot.clear()
        self._class_plot.addLegend(offset=(-10, 10))

        if cm is None or cm.size == 0:
            return

        n_classes = cm.shape[0]

        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1 = np.zeros(n_classes)

        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[i] = (
                2 * precision[i] * recall[i] / (precision[i] + recall[i])
                if (precision[i] + recall[i]) > 0
                else 0
            )

        bar_width = 0.25
        x = np.arange(n_classes)

        precision_bars = pg.BarGraphItem(
            x=x - bar_width,
            height=precision,
            width=bar_width,
            brush=pg.mkBrush(STATUS_SUCCESS),
            name="Precision",
        )
        recall_bars = pg.BarGraphItem(
            x=x, height=recall, width=bar_width, brush=pg.mkBrush(ACCENT), name="Recall"
        )
        f1_bars = pg.BarGraphItem(
            x=x + bar_width,
            height=f1,
            width=bar_width,
            brush=pg.mkBrush(STATUS_WARNING_ALT),
            name="F1",
        )

        self._class_plot.addItem(precision_bars)
        self._class_plot.addItem(recall_bars)
        self._class_plot.addItem(f1_bars)

        class_names = self._class_names or {i: f"Class {i}" for i in range(n_classes)}
        ax = self._class_plot.getAxis("bottom")
        ax.setTicks([[(i, class_names.get(i, f"Class {i}")) for i in range(n_classes)]])
        self._class_plot.setXRange(-0.5, n_classes - 0.5, padding=0.1)

    def _update_cv_visualization(self, cv_results: dict[str, Any] | None, n_epochs: int) -> None:
        """Update CV fold visualization on accuracy plot."""
        for item in self._cv_items:
            self._acc_plot.removeItem(item)
        self._cv_items = []

        if cv_results is None:
            return

        fold_scores = cv_results.get("fold_scores", [])
        mean_acc = cv_results.get("mean_accuracy", 0)
        std_acc = cv_results.get("std_accuracy", 0)

        if not fold_scores:
            return

        x_pos = n_epochs + 1 if n_epochs > 0 else 1
        n_folds = len(fold_scores)

        fold_scatter = pg.ScatterPlotItem(
            x=[x_pos] * n_folds,
            y=fold_scores,
            size=10,
            pen=pg.mkPen("w", width=1),
            brush=pg.mkBrush(STATUS_SUCCESS),
            symbol="o",
            name="CV Folds",
        )
        self._acc_plot.addItem(fold_scatter)
        self._cv_items.append(fold_scatter)

        mean_line = pg.InfiniteLine(
            pos=mean_acc,
            angle=0,
            pen=pg.mkPen(color=STATUS_SUCCESS, width=2, style=QtCore.Qt.PenStyle.DashLine),
            label=f"CV mean: {mean_acc * 100:.1f}%",
            labelOpts={"position": 0.05, "color": STATUS_SUCCESS},
        )
        self._acc_plot.addItem(mean_line)
        self._cv_items.append(mean_line)

        if std_acc > 0:
            x_vals = [0, n_epochs + 2] if n_epochs > 0 else [0, 2]
            upper = pg.PlotDataItem(x_vals, [mean_acc + std_acc] * 2)
            lower = pg.PlotDataItem(x_vals, [mean_acc - std_acc] * 2)
            fill = pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(80, 180, 120, 40))
            self._acc_plot.addItem(fill)
            self._cv_items.append(fill)
