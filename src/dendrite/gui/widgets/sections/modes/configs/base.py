"""
Base classes and infrastructure for mode configuration widgets.

Contains the registry pattern, abstract base class, and factory for common widgets.
"""

import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

from PyQt6 import QtWidgets

logger = logging.getLogger(__name__)

from dendrite.data.storage.database import Database, DecoderRepository
from dendrite.gui.config.mode_config_manager import get_mode_config_manager
from dendrite.gui.config.stream_config_manager import get_stream_config_manager
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.widgets.common.pill_navigation import VerticalPillNavigation
from dendrite.gui.widgets.sections.modes.decoder_status import DecoderStatusWidget

from .factory import ModeWidgetFactory


class DecoderSource:
    """Constants for decoder source selection."""

    PRETRAINED = "pretrained"
    DATABASE = "database"
    SYNC_MODE = "sync_mode"
    NEW = "new"
    RAW = "raw"
    NONE = "none"


class BaseModeConfig(ABC):
    """Abstract base class for mode configurations"""

    def __init__(self, available_decoders: list[str], instance_name: str | None = None):
        self.available_decoders = available_decoders
        self.widgets: dict[str, QtWidgets.QWidget] = {}
        self.instance_name = instance_name
        self.mode_config_manager = get_mode_config_manager()

        # Decoder state
        self._selected_database_decoder: dict[str, Any] | None = None
        self._decoder_locked = False

        # Connect to manager signals for reactive updates
        if self.instance_name:
            self.mode_config_manager.instance_updated.connect(self._on_instance_updated)

    def _on_instance_updated(self, updated_name: str, _config: dict[str, Any]):
        """
        Handle manager updates by refreshing displays.

        Read-only handler - does not modify manager state.
        Only updates display widgets to reflect new manager state.
        """
        if updated_name == self.instance_name:
            self.refresh_status_displays()

    @contextmanager
    def _block_signals(self, widget_keys: list[str]):
        """
        Context manager to temporarily block widget signals.

        Usage:
            with self._block_signals(['widget1', 'widget2']):
                # Update widgets without triggering signals
                self.widgets['widget1'].setText("new value")
        """
        widgets = [self.widgets[k] for k in widget_keys if k in self.widgets]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            yield
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def get_config_from_manager(self) -> dict[str, Any]:
        """Get current config from manager"""
        if self.instance_name:
            return self.mode_config_manager.get_instance(self.instance_name) or {}
        return {}

    def _create_standard_tab_widget_container(
        self, parent: QtWidgets.QWidget, tabs: list[tuple[str, str]]
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QStackedWidget, "VerticalPillNavigation"]:
        """
        Create sidebar navigation container with stacked widget.

        Helper method to eliminate duplicate navigation creation code.
        Modes can call this and then add their specific tab widgets.

        Args:
            parent: Parent widget
            tabs: List of (tab_id, tab_label) tuples

        Returns:
            Tuple of (page_widget, stacked_widget, pill_nav) ready for adding tab widgets
        """
        page = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(LAYOUT["spacing"])

        # Left: vertical navigation sidebar
        pill_nav = VerticalPillNavigation(tabs=tabs)
        main_layout.addWidget(pill_nav)

        # Right: content stack (expanding)
        stacked_widget = QtWidgets.QStackedWidget()
        stacked_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )
        main_layout.addWidget(stacked_widget, stretch=1)

        # Connect navigation to stacked widget - map tab_id to index
        tab_id_to_index = {tab_id: idx for idx, (tab_id, _) in enumerate(tabs)}
        pill_nav.section_changed.connect(
            lambda tab_id: stacked_widget.setCurrentIndex(tab_id_to_index[tab_id])
        )

        self.widgets["stacked_widget"] = stacked_widget
        self.widgets["pill_nav"] = pill_nav

        return page, stacked_widget, pill_nav

    def _create_general_tab_with_channels(
        self, config: dict[str, Any], right_panel_builder=None
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
        """
        Create general tab with vertical layout.

        Channel selection at top (full width), additional widgets below.

        Args:
            config: Mode configuration dictionary
            right_panel_builder: Ignored (kept for API compatibility)

        Returns:
            Tuple of (tab_widget, layout) for adding more widgets
        """
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
        )
        layout.setSpacing(LAYOUT["spacing_xl"])

        # Channel selection with header
        channel_group = QtWidgets.QGroupBox("Channel Selection")
        channel_group.setStyleSheet(WidgetStyles.groupbox())
        channel_layout = QtWidgets.QVBoxLayout(channel_group)
        channel_layout.setContentsMargins(
            LAYOUT["spacing"], LAYOUT["spacing"],
            LAYOUT["spacing"], LAYOUT["spacing"]
        )
        channel_layout.setSpacing(LAYOUT["spacing"])

        channel_widget = ModeWidgetFactory.create_channel_selection_widget(
            config.get("channel_selection", {}), channel_group
        )
        self.widgets["channel_widget"] = channel_widget
        channel_layout.addWidget(channel_widget)

        layout.addWidget(channel_group)

        return tab, layout

    @abstractmethod
    def create_config_widget(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Create the configuration widget for this mode"""
        pass

    @abstractmethod
    def get_config_data(self) -> dict[str, Any]:
        """Extract configuration data from widgets"""
        pass

    def set_config_data(self, data: dict[str, Any]):
        """
        Set configuration data to widgets.

        Does NOT update the manager - caller is responsible for that.
        This prevents bidirectional sync issues.
        """
        # Update widgets from config data
        self._update_widgets_from_config(data)

    @abstractmethod
    def _update_widgets_from_config(self, data: dict[str, Any]):
        """Update widgets from configuration data - subclasses must implement"""
        pass

    def refresh_status_displays(self) -> None:
        """Refresh decoder status when modality config changes."""
        self._update_decoder_status_display()

    def _update_decoder_status_display(self) -> None:
        """Update decoder status display based on selected source."""
        status_widget = self.widgets.get("decoder_status_label")
        if not status_widget:
            return

        source = self.widgets.get("source_card_group")
        source_id = source.get_selected() if source else self._get_default_source()

        if source_id == DecoderSource.PRETRAINED:
            self._update_pretrained_status(status_widget)
        elif source_id == DecoderSource.DATABASE:
            self._update_database_status(status_widget)
        else:
            self._update_new_model_status(status_widget)

    @abstractmethod
    def _get_default_source(self) -> str:
        """Return default source type for this mode ('pretrained', 'raw', etc.)."""
        pass

    def _update_new_model_status(self, status_widget: DecoderStatusWidget) -> None:
        """Status for new/raw model. Subclasses can override for custom messages."""
        status_widget.set_info("New Model", "Configure events and start session to train")

    def get_default_model(self) -> str:
        """Get default model for this mode."""
        return self.available_decoders[0] if self.available_decoders else ""

    def get_current_model_name(self) -> str:
        """Get current model name from widget state."""
        from dendrite.ml.decoders import get_decoder_metadata

        # Check if using pretrained or database decoder via card group
        source_id = None
        if "source_card_group" in self.widgets:
            source_id = self.widgets["source_card_group"].get_selected()

        if source_id in (DecoderSource.PRETRAINED, DecoderSource.DATABASE):
            pretrained_path = self.widgets.get("pretrained_path_edit")
            if pretrained_path and pretrained_path.text().strip():
                try:
                    decoder_info = get_decoder_metadata(pretrained_path.text().strip())
                    return decoder_info.get("model_type", "Unknown Model")
                except (FileNotFoundError, ValueError):
                    pass  # Fall through to default

        # Get model from userData (stored when combo was populated)
        if "model_combo" in self.widgets:
            return self.widgets["model_combo"].currentData() or "EEGNet"

        config = self.get_config_from_manager()
        return config.get("model_type", "EEGNet")

    def _get_current_decoder_path(self) -> str:
        """Get the current decoder path from pretrained/database decoder selection."""
        source_id = None
        if "source_card_group" in self.widgets:
            source_id = self.widgets["source_card_group"].get_selected()

        if source_id in (DecoderSource.PRETRAINED, DecoderSource.DATABASE):
            path_edit = self.widgets.get("pretrained_path_edit")
            if path_edit:
                return path_edit.text().strip()
        return ""

    # Decoder helper methods
    @abstractmethod
    def _get_time_window_duration(self) -> float:
        """
        Get time window duration in seconds.

        Subclasses must implement this to specify how their time window is calculated.
        - Synchronous mode: end_offset - start_offset
        - Asynchronous mode: window_duration
        - Neurofeedback mode: window_length_sec
        """
        pass

    def _get_expected_shapes(self) -> dict[str, tuple[int, int]]:
        """Get expected input shapes from current configuration."""
        config = self.get_config_from_manager()
        channel_selection = config.get("channel_selection") or {}
        sample_rate = self._get_sample_rate()

        if sample_rate is None:
            return {}

        time_samples = int(self._get_time_window_duration() * sample_rate)

        return {
            modality: (len(channels), time_samples)
            for modality, channels in channel_selection.items()
            if isinstance(channels, list)
        }

    def _get_sample_rate(self) -> float:
        """Get current system sample rate."""
        stream_manager = get_stream_config_manager()
        return stream_manager.get_system_sample_rate()

    @staticmethod
    def _format_decoder_info(metadata: dict[str, Any]) -> str:
        """Format decoder metadata showing channels, duration, and rate."""
        rate = metadata.get("sample_rate") or 500.0

        if "input_shapes" in metadata:
            parts = []
            for mod, shape in metadata["input_shapes"].items():
                if len(shape) >= 2:
                    duration = shape[1] / rate
                    parts.append(f"{mod.upper()} {shape[0]}ch × {duration:.2f}s")
            if parts:
                return f"{', '.join(parts)} @ {rate:.0f}Hz"

        model_type = metadata.get("model_type") or metadata.get("classifier_type", "Unknown")
        return f"{model_type} (no shape info)"

    def _get_expected_labels(self) -> dict[str, list[str]]:
        """Get expected channel labels from current configuration."""
        config = self.get_config_from_manager()
        return config.get("modality_labels") or {}

    def apply_decoder_config_to_gui(self, metadata: dict[str, Any]) -> None:
        """
        Auto-fill and lock GUI fields from pretrained decoder metadata.

        Extracts time window and channel configuration from decoder and applies
        to GUI widgets, disabling them to prevent incompatible changes.
        """
        # Auto-set time window from decoder's expected input shape
        if metadata.get("input_shapes"):
            # Get time samples from first modality
            first_shape = list(metadata["input_shapes"].values())[0]
            if len(first_shape) >= 2:
                time_samples = first_shape[1]
                sample_rate = metadata.get("sample_rate") or self._get_sample_rate()
                duration = time_samples / sample_rate

                # Update and lock time window widgets (synchronous mode)
                if "start_edit" in self.widgets:
                    self.widgets["start_edit"].setText("0.0")
                    self.widgets["start_edit"].setEnabled(False)
                if "end_edit" in self.widgets:
                    self.widgets["end_edit"].setText(f"{duration:.3f}")
                    self.widgets["end_edit"].setEnabled(False)

                # Update window length display (asynchronous mode)
                if "window_length_display" in self.widgets:
                    self.widgets["window_length_display"].setText(f"{duration:.3f}s")

                # Store decoder window duration for config output
                self._decoder_window_sec = duration

        # Store locked state for later unlock
        self._decoder_locked = True

    def unlock_decoder_config(self) -> None:
        """Re-enable GUI fields when switching away from pretrained decoder."""
        # Re-enable time window editing
        if "start_edit" in self.widgets:
            self.widgets["start_edit"].setEnabled(True)
        if "end_edit" in self.widgets:
            self.widgets["end_edit"].setEnabled(True)

        self._decoder_locked = False

    # Database decoder methods (shared across modes)

    def _get_current_study_name(self) -> str | None:
        """Get current study from main window's general params."""
        widget = self.widgets.get("stacked_widget")
        while widget:
            if hasattr(widget, "general_params"):
                return widget.general_params.study_name_combo.currentText()
            widget = widget.parent()
        return None

    def _update_database_status(self, status_widget: DecoderStatusWidget) -> None:
        """Update status for database decoder source."""
        if not self._selected_database_decoder:
            status_widget.set_empty("No decoder selected - select from list above")
            return

        decoder = self._selected_database_decoder
        name = decoder.get("decoder_name", "Unknown")
        decoder_path = decoder.get("decoder_path", "")

        if not decoder_path:
            status_widget.set_error(
                name, "No decoder file path in database record. Entry may be corrupted."
            )
            return

        self._update_decoder_status(status_widget, decoder_path, name)

    def _restore_database_decoder(self, decoder_id: int, decoder_path: str) -> None:
        """Restore database decoder selection from saved config."""
        try:
            db = Database()
            db.init_db()
            repo = DecoderRepository(db)

            decoder = repo.get_decoder_by_id(decoder_id)
            if decoder:
                self._selected_database_decoder = decoder
            else:
                # Decoder not found in database - use path as fallback
                logger.warning(
                    f"Decoder ID {decoder_id} not found in database, using path fallback"
                )
                self._selected_database_decoder = {
                    "decoder_id": decoder_id,
                    "decoder_path": decoder_path,
                    "decoder_name": os.path.basename(decoder_path) if decoder_path else "Unknown",
                }

        except Exception as e:
            # Fallback if database access fails
            logger.warning(f"Failed to restore decoder {decoder_id} from database: {e}")
            self._selected_database_decoder = {
                "decoder_id": decoder_id,
                "decoder_path": decoder_path,
            }

    def _update_pretrained_status(self, status_widget: DecoderStatusWidget) -> None:
        """Update status for pretrained decoder source."""
        path = self.widgets.get("pretrained_path_edit", QtWidgets.QLineEdit()).text().strip()

        if not path:
            status_widget.set_empty("No decoder selected - click Browse File...")
            return

        self._update_decoder_status(status_widget, path, os.path.basename(path))

    def _update_decoder_status(
        self, status_widget: DecoderStatusWidget, path: str, display_name: str
    ) -> None:
        """Core validation and display logic for decoder files."""
        # Local import to avoid circular dependency
        from dendrite.ml.decoders import validate_decoder_file

        if not os.path.exists(path):
            status_widget.set_error(display_name, f"File not found: {path}")
            return

        expected_shapes = self._get_expected_shapes()
        sample_rate = self._get_sample_rate()
        expected_labels = self._get_expected_labels()

        metadata, issues = validate_decoder_file(
            path, expected_shapes, sample_rate, expected_labels
        )

        if metadata is None:
            error = issues[0] if issues else "Failed to load decoder"
            status_widget.set_error(display_name, error)
            return

        info = self._format_decoder_info(metadata)
        needs_resampling = metadata.get("_needs_resampling", False)
        resample_note = " (will resample)" if needs_resampling else ""
        system_channels = self._format_system_channels(expected_shapes, sample_rate)

        if issues:
            status_widget.set_warning(
                display_name,
                requires=info,
                system=f"{system_channels}{resample_note}",
                issues=issues,
            )
        else:
            note = "Will resample to match decoder" if needs_resampling else ""
            status_widget.set_valid(
                display_name, requires=info, system=f"{system_channels}{resample_note}", note=note
            )
            self.apply_decoder_config_to_gui(metadata)

    def _format_system_channels(
        self, expected_shapes: dict[str, tuple[int, int]], sample_rate: float
    ) -> str:
        """Format system channels (only what matters for validation)."""
        if not expected_shapes:
            return "Not configured"

        parts = []
        for mod, shape in expected_shapes.items():
            if len(shape) >= 2:
                parts.append(f"{mod.upper()} {shape[0]}ch")

        return f"{', '.join(parts)} @ {sample_rate:.0f}Hz"
