#!/usr/bin/env python
"""
Mode Manager

Manages mode registration, routing, and UI initialization for different BMI mode types.
"""

import logging
from collections.abc import Callable
from typing import Any

import pyqtgraph as pg
from PyQt6 import QtWidgets

from dendrite.auxiliary.dashboard.widgets.components import SECTION_TITLE_STYLE
from dendrite.constants import MODE_ASYNCHRONOUS, MODE_NEUROFEEDBACK, MODE_SYNCHRONOUS
from dendrite.gui.widgets.common.pill_navigation import PillNavigation


class ModeManager:
    """Manages BMI modes and routes data to appropriate handlers"""

    def __init__(self, clf_stack: QtWidgets.QStackedWidget, clf_nav_container: QtWidgets.QLayout):
        self.clf_stack = clf_stack
        self.clf_nav_container = clf_nav_container
        self.clf_nav: PillNavigation | None = None

        # Track registered modes
        self.registered_modes: set[str] = set()
        self.mode_types: dict[str, str] = {}
        self.mode_indices: dict[str, int] = {}  # mode_name -> stack index

        # Mode-specific UI components
        self.mode_widgets: dict[str, dict[str, Any]] = {}
        self._tab_widgets: dict[str, QtWidgets.QWidget] = {}  # mode_name -> tab widget

        # Handlers for different data types
        self.performance_handler: Callable = None
        self.erp_handler: Callable = None
        self.async_handler: Callable = None  # Unified async mode handler
        self.neurofeedback_handler: Callable = None

    def register_handlers(
        self,
        performance_handler: Callable | None = None,
        erp_handler: Callable | None = None,
        async_handler: Callable | None = None,
        neurofeedback_handler: Callable | None = None,
    ):
        """Register data handlers for different mode types."""
        self.performance_handler = performance_handler
        self.erp_handler = erp_handler
        self.async_handler = async_handler
        self.neurofeedback_handler = neurofeedback_handler

    def register_mode(self, mode_name: str, mode_type: str) -> bool:
        """Register a new mode and return True if it's a new mode."""
        is_new = mode_name not in self.registered_modes
        if is_new:
            self.registered_modes.add(mode_name)
            self.mode_types[mode_name] = mode_type
            logging.info(f"Registered new mode '{mode_name}' with type '{mode_type}'")
        return is_new

    def _update_navigation(self):
        """Rebuild navigation when modes change."""
        if not self.registered_modes:
            if self.clf_nav:
                self.clf_nav.hide()
            return

        tabs = [(name, name) for name in sorted(self.registered_modes)]

        # Remove old navigation if it exists
        if self.clf_nav:
            self.clf_nav.deleteLater()
            self.clf_nav = None

        # Create new navigation
        self.clf_nav = PillNavigation(tabs=tabs, size="normal")
        self.clf_nav.section_changed.connect(self._on_tab_changed)
        self.clf_nav_container.insertWidget(0, self.clf_nav)

        # Select first mode's tab (PillNavigation checks button but doesn't emit signal)
        first_mode = sorted(self.registered_modes)[0]
        if first_mode in self.mode_indices:
            self.clf_stack.setCurrentIndex(self.mode_indices[first_mode])

    def _on_tab_changed(self, mode_name: str):
        """Handle navigation tab selection."""
        if mode_name in self.mode_indices:
            self.clf_stack.setCurrentIndex(self.mode_indices[mode_name])

    def get_mode_type(self, mode_name: str) -> str:
        """Get the type of a registered mode."""
        return self.mode_types.get(mode_name, "unknown")

    def route_data(self, item: dict[str, Any]):
        """Route data to appropriate handler based on mode type."""
        payload_type = item.get("output_type") or item.get("type")
        mode_name = item.get("mode_name")
        mode_type = item.get("mode_type")

        if not (payload_type and mode_name and mode_type):
            logging.warning("Invalid packet: missing required fields")
            return

        if mode_type == MODE_SYNCHRONOUS:
            if payload_type == "performance" and self.performance_handler:
                self.performance_handler(mode_name, item)
            elif (
                payload_type.endswith("_erp") or "eeg_data" in item.get("data", {})
            ) and self.erp_handler:
                self.erp_handler(mode_name, item)
        elif mode_type == MODE_ASYNCHRONOUS:
            if payload_type in ["performance", "classification"] and self.async_handler:
                self.async_handler(mode_name, item)
        elif mode_type == MODE_NEUROFEEDBACK:
            if payload_type == "neurofeedback_features" and self.neurofeedback_handler:
                self.neurofeedback_handler(mode_name, item.get("data", {}))
        else:
            logging.warning(f"Unknown mode type '{mode_type}' for mode '{mode_name}'")

    def _create_plot_section(
        self,
        title: str,
        tab_layout: QtWidgets.QVBoxLayout,
        stretch: int,
        min_height: int = 150,
        create_controls: Callable | None = None,
        mode_name: str | None = None,
    ) -> tuple[pg.GraphicsLayoutWidget, QtWidgets.QVBoxLayout]:
        """Create a titled plot section with optional controls."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QtWidgets.QLabel(title)
        label.setStyleSheet(SECTION_TITLE_STYLE)
        layout.addWidget(label)
        if create_controls and mode_name:
            layout.addWidget(create_controls(mode_name))
        plot_widget = pg.GraphicsLayoutWidget()
        plot_widget.setBackground(None)
        plot_widget.setMinimumHeight(min_height)
        plot_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(plot_widget, stretch=1)
        tab_layout.addWidget(container, stretch=stretch)
        return plot_widget, layout

    def create_mode_tab(
        self,
        mode_name: str,
        mode_type: str,
        create_erp_controls: Callable | None = None,
        create_perf_controls: Callable | None = None,
    ) -> dict[str, Any]:
        """Create UI tab for a mode."""
        if mode_name in self.mode_widgets:
            return self.mode_widgets[mode_name]

        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(4, 2, 4, 4)

        mode_ui = {}

        if mode_type == MODE_ASYNCHRONOUS:
            pw, layout = self._create_plot_section("Predictions", tab_layout, stretch=1)
            mode_ui["PredictionTrace"] = {"widget": pw, "layout": layout}

        elif mode_type == MODE_SYNCHRONOUS:
            pw, layout = self._create_plot_section(
                "Performance",
                tab_layout,
                stretch=2,
                create_controls=create_perf_controls,
                mode_name=mode_name,
            )
            mode_ui["Performance"] = {"widget": pw, "layout": layout}

            pw, layout = self._create_plot_section(
                "ERP",
                tab_layout,
                stretch=3,
                create_controls=create_erp_controls,
                mode_name=mode_name,
            )
            mode_ui["ERP"] = {"widget": pw, "layout": layout}

        index = self.clf_stack.addWidget(tab)
        self.mode_indices[mode_name] = index
        self.mode_widgets[mode_name] = mode_ui
        self._tab_widgets[mode_name] = tab

        self._update_navigation()

        logging.info(f"Created UI tab for mode: {mode_name} (type: {mode_type})")
        return mode_ui

    def add_external_tab(self, mode_name: str, widget: QtWidgets.QWidget) -> int:
        """Add an externally-created tab widget to the stack.

        Args:
            mode_name: The name of the mode this tab belongs to
            widget: The widget to add to the stack

        Returns:
            The index of the widget in the stack
        """
        index = self.clf_stack.addWidget(widget)
        self.mode_indices[mode_name] = index
        self._tab_widgets[mode_name] = widget
        self._update_navigation()
        return index

    def clear_all(self):
        """Clear all modes."""
        self.registered_modes.clear()
        self.mode_types.clear()
        self.mode_widgets.clear()
        self.mode_indices.clear()
        self._tab_widgets.clear()

        while self.clf_stack.count() > 0:
            widget = self.clf_stack.widget(0)
            self.clf_stack.removeWidget(widget)
            widget.deleteLater()

        if self.clf_nav:
            self.clf_nav.deleteLater()
            self.clf_nav = None
