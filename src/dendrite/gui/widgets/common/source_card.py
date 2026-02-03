"""
Source Card Widget

Clickable option cards for source selection (e.g., decoder source: File/Database/Sync).
Provides modern card-based UI as alternative to radio buttons.
"""

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles


class SelectableCard(QtWidgets.QFrame):
    """Base class for clickable, selectable card widgets."""

    clicked = QtCore.pyqtSignal()

    def __init__(self, parent=None, radius: str = "radius"):
        super().__init__(parent)
        self.is_selected = False
        self._radius_key = radius
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def _update_style(self):
        """Update card style based on selection state."""
        radius = LAYOUT[self._radius_key]
        if self.is_selected:
            self.setStyleSheet(
                WidgetStyles.frame(
                    bg=WidgetStyles.colors["background_input"],
                    border=True,
                    border_color=WidgetStyles.colors["accent_primary"],
                    radius=radius,
                    padding=0,
                )
            )
        else:
            self.setStyleSheet(
                WidgetStyles.frame(
                    bg=WidgetStyles.colors["background_secondary"],
                    border=True,
                    border_color=WidgetStyles.colors["border"],
                    radius=radius,
                    padding=0,
                    hover_bg=WidgetStyles.colors["background_input"],
                    hover_border=WidgetStyles.colors["accent_primary"],
                )
            )

    def set_selected(self, selected: bool):
        """Set selection state."""
        self.is_selected = selected
        self._update_style()

    def mousePressEvent(self, event):
        """Handle mouse press."""
        self.clicked.emit()
        super().mousePressEvent(event)


class SourceCard(SelectableCard):
    """Compact text-only card for source selection."""

    def __init__(self, card_id: str, title: str, parent=None):
        super().__init__(parent, radius="radius")
        self.card_id = card_id
        self.title = title
        self._setup_ui()

    def _setup_ui(self):
        """Set up the card UI."""
        self.setFixedWidth(80)
        self.setFixedHeight(40)
        self._update_style()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(0)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        title_label = QtWidgets.QLabel(self.title)
        title_label.setStyleSheet(WidgetStyles.label("small"))
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        self.title_label = title_label


class SourceCardGroup(QtWidgets.QWidget):
    """Horizontal row of source cards with exclusive selection."""

    selection_changed = QtCore.pyqtSignal(str)  # emits card_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards: list[SourceCard] = []
        self._selected_id: str | None = None
        self._setup_ui()

    def _setup_ui(self):
        """Set up the group layout."""
        self._layout = QtWidgets.QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(LAYOUT["spacing"])
        self._layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

    def add_card(self, card_id: str, title: str) -> SourceCard:
        """Add a card to the group."""
        card = SourceCard(card_id, title, self)
        card.clicked.connect(lambda: self._on_card_clicked(card_id))
        self._cards.append(card)
        self._layout.addWidget(card)

        # Select first card by default
        if len(self._cards) == 1:
            self.set_selected(card_id)

        return card

    def _on_card_clicked(self, card_id: str):
        """Handle card click."""
        if card_id != self._selected_id:
            self.set_selected(card_id)
            self.selection_changed.emit(card_id)

    def get_selected(self) -> str | None:
        """Get the currently selected card ID."""
        return self._selected_id

    def set_selected(self, card_id: str, emit_signal: bool = False):
        """Programmatically select a card."""
        for card in self._cards:
            card.set_selected(card.card_id == card_id)

        self._selected_id = card_id

        if emit_signal:
            self.selection_changed.emit(card_id)

    def clear(self):
        """Remove all cards."""
        for card in self._cards:
            card.deleteLater()
        self._cards.clear()
        self._selected_id = None


class DescriptiveSourceCard(SelectableCard):
    """Card with title and description for clearer source selection."""

    def __init__(self, card_id: str, title: str, description: str, parent=None):
        super().__init__(parent, radius="radius")
        self.card_id = card_id
        self.title = title
        self.description = description
        self._setup_ui()

    def _setup_ui(self):
        """Set up the card UI with title and description."""
        self.setFixedWidth(140)
        self.setMinimumHeight(50)
        self._update_style()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(2)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        title_label = QtWidgets.QLabel(self.title)
        title_label.setStyleSheet(WidgetStyles.label("small", weight="bold"))
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        desc_label = QtWidgets.QLabel(self.description)
        desc_label.setStyleSheet(
            WidgetStyles.label("tiny", color=WidgetStyles.colors["text_muted"])
        )
        desc_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        self.title_label = title_label
        self.desc_label = desc_label


class DescriptiveSourceCardGroup(QtWidgets.QWidget):
    """Horizontal row of descriptive source cards with exclusive selection."""

    selection_changed = QtCore.pyqtSignal(str)  # emits card_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards: list[DescriptiveSourceCard] = []
        self._selected_id: str | None = None
        self._setup_ui()

    def _setup_ui(self):
        """Set up the group layout."""
        self._layout = QtWidgets.QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(LAYOUT["spacing"])
        self._layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

    def add_card(self, card_id: str, title: str, description: str) -> DescriptiveSourceCard:
        """Add a descriptive card to the group."""
        card = DescriptiveSourceCard(card_id, title, description, self)
        card.clicked.connect(lambda: self._on_card_clicked(card_id))
        self._cards.append(card)
        self._layout.addWidget(card)

        # Select first card by default
        if len(self._cards) == 1:
            self.set_selected(card_id)

        return card

    def _on_card_clicked(self, card_id: str):
        """Handle card click."""
        if card_id != self._selected_id:
            self.set_selected(card_id)
            self.selection_changed.emit(card_id)

    def get_selected(self) -> str | None:
        """Get the currently selected card ID."""
        return self._selected_id

    def set_selected(self, card_id: str, emit_signal: bool = False):
        """Programmatically select a card."""
        for card in self._cards:
            card.set_selected(card.card_id == card_id)

        self._selected_id = card_id

        if emit_signal:
            self.selection_changed.emit(card_id)

    def clear(self):
        """Remove all cards."""
        for card in self._cards:
            card.deleteLater()
        self._cards.clear()
        self._selected_id = None
