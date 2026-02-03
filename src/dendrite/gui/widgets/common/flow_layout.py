"""
Flow Layout

A layout that arranges widgets horizontally and wraps to new rows as needed.
Based on Qt's FlowLayout example.
"""

from PyQt6 import QtCore, QtWidgets


class FlowLayout(QtWidgets.QLayout):
    """Layout that flows widgets horizontally, wrapping to new rows."""

    def __init__(self, parent=None, spacing: int = 6):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def spacing(self):
        return self._spacing

    def setSpacing(self, spacing):
        self._spacing = spacing

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QtCore.QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        row_height = 0

        for item in self._items:
            widget = item.widget()
            if widget is None:
                continue

            space_x = self._spacing
            space_y = self._spacing

            item_width = item.sizeHint().width()
            item_height = item.sizeHint().height()

            # Wrap to next row if needed
            if x + item_width > rect.right() and row_height > 0:
                x = rect.x()
                y = y + row_height + space_y
                row_height = 0

            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))

            x = x + item_width + space_x
            row_height = max(row_height, item_height)

        return y + row_height - rect.y()
