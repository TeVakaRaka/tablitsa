from typing import List
from PySide2.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QCheckBox, QFrame, QScrollArea, QSizePolicy
)
from PySide2.QtCore import Qt, Signal

from ...models import Table


class TableListItem(QFrame):
    """Single item in the table list."""

    toggled = Signal(str, bool)  # table_id, enabled
    selected = Signal(str)  # table_id

    def __init__(self, table: Table, parent=None):
        super().__init__(parent)
        self.table = table
        self.is_selected = False
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameShape(QFrame.Box)
        self.setCursor(Qt.PointingHandCursor)
        self._update_style()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Checkbox
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(self.table.enabled)
        self.checkbox.stateChanged.connect(self._on_toggle)
        layout.addWidget(self.checkbox)

        # Info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        self.name_label = QLabel(self.table.name)
        self.name_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.name_label)

        size_text = f"{self.table.rows} rows × {self.table.cols} cols"
        self.size_label = QLabel(size_text)
        self.size_label.setStyleSheet("color: #666; font-size: 11px;")
        info_layout.addWidget(self.size_label)

        layout.addLayout(info_layout, 1)

        # Warnings indicator
        if self.table.warnings:
            warning_label = QLabel("⚠️")
            warning_label.setToolTip("\n".join(self.table.warnings))
            layout.addWidget(warning_label)

    def _update_style(self):
        if self.is_selected:
            self.setStyleSheet("""
                TableListItem {
                    background-color: #e3f2fd;
                    border: 2px solid #2196F3;
                    border-radius: 4px;
                }
            """)
        else:
            self.setStyleSheet("""
                TableListItem {
                    background-color: white;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                TableListItem:hover {
                    background-color: #f5f5f5;
                }
            """)

    def set_selected(self, selected: bool):
        self.is_selected = selected
        self._update_style()

    def update_name(self, name: str):
        self.name_label.setText(name)

    def _on_toggle(self, state: int):
        self.toggled.emit(self.table.id, state == Qt.Checked)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Check if click was not on checkbox
            if not self.checkbox.geometry().contains(event.pos()):
                self.selected.emit(self.table.id)
        super().mousePressEvent(event)


class TableList(QScrollArea):
    """List of detected tables with checkboxes."""

    table_selected = Signal(str)  # table_id
    table_toggled = Signal(str, bool)  # table_id, enabled

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = {}  # table_id -> TableListItem
        self.current_selection = None
        self._setup_ui()

    def _setup_ui(self):
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        # Container
        self.container = QWidget()
        self.setWidget(self.container)

        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        self.layout.addStretch()

    def set_tables(self, tables: List[Table]):
        """Set the list of tables."""
        # Clear existing items
        for item in self.items.values():
            item.deleteLater()
        self.items.clear()

        # Remove stretch
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add new items
        for table in tables:
            item = TableListItem(table)
            item.selected.connect(self._on_item_selected)
            item.toggled.connect(self._on_item_toggled)
            self.items[table.id] = item
            self.layout.addWidget(item)

        self.layout.addStretch()

        # Select first if available
        if tables:
            self._on_item_selected(tables[0].id)

    def _on_item_selected(self, table_id: str):
        # Update selection
        if self.current_selection and self.current_selection in self.items:
            self.items[self.current_selection].set_selected(False)

        self.current_selection = table_id
        if table_id in self.items:
            self.items[table_id].set_selected(True)

        self.table_selected.emit(table_id)

    def _on_item_toggled(self, table_id: str, enabled: bool):
        self.table_toggled.emit(table_id, enabled)

    def update_table_name(self, table_id: str, name: str):
        """Update table name in the list."""
        if table_id in self.items:
            self.items[table_id].update_name(name)
