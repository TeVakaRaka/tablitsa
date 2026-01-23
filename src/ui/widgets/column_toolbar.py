from PySide2.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QScrollArea,
    QCheckBox, QLineEdit, QLabel, QFrame, QPushButton,
    QSizePolicy
)
from PySide2.QtCore import Qt, Signal
from PySide2.QtGui import QFont

from typing import List, Optional
from ...models.table import Table, Column


class ColumnChip(QFrame):
    """Single column chip with checkbox and editable name."""

    toggled = Signal(int, bool)  # column index, enabled
    name_changed = Signal(int, str)  # column index, new name

    def __init__(self, column: Column, parent=None):
        super().__init__(parent)
        self.column = column
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        # Checkbox
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(self.column.enabled)
        self.checkbox.toggled.connect(self._on_toggled)
        layout.addWidget(self.checkbox)

        # Editable name
        self.name_edit = QLineEdit(self.column.name)
        self.name_edit.setMaximumWidth(100)
        self.name_edit.setMinimumWidth(60)
        self.name_edit.editingFinished.connect(self._on_name_changed)
        self.name_edit.setStyleSheet("""
            QLineEdit {
                border: none;
                background: transparent;
                padding: 2px 4px;
                font-size: 11px;
            }
            QLineEdit:focus {
                background: white;
                border: 1px solid #90caf9;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.name_edit)

        self._update_style()

    def _update_style(self):
        if self.column.enabled:
            self.setStyleSheet("""
                ColumnChip {
                    background-color: #e3f2fd;
                    border: 1px solid #90caf9;
                    border-radius: 4px;
                }
                ColumnChip:hover {
                    background-color: #bbdefb;
                }
            """)
            self.name_edit.setStyleSheet("""
                QLineEdit {
                    border: none;
                    background: transparent;
                    padding: 2px 4px;
                    font-size: 11px;
                    color: #333;
                }
                QLineEdit:focus {
                    background: white;
                    border: 1px solid #90caf9;
                    border-radius: 2px;
                }
            """)
        else:
            self.setStyleSheet("""
                ColumnChip {
                    background-color: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                ColumnChip:hover {
                    background-color: #eee;
                }
            """)
            self.name_edit.setStyleSheet("""
                QLineEdit {
                    border: none;
                    background: transparent;
                    padding: 2px 4px;
                    font-size: 11px;
                    color: #999;
                }
                QLineEdit:focus {
                    background: white;
                    border: 1px solid #90caf9;
                    border-radius: 2px;
                    color: #333;
                }
            """)

    def _on_toggled(self, checked: bool):
        self.column.enabled = checked
        self._update_style()
        self.toggled.emit(self.column.index, checked)

    def _on_name_changed(self):
        new_name = self.name_edit.text().strip()
        if new_name and new_name != self.column.name:
            self.column.name = new_name
            self.name_changed.emit(self.column.index, new_name)

    def update_from_column(self):
        """Update UI from column data."""
        self.checkbox.blockSignals(True)
        self.checkbox.setChecked(self.column.enabled)
        self.checkbox.blockSignals(False)
        self.name_edit.setText(self.column.name)
        self._update_style()


class ColumnToolbar(QWidget):
    """Horizontal toolbar for column management."""

    column_toggled = Signal(int, bool)  # column index, enabled
    column_renamed = Signal(int, str)  # column index, new name
    select_all_clicked = Signal()
    select_none_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table: Optional[Table] = None
        self.column_chips: List[ColumnChip] = []
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        # Header with label and buttons
        header = QHBoxLayout()

        label = QLabel("Columns")
        label.setFont(QFont("", 11, QFont.Bold))
        label.setStyleSheet("color: #333;")
        header.addWidget(label)

        header.addStretch()

        # Select All / None buttons
        self.btn_select_all = QPushButton("All")
        self.btn_select_all.setFixedHeight(24)
        self.btn_select_all.setCursor(Qt.PointingHandCursor)
        self.btn_select_all.setStyleSheet("""
            QPushButton {
                background-color: #e8f5e9;
                border: 1px solid #a5d6a7;
                border-radius: 3px;
                padding: 2px 10px;
                font-size: 11px;
                color: #2e7d32;
            }
            QPushButton:hover {
                background-color: #c8e6c9;
            }
        """)
        self.btn_select_all.clicked.connect(self._on_select_all)
        header.addWidget(self.btn_select_all)

        self.btn_select_none = QPushButton("None")
        self.btn_select_none.setFixedHeight(24)
        self.btn_select_none.setCursor(Qt.PointingHandCursor)
        self.btn_select_none.setStyleSheet("""
            QPushButton {
                background-color: #ffebee;
                border: 1px solid #ef9a9a;
                border-radius: 3px;
                padding: 2px 10px;
                font-size: 11px;
                color: #c62828;
            }
            QPushButton:hover {
                background-color: #ffcdd2;
            }
        """)
        self.btn_select_none.clicked.connect(self._on_select_none)
        header.addWidget(self.btn_select_none)

        main_layout.addLayout(header)

        # Scrollable columns area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFixedHeight(50)
        scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-radius: 4px;
                background: #fafafa;
            }
        """)

        self.columns_container = QWidget()
        self.columns_layout = QHBoxLayout(self.columns_container)
        self.columns_layout.setContentsMargins(4, 4, 4, 4)
        self.columns_layout.setSpacing(6)
        self.columns_layout.addStretch()

        scroll.setWidget(self.columns_container)
        main_layout.addWidget(scroll)

        self.setFixedHeight(80)

    def set_table(self, table: Optional[Table]):
        """Set table to display columns for."""
        self.table = table
        self._rebuild_columns()

    def _rebuild_columns(self):
        """Rebuild column chips from table."""
        # Clear existing chips
        for chip in self.column_chips:
            chip.deleteLater()
        self.column_chips.clear()

        # Remove stretch
        while self.columns_layout.count() > 0:
            item = self.columns_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.table or not self.table.columns:
            empty_label = QLabel("No columns")
            empty_label.setStyleSheet("color: #999; font-size: 11px;")
            self.columns_layout.addWidget(empty_label)
            self.columns_layout.addStretch()
            return

        # Create chips for each column
        for column in self.table.columns:
            chip = ColumnChip(column)
            chip.toggled.connect(self._on_column_toggled)
            chip.name_changed.connect(self._on_column_renamed)
            self.column_chips.append(chip)
            self.columns_layout.addWidget(chip)

        self.columns_layout.addStretch()

    def _on_column_toggled(self, index: int, enabled: bool):
        self.column_toggled.emit(index, enabled)

    def _on_column_renamed(self, index: int, name: str):
        self.column_renamed.emit(index, name)

    def _on_select_all(self):
        if not self.table:
            return
        for col in self.table.columns:
            col.enabled = True
        for chip in self.column_chips:
            chip.update_from_column()
        self.select_all_clicked.emit()

    def _on_select_none(self):
        if not self.table:
            return
        for col in self.table.columns:
            col.enabled = False
        for chip in self.column_chips:
            chip.update_from_column()
        self.select_none_clicked.emit()

    def refresh(self):
        """Refresh column display."""
        for chip in self.column_chips:
            chip.update_from_column()
