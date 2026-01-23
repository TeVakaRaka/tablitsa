from PySide2.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSplitter, QGroupBox, QCheckBox,
    QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QMenu, QMessageBox,
    QUndoStack, QUndoCommand
)
from PySide2.QtCore import Qt, Signal
from PySide2.QtGui import QColor, QBrush

from ..models import Project, Table, Cell
from .widgets.table_list import TableList
from .widgets.table_view import TableView
from .widgets.cell_inspector import CellInspector
from .widgets.column_toolbar import ColumnToolbar


class CellEditCommand(QUndoCommand):
    """Undo command for cell edits."""

    def __init__(self, table: Table, row: int, col: int, old_text: str, new_text: str):
        super().__init__(f"Edit cell ({row}, {col})")
        self.table = table
        self.row = row
        self.col = col
        self.old_text = old_text
        self.new_text = new_text

    def redo(self):
        cell = self.table.get_cell(self.row, self.col)
        if cell:
            cell.text = self.new_text

    def undo(self):
        cell = self.table.get_cell(self.row, self.col)
        if cell:
            cell.text = self.old_text


class Step2Edit(QWidget):
    """Step 2: Edit extracted tables."""

    table_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.current_table = None
        self.undo_stack = QUndoStack(self)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)

        # Left panel - Table list and warnings
        left_panel = QWidget()
        left_panel.setMaximumWidth(280)
        left_panel.setMinimumWidth(200)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Table list group
        list_group = QGroupBox("Detected Tables")
        list_layout = QVBoxLayout(list_group)

        self.table_list = TableList()
        list_layout.addWidget(self.table_list)

        # Select all / none buttons
        btn_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                font-size: 11px;
            }
        """)
        self.btn_select_none = QPushButton("Select None")
        self.btn_select_none.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                font-size: 11px;
            }
        """)
        btn_layout.addWidget(self.btn_select_all)
        btn_layout.addWidget(self.btn_select_none)
        list_layout.addLayout(btn_layout)

        left_layout.addWidget(list_group)

        # Warnings group
        self.warnings_group = QGroupBox("Warnings")
        self.warnings_layout = QVBoxLayout(self.warnings_group)
        self.warnings_label = QLabel("")
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setStyleSheet("color: #f57c00;")
        self.warnings_layout.addWidget(self.warnings_label)
        self.warnings_group.setVisible(False)
        left_layout.addWidget(self.warnings_group)

        left_layout.addStretch()

        main_splitter.addWidget(left_panel)

        # Center panel - Column toolbar + Table view
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)

        # Toolbar row with name and undo/redo
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.table_name_edit = QLineEdit()
        self.table_name_edit.setPlaceholderText("Table name")
        self.table_name_edit.setMaximumWidth(200)
        self.table_name_edit.setStyleSheet("""
            QLineEdit {
                padding: 6px 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #2196f3;
            }
        """)
        toolbar.addWidget(QLabel("Name:"))
        toolbar.addWidget(self.table_name_edit)

        toolbar.addStretch()

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setEnabled(False)
        self.btn_undo.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background: #f5f5f5;
            }
            QPushButton:hover:enabled {
                background: #e0e0e0;
            }
            QPushButton:disabled {
                color: #999;
            }
        """)
        self.btn_redo = QPushButton("Redo")
        self.btn_redo.setEnabled(False)
        self.btn_redo.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background: #f5f5f5;
            }
            QPushButton:hover:enabled {
                background: #e0e0e0;
            }
            QPushButton:disabled {
                color: #999;
            }
        """)
        toolbar.addWidget(self.btn_undo)
        toolbar.addWidget(self.btn_redo)

        center_layout.addLayout(toolbar)

        # Column toolbar (horizontal)
        self.column_toolbar = ColumnToolbar()
        center_layout.addWidget(self.column_toolbar)

        # Table view + Cell Inspector splitter
        table_inspector_splitter = QSplitter(Qt.Horizontal)

        # Table view
        self.table_view = TableView()
        table_inspector_splitter.addWidget(self.table_view)

        # Cell Inspector
        self.cell_inspector = CellInspector()
        self.cell_inspector.setMinimumWidth(250)
        self.cell_inspector.setMaximumWidth(350)
        table_inspector_splitter.addWidget(self.cell_inspector)

        table_inspector_splitter.setSizes([600, 280])
        center_layout.addWidget(table_inspector_splitter, 1)

        # Legend
        legend = QHBoxLayout()
        legend.setSpacing(12)
        legend.addWidget(self._create_legend_item("#ffcdd2", "Needs review"))
        legend.addWidget(self._create_legend_item("#fff3e0", "Low confidence"))
        legend.addWidget(self._create_legend_item("#e3f2fd", "Edited"))
        legend.addWidget(self._create_legend_item("#fce4ec", "Merged cell"))
        legend.addStretch()
        center_layout.addLayout(legend)

        main_splitter.addWidget(center_panel)
        main_splitter.setSizes([250, 750])

    def _create_legend_item(self, color: str, text: str) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        color_box = QLabel()
        color_box.setFixedSize(14, 14)
        color_box.setStyleSheet(f"background-color: {color}; border: 1px solid #ccc; border-radius: 2px;")
        layout.addWidget(color_box)

        label = QLabel(text)
        label.setStyleSheet("font-size: 11px; color: #666;")
        layout.addWidget(label)

        return widget

    def _connect_signals(self):
        # Table list signals
        self.table_list.table_selected.connect(self._on_table_selected)
        self.table_list.table_toggled.connect(self._on_table_toggled)
        self.btn_select_all.clicked.connect(self._select_all_tables)
        self.btn_select_none.clicked.connect(self._select_none_tables)

        # Table name
        self.table_name_edit.textChanged.connect(self._on_name_changed)

        # Table view signals
        self.table_view.cell_edited.connect(self._on_cell_edited)
        self.table_view.cell_selected.connect(self._on_cell_selected)

        # Column toolbar signals
        self.column_toolbar.column_toggled.connect(self._on_column_toggled)
        self.column_toolbar.column_renamed.connect(self._on_column_renamed)
        self.column_toolbar.select_all_clicked.connect(self._on_columns_select_all)
        self.column_toolbar.select_none_clicked.connect(self._on_columns_select_none)

        # Cell inspector signals
        self.cell_inspector.alternative_applied.connect(self._on_alternative_applied)
        self.cell_inspector.cell_reviewed.connect(self._on_cell_reviewed)

        # Undo/redo
        self.btn_undo.clicked.connect(self.undo_stack.undo)
        self.btn_redo.clicked.connect(self.undo_stack.redo)
        self.undo_stack.canUndoChanged.connect(self.btn_undo.setEnabled)
        self.undo_stack.canRedoChanged.connect(self.btn_redo.setEnabled)

    def set_project(self, project: Project):
        """Set project data."""
        self.project = project
        self.undo_stack.clear()

        # Populate table list
        self.table_list.set_tables(project.tables)

        # Clear inspector
        self.cell_inspector.clear()

        # Select first table
        if project.tables:
            self._on_table_selected(project.tables[0].id)

    def _on_table_selected(self, table_id: str):
        """Handle table selection."""
        if not self.project:
            return

        table = self.project.get_table(table_id)
        if not table:
            return

        self.current_table = table

        # Update name edit
        self.table_name_edit.setText(table.name)

        # Update table view
        self.table_view.set_table(table)

        # Update column toolbar
        self.column_toolbar.set_table(table)

        # Update warnings
        self._update_warnings(table)

        # Clear cell inspector
        self.cell_inspector.clear()

    def _on_table_toggled(self, table_id: str, enabled: bool):
        """Handle table enable/disable toggle."""
        if not self.project:
            return

        table = self.project.get_table(table_id)
        if table:
            table.enabled = enabled
            self.table_changed.emit()

    def _select_all_tables(self):
        """Enable all tables."""
        if self.project:
            for table in self.project.tables:
                table.enabled = True
            self.table_list.set_tables(self.project.tables)

    def _select_none_tables(self):
        """Disable all tables."""
        if self.project:
            for table in self.project.tables:
                table.enabled = False
            self.table_list.set_tables(self.project.tables)

    def _on_name_changed(self, name: str):
        """Handle table name change."""
        if self.current_table:
            self.current_table.name = name
            self.table_list.update_table_name(self.current_table.id, name)

    def _on_cell_edited(self, row: int, col: int, old_text: str, new_text: str):
        """Handle cell edit."""
        if not self.current_table:
            return

        command = CellEditCommand(self.current_table, row, col, old_text, new_text)
        self.undo_stack.push(command)
        self.table_view.refresh()
        self.table_changed.emit()

        # Update cell inspector if this cell is selected
        cell = self.current_table.get_cell(row, col)
        if cell:
            self.cell_inspector.set_cell(cell, row, col)

    def _on_cell_selected(self, cell: Cell, row: int, col: int):
        """Handle cell selection - update inspector."""
        self.cell_inspector.set_cell(cell, row, col)

    def _on_column_toggled(self, index: int, enabled: bool):
        """Handle column enable/disable from toolbar."""
        self.table_view.refresh()
        self.table_changed.emit()

    def _on_column_renamed(self, index: int, name: str):
        """Handle column rename from toolbar."""
        self.table_view.update_headers()
        self.table_changed.emit()

    def _on_columns_select_all(self):
        """Handle select all columns."""
        self.table_view.refresh()
        self.table_changed.emit()

    def _on_columns_select_none(self):
        """Handle select none columns."""
        self.table_view.refresh()
        self.table_changed.emit()

    def _on_alternative_applied(self, row: int, col: int, new_text: str):
        """Handle alternative selection from cell inspector."""
        if not self.current_table:
            return

        cell = self.current_table.get_cell(row, col)
        if cell:
            old_text = cell.text

            # Update via table view (this will trigger cell_edited)
            self.table_view.update_cell_text(row, col, new_text)

    def _on_cell_reviewed(self, row: int, col: int):
        """Handle cell marked as reviewed from inspector."""
        if not self.current_table:
            return

        self.table_view.mark_cell_reviewed(row, col)
        self.table_changed.emit()

    def _update_warnings(self, table: Table):
        """Update warnings display."""
        if table.warnings:
            self.warnings_label.setText("\n".join(f"• {w}" for w in table.warnings))
            self.warnings_group.setVisible(True)
        else:
            self.warnings_group.setVisible(False)
