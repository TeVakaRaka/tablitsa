from PySide2.QtWidgets import (
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMenu, QAction
)
from PySide2.QtCore import Qt, Signal
from PySide2.QtGui import QColor, QBrush, QFont

from ...models import Table, Cell
from ...core.multi_ocr import OCR_WEIGHTS


class TableView(QTableWidget):
    """Editable table view for extracted data."""

    cell_edited = Signal(int, int, str, str)  # row, col, old_text, new_text
    cell_selected = Signal(object, int, int)  # Cell object, row, col

    # Colors for different OCR sources
    SOURCE_COLORS = {
        "paddleocr": "#4caf50",  # Green
        "easyocr": "#2196f3",    # Blue
        "tesseract": "#ff9800",  # Orange
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setSelectionMode(QAbstractItemView.SingleSelection)

        # Headers
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(True)

        # Context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # Style
        self.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #bbdefb;
                color: black;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 6px;
                border: none;
                border-bottom: 1px solid #ddd;
                border-right: 1px solid #ddd;
            }
        """)

    def _connect_signals(self):
        self.cellChanged.connect(self._on_cell_changed)
        self.currentCellChanged.connect(self._on_current_cell_changed)

    def _on_current_cell_changed(self, row: int, col: int, prev_row: int, prev_col: int):
        """Handle cell selection change."""
        if not self.table:
            return

        cell = self.table.get_cell(row, col)
        self.cell_selected.emit(cell, row, col)

    def set_table(self, table: Table):
        """Set table to display."""
        self.table = table
        self._populate()

    def _populate(self):
        """Populate table widget with data."""
        if not self.table:
            return

        self.blockSignals(True)

        # Set dimensions
        self.setRowCount(self.table.rows)
        self.setColumnCount(self.table.cols)

        # Set headers
        headers = []
        for col in self.table.columns:
            headers.append(col.name if col.enabled else f"[{col.name}]")
        self.setHorizontalHeaderLabels(headers)

        # Populate cells
        for cell in self.table.cells:
            item = QTableWidgetItem(cell.text)

            # Style based on cell state (priority order)
            if cell.needs_review:
                # Needs review - red border effect via background
                item.setBackground(QBrush(QColor("#ffcdd2")))  # Light red
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                # Add detailed tooltip with all OCR results
                item.setToolTip(self._build_alternatives_tooltip(cell, disagreed=True))
            elif cell.is_merged:
                item.setBackground(QBrush(QColor("#fce4ec")))
            elif cell.is_edited:
                item.setBackground(QBrush(QColor("#e3f2fd")))
            elif cell.has_low_confidence:
                item.setBackground(QBrush(QColor("#fff3e0")))

            # Always show alternatives tooltip if available (except for needs_review which has its own)
            if cell.alternatives and not cell.needs_review:
                item.setToolTip(self._build_alternatives_tooltip(cell, disagreed=False))

            # Store original text
            item.setData(Qt.UserRole, cell.text)

            self.setItem(cell.row, cell.col, item)

        # Gray out disabled columns
        for col in self.table.columns:
            if not col.enabled:
                for row in range(self.table.rows):
                    item = self.item(row, col.index)
                    if item:
                        item.setForeground(QBrush(QColor("#999")))

        self.blockSignals(False)

    def _build_alternatives_tooltip(self, cell: Cell, disagreed: bool = False) -> str:
        """Build rich tooltip showing all OCR alternatives."""
        if not cell.alternatives:
            return ""

        # Sort alternatives by weight (primary first)
        sorted_alts = sorted(
            cell.alternatives,
            key=lambda a: OCR_WEIGHTS.get(a.source, 1.0),
            reverse=True
        )

        lines = []
        if disagreed:
            lines.append("⚠ OCR engines disagreed:")
        else:
            lines.append("OCR Results:")

        lines.append("")

        for alt in sorted_alts:
            conf_pct = int(alt.confidence * 100)
            is_current = alt.text == cell.text
            marker = "✓" if is_current else " "
            source_name = alt.source.upper()
            lines.append(f"{marker} {source_name}: \"{alt.text}\" ({conf_pct}%)")

        return "\n".join(lines)

    def update_headers(self):
        """Update column headers."""
        if not self.table:
            return

        headers = []
        for col in self.table.columns:
            headers.append(col.name if col.enabled else f"[{col.name}]")
        self.setHorizontalHeaderLabels(headers)

    def refresh(self):
        """Refresh table display."""
        self._populate()

    def _on_cell_changed(self, row: int, col: int):
        """Handle cell edit."""
        if not self.table:
            return

        item = self.item(row, col)
        if not item:
            return

        old_text = item.data(Qt.UserRole) or ""
        new_text = item.text()

        if old_text != new_text:
            # Update the model
            cell = self.table.get_cell(row, col)
            if cell:
                cell.text = new_text

            # Update item data
            item.setData(Qt.UserRole, new_text)

            # Mark as edited
            item.setBackground(QBrush(QColor("#e3f2fd")))

            # Emit signal
            self.cell_edited.emit(row, col, old_text, new_text)

    def _show_context_menu(self, pos):
        """Show context menu."""
        item = self.itemAt(pos)
        if not item:
            return

        row = item.row()
        col = item.column()
        cell = self.table.get_cell(row, col) if self.table else None

        menu = QMenu(self)

        # Show alternatives if available
        if cell and cell.alternatives:
            alt_menu = menu.addMenu("OCR Alternatives")

            # Sort by weight (primary first)
            sorted_alts = sorted(
                cell.alternatives,
                key=lambda a: OCR_WEIGHTS.get(a.source, 1.0),
                reverse=True
            )

            for alt in sorted_alts:
                conf_pct = int(alt.confidence * 100)
                is_current = alt.text == cell.text
                marker = "✓ " if is_current else "   "
                alt_text = f"{marker}{alt.source.upper()}: \"{alt.text}\" ({conf_pct}%)"
                action = alt_menu.addAction(alt_text)

                # Disable if this is already the current value
                if is_current:
                    action.setEnabled(False)
                else:
                    # Capture alt.text in lambda
                    action.triggered.connect(
                        lambda checked, t=alt.text: self._apply_alternative(item, t)
                    )
            menu.addSeparator()

        # Reset cell
        reset_action = menu.addAction("Reset to OCR result")
        reset_action.triggered.connect(lambda: self._reset_cell(item))

        # Mark as reviewed
        if cell and cell.needs_review:
            review_action = menu.addAction("Mark as reviewed")
            review_action.triggered.connect(lambda: self._mark_reviewed(item))

        menu.addSeparator()

        # Copy
        copy_action = menu.addAction("Copy")
        copy_action.triggered.connect(lambda: self._copy_cell(item))

        menu.exec_(self.mapToGlobal(pos))

    def _apply_alternative(self, item: QTableWidgetItem, text: str):
        """Apply alternative OCR result to cell."""
        if not self.table:
            return

        row = item.row()
        col = item.column()
        cell = self.table.get_cell(row, col)

        if cell:
            old_text = item.text()
            cell.text = text
            cell.needs_review = False
            item.setText(text)
            item.setData(Qt.UserRole, text)

            # Update background
            item.setBackground(QBrush(QColor("#e3f2fd")))  # Edited color
            font = item.font()
            font.setBold(False)
            item.setFont(font)

            self.cell_edited.emit(row, col, old_text, text)

    def _mark_reviewed(self, item: QTableWidgetItem):
        """Mark cell as reviewed (no longer needs attention)."""
        if not self.table:
            return

        row = item.row()
        col = item.column()
        cell = self.table.get_cell(row, col)

        if cell:
            cell.needs_review = False

            # Update background
            if cell.has_low_confidence:
                item.setBackground(QBrush(QColor("#fff3e0")))
            else:
                item.setBackground(QBrush(QColor("white")))

            font = item.font()
            font.setBold(False)
            item.setFont(font)

    def _reset_cell(self, item: QTableWidgetItem):
        """Reset cell to original OCR value."""
        if not self.table:
            return

        row = item.row()
        col = item.column()
        cell = self.table.get_cell(row, col)

        if cell:
            old_text = item.text()
            cell.text = cell.original_text
            item.setText(cell.original_text)
            item.setData(Qt.UserRole, cell.original_text)

            # Update background
            if cell.has_low_confidence:
                item.setBackground(QBrush(QColor("#fff3e0")))
            else:
                item.setBackground(QBrush(QColor("white")))

            self.cell_edited.emit(row, col, old_text, cell.original_text)

    def _copy_cell(self, item: QTableWidgetItem):
        """Copy cell text to clipboard."""
        from PySide2.QtWidgets import QApplication
        QApplication.clipboard().setText(item.text())

    def update_cell_text(self, row: int, col: int, text: str):
        """Update cell text from external source (e.g., CellInspector)."""
        if not self.table:
            return

        item = self.item(row, col)
        if not item:
            return

        old_text = item.text()
        if old_text == text:
            return

        self.blockSignals(True)

        # Update item
        item.setText(text)
        item.setData(Qt.UserRole, text)

        # Update model
        cell = self.table.get_cell(row, col)
        if cell:
            cell.text = text

        # Mark as edited
        item.setBackground(QBrush(QColor("#e3f2fd")))
        font = item.font()
        font.setBold(False)
        item.setFont(font)

        self.blockSignals(False)

        # Emit signal
        self.cell_edited.emit(row, col, old_text, text)

    def mark_cell_reviewed(self, row: int, col: int):
        """Mark cell as reviewed from external source."""
        if not self.table:
            return

        item = self.item(row, col)
        cell = self.table.get_cell(row, col)

        if item and cell:
            cell.needs_review = False

            # Update background
            if cell.is_edited:
                item.setBackground(QBrush(QColor("#e3f2fd")))
            elif cell.has_low_confidence:
                item.setBackground(QBrush(QColor("#fff3e0")))
            else:
                item.setBackground(QBrush(QColor("white")))

            font = item.font()
            font.setBold(False)
            item.setFont(font)
