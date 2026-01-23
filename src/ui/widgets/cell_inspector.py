from PySide2.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QGroupBox, QSizePolicy
)
from PySide2.QtCore import Qt, Signal
from PySide2.QtGui import QFont

from ...models.cell import Cell, OCRAlternative
from ...core.multi_ocr import OCR_WEIGHTS


class AlternativeWidget(QFrame):
    """Widget displaying a single OCR alternative."""

    apply_clicked = Signal(OCRAlternative)

    # Colors for different OCR sources
    SOURCE_COLORS = {
        "paddleocr": "#4caf50",  # Green - primary
        "easyocr": "#2196f3",    # Blue - secondary
        "tesseract": "#ff9800",  # Orange - tertiary
    }

    def __init__(self, alternative: OCRAlternative, is_current: bool = False, parent=None):
        super().__init__(parent)
        self.alternative = alternative
        self.is_current = is_current
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header: source name + indicator
        header_layout = QHBoxLayout()

        # Source indicator (radio-like)
        indicator = QLabel("●" if self.is_current else "○")
        indicator.setFont(QFont("", 10))
        source_color = self.SOURCE_COLORS.get(self.alternative.source, "#999")
        indicator.setStyleSheet(f"color: {source_color}; font-weight: bold;")
        header_layout.addWidget(indicator)

        # Source name with weight
        weight = OCR_WEIGHTS.get(self.alternative.source, 1.0)
        source_label = QLabel(f"{self.alternative.source.upper()}")
        source_label.setFont(QFont("", 10, QFont.Bold))
        source_label.setStyleSheet(f"color: {source_color};")
        header_layout.addWidget(source_label)

        header_layout.addStretch()

        # Confidence badge
        conf_pct = int(self.alternative.confidence * 100)
        conf_color = "#4caf50" if conf_pct >= 80 else "#ff9800" if conf_pct >= 60 else "#f44336"
        conf_label = QLabel(f"{conf_pct}%")
        conf_label.setStyleSheet(f"""
            background-color: {conf_color};
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: bold;
        """)
        header_layout.addWidget(conf_label)

        layout.addLayout(header_layout)

        # Text content
        text_label = QLabel(f'"{self.alternative.text}"')
        text_label.setWordWrap(True)
        text_label.setStyleSheet("color: #333; font-size: 12px; padding: 4px 0;")
        layout.addWidget(text_label)

        # Apply button (hidden if this is current)
        if not self.is_current:
            apply_btn = QPushButton("Apply")
            apply_btn.setFixedHeight(24)
            apply_btn.setCursor(Qt.PointingHandCursor)
            apply_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e3f2fd;
                    border: 1px solid #90caf9;
                    border-radius: 3px;
                    padding: 2px 12px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #bbdefb;
                }
                QPushButton:pressed {
                    background-color: #90caf9;
                }
            """)
            apply_btn.clicked.connect(lambda: self.apply_clicked.emit(self.alternative))
            layout.addWidget(apply_btn, alignment=Qt.AlignRight)

        # Frame styling
        border_color = self.SOURCE_COLORS.get(self.alternative.source, "#ddd")
        bg_color = "#f5fff5" if self.is_current else "#fafafa"
        self.setStyleSheet(f"""
            AlternativeWidget {{
                background-color: {bg_color};
                border: 1px solid {border_color if self.is_current else '#ddd'};
                border-radius: 4px;
                border-left: 3px solid {border_color};
            }}
        """)


class CellInspector(QWidget):
    """Inspector panel showing OCR details for selected cell."""

    alternative_applied = Signal(int, int, str)  # row, col, new_text
    cell_reviewed = Signal(int, int)  # row, col

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_cell = None
        self.current_row = -1
        self.current_col = -1
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Title
        title = QLabel("Cell Inspector")
        title.setFont(QFont("", 12, QFont.Bold))
        title.setStyleSheet("color: #333;")
        layout.addWidget(title)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #ddd;")
        layout.addWidget(sep)

        # Current value section
        self.current_group = QGroupBox("Current Value")
        self.current_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        current_layout = QVBoxLayout(self.current_group)

        self.current_text_label = QLabel("")
        self.current_text_label.setWordWrap(True)
        self.current_text_label.setStyleSheet("font-size: 13px; color: #333; padding: 4px;")
        current_layout.addWidget(self.current_text_label)

        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setStyleSheet("font-size: 11px; color: #666;")
        current_layout.addWidget(self.confidence_label)

        layout.addWidget(self.current_group)

        # Alternatives section
        alt_label = QLabel("OCR Alternatives")
        alt_label.setFont(QFont("", 11, QFont.Bold))
        alt_label.setStyleSheet("color: #333; margin-top: 8px;")
        layout.addWidget(alt_label)

        # Scroll area for alternatives
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)

        self.alternatives_container = QWidget()
        self.alternatives_layout = QVBoxLayout(self.alternatives_container)
        self.alternatives_layout.setContentsMargins(0, 0, 0, 0)
        self.alternatives_layout.setSpacing(8)
        self.alternatives_layout.addStretch()

        scroll.setWidget(self.alternatives_container)
        layout.addWidget(scroll, 1)

        # Mark as reviewed button
        self.review_btn = QPushButton("Mark as Reviewed")
        self.review_btn.setFixedHeight(32)
        self.review_btn.setCursor(Qt.PointingHandCursor)
        self.review_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #43a047;
            }
            QPushButton:pressed {
                background-color: #388e3c;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self.review_btn.clicked.connect(self._on_review_clicked)
        layout.addWidget(self.review_btn)

        # Empty state label
        self.empty_label = QLabel("Select a cell to inspect")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color: #999; font-size: 12px;")
        layout.addWidget(self.empty_label)

        # Initial state
        self._show_empty_state()

        # Set minimum width
        self.setMinimumWidth(250)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

    def set_cell(self, cell: Cell, row: int, col: int):
        """Set cell to inspect."""
        self.current_cell = cell
        self.current_row = row
        self.current_col = col

        if cell is None:
            self._show_empty_state()
            return

        self._show_cell_details()

    def _show_empty_state(self):
        """Show empty state when no cell selected."""
        self.current_group.hide()
        self.review_btn.hide()
        self.empty_label.show()

        # Clear alternatives
        self._clear_alternatives()

    def _show_cell_details(self):
        """Show cell details."""
        self.empty_label.hide()
        self.current_group.show()

        cell = self.current_cell

        # Update current value
        display_text = f'"{cell.text}"' if cell.text else "(empty)"
        self.current_text_label.setText(display_text)

        # Update confidence
        conf_pct = int(cell.confidence * 100)
        conf_color = "#4caf50" if conf_pct >= 80 else "#ff9800" if conf_pct >= 60 else "#f44336"
        self.confidence_label.setText(f"Confidence: <span style='color: {conf_color}; font-weight: bold;'>{conf_pct}%</span>")

        # Update review button
        if cell.needs_review:
            self.review_btn.show()
            self.review_btn.setEnabled(True)
        else:
            self.review_btn.hide()

        # Update alternatives
        self._update_alternatives()

    def _clear_alternatives(self):
        """Clear alternatives list."""
        while self.alternatives_layout.count() > 1:  # Keep stretch
            item = self.alternatives_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _update_alternatives(self):
        """Update alternatives list."""
        self._clear_alternatives()

        if not self.current_cell or not self.current_cell.alternatives:
            no_alt_label = QLabel("No alternatives available")
            no_alt_label.setStyleSheet("color: #999; font-size: 11px; padding: 8px;")
            self.alternatives_layout.insertWidget(0, no_alt_label)
            return

        # Sort alternatives by weight (highest first)
        sorted_alts = sorted(
            self.current_cell.alternatives,
            key=lambda a: OCR_WEIGHTS.get(a.source, 1.0),
            reverse=True
        )

        # Add alternative widgets
        for i, alt in enumerate(sorted_alts):
            is_current = alt.text == self.current_cell.text
            widget = AlternativeWidget(alt, is_current=is_current)
            widget.apply_clicked.connect(self._on_alternative_applied)
            self.alternatives_layout.insertWidget(i, widget)

    def _on_alternative_applied(self, alternative: OCRAlternative):
        """Handle alternative selection."""
        if self.current_cell is None:
            return

        self.alternative_applied.emit(
            self.current_row,
            self.current_col,
            alternative.text
        )

        # Update display
        self.current_cell.text = alternative.text
        self.current_cell.needs_review = False
        self._show_cell_details()

    def _on_review_clicked(self):
        """Handle review button click."""
        if self.current_cell is None:
            return

        self.cell_reviewed.emit(self.current_row, self.current_col)
        self.current_cell.needs_review = False
        self.review_btn.hide()

    def clear(self):
        """Clear inspector."""
        self.current_cell = None
        self.current_row = -1
        self.current_col = -1
        self._show_empty_state()
