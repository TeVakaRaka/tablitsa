from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import uuid

from .cell import Cell


@dataclass
class Column:
    """Represents a table column with metadata."""

    index: int
    name: str = ""
    enabled: bool = True
    width: int = 100

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "enabled": self.enabled,
            "width": self.width,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Column":
        return cls(
            index=data["index"],
            name=data.get("name", ""),
            enabled=data.get("enabled", True),
            width=data.get("width", 100),
        )


@dataclass
class Table:
    """Represents an extracted table."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    rows: int = 0
    cols: int = 0
    cells: List[Cell] = field(default_factory=list)
    columns: List[Column] = field(default_factory=list)
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    enabled: bool = True
    has_merged_cells: bool = False
    avg_confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            self.name = f"Table_{self.id}"
        if self.cols > 0 and not self.columns:
            self.columns = [Column(i, f"Col_{i + 1}") for i in range(self.cols)]

    def get_cell(self, row: int, col: int) -> Optional[Cell]:
        """Get cell at specified position."""
        for cell in self.cells:
            if cell.row == row and cell.col == col:
                return cell
        return None

    def set_cell(self, row: int, col: int, text: str):
        """Set cell text at specified position."""
        cell = self.get_cell(row, col)
        if cell:
            cell.text = text
        else:
            self.cells.append(Cell(row=row, col=col, text=text))

    def get_row(self, row_idx: int) -> List[Cell]:
        """Get all cells in a row."""
        return sorted(
            [c for c in self.cells if c.row == row_idx],
            key=lambda c: c.col
        )

    def get_column(self, col_idx: int) -> List[Cell]:
        """Get all cells in a column."""
        return sorted(
            [c for c in self.cells if c.col == col_idx],
            key=lambda c: c.row
        )

    def get_enabled_columns(self) -> List[Column]:
        """Get only enabled columns."""
        return [col for col in self.columns if col.enabled]

    def get_data_matrix(self, enabled_only: bool = True) -> List[List[str]]:
        """Get table data as 2D matrix of strings."""
        enabled_cols = set(
            col.index for col in self.columns if col.enabled
        ) if enabled_only else set(range(self.cols))

        matrix = []
        for row_idx in range(self.rows):
            row_data = []
            for col_idx in range(self.cols):
                if col_idx in enabled_cols:
                    cell = self.get_cell(row_idx, col_idx)
                    row_data.append(cell.text if cell else "")
            matrix.append(row_data)
        return matrix

    def get_headers(self, enabled_only: bool = True) -> List[str]:
        """Get column headers."""
        if enabled_only:
            return [col.name for col in self.columns if col.enabled]
        return [col.name for col in self.columns]

    def calculate_confidence(self):
        """Calculate average confidence score."""
        if not self.cells:
            self.avg_confidence = 0.0
            return
        self.avg_confidence = sum(c.confidence for c in self.cells) / len(self.cells)

    def check_merged_cells(self):
        """Check for merged cells and set warning."""
        self.has_merged_cells = any(c.is_merged for c in self.cells)
        if self.has_merged_cells:
            if "Merged cells detected" not in self.warnings:
                self.warnings.append("Merged cells detected")

    def to_dict(self) -> dict:
        """Serialize table to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "rows": self.rows,
            "cols": self.cols,
            "cells": [cell.to_dict() for cell in self.cells],
            "columns": [col.to_dict() for col in self.columns],
            "bbox": self.bbox,
            "enabled": self.enabled,
            "has_merged_cells": self.has_merged_cells,
            "avg_confidence": self.avg_confidence,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Table":
        """Deserialize table from dictionary."""
        table = cls(
            id=data["id"],
            name=data.get("name", ""),
            rows=data.get("rows", 0),
            cols=data.get("cols", 0),
            bbox=tuple(data["bbox"]) if data.get("bbox") else None,
            enabled=data.get("enabled", True),
            has_merged_cells=data.get("has_merged_cells", False),
            avg_confidence=data.get("avg_confidence", 0.0),
            warnings=data.get("warnings", []),
        )
        table.cells = [Cell.from_dict(c) for c in data.get("cells", [])]
        table.columns = [Column.from_dict(c) for c in data.get("columns", [])]
        return table
