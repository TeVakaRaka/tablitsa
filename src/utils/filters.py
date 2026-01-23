from typing import List, Callable
from ..models import Table


class TableFilter:
    """Filters for detected tables."""

    @staticmethod
    def by_min_area(tables: List[Table], min_area: int = 5000) -> List[Table]:
        """Filter tables by minimum area."""
        result = []
        for table in tables:
            if table.bbox:
                area = table.bbox[2] * table.bbox[3]  # width * height
                if area >= min_area:
                    result.append(table)
            else:
                result.append(table)  # Keep if no bbox
        return result

    @staticmethod
    def by_min_size(
        tables: List[Table],
        min_rows: int = 2,
        min_cols: int = 2
    ) -> List[Table]:
        """Filter tables by minimum row/column count."""
        return [
            t for t in tables
            if t.rows >= min_rows and t.cols >= min_cols
        ]

    @staticmethod
    def by_min_cells(tables: List[Table], min_cells: int = 4) -> List[Table]:
        """Filter tables by minimum cell count."""
        return [t for t in tables if len(t.cells) >= min_cells]

    @staticmethod
    def by_text_fill(tables: List[Table], min_fill: float = 0.1) -> List[Table]:
        """Filter tables by minimum text fill ratio.

        Args:
            min_fill: Minimum ratio of non-empty cells (0.0 to 1.0)
        """
        result = []
        for table in tables:
            if not table.cells:
                continue

            non_empty = sum(1 for c in table.cells if c.text.strip())
            fill_ratio = non_empty / len(table.cells)

            if fill_ratio >= min_fill:
                result.append(table)

        return result

    @staticmethod
    def by_confidence(tables: List[Table], min_confidence: float = 0.3) -> List[Table]:
        """Filter tables by minimum average OCR confidence."""
        return [t for t in tables if t.avg_confidence >= min_confidence]

    @staticmethod
    def apply_filters(
        tables: List[Table],
        filters: List[Callable[[List[Table]], List[Table]]]
    ) -> List[Table]:
        """Apply multiple filters in sequence."""
        result = tables
        for f in filters:
            result = f(result)
        return result


def create_default_filter() -> Callable[[List[Table]], List[Table]]:
    """Create default filter chain."""
    def filter_chain(tables: List[Table]) -> List[Table]:
        return TableFilter.apply_filters(tables, [
            lambda t: TableFilter.by_min_size(t, 2, 2),
            lambda t: TableFilter.by_min_area(t, 5000),
            lambda t: TableFilter.by_text_fill(t, 0.1),
        ])
    return filter_chain
