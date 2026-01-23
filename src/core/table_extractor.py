import cv2
import numpy as np
from typing import List, Optional

from ..models import Table, Cell
from .table_detector import TableDetector, DetectedTable, HybridTableDetector
from .ocr_engine import OCREngine
from .multi_ocr import MultiOCREngine, EASYOCR_AVAILABLE
from .preprocessing import ImagePreprocessor
from ..utils.config import DetectorConfig, get_detector_config


class TableExtractor:
    """Extracts tables from images using detection and OCR."""

    def __init__(
        self,
        languages: str = "rus+eng",
        tessdata_path: Optional[str] = None,
        use_multi_ocr: bool = True,
        use_hybrid_detector: bool = True,
        detector_config: Optional[DetectorConfig] = None,
    ):
        # Use provided config or get global config
        self.detector_config = detector_config or get_detector_config()

        # Use hybrid detector by default for better accuracy
        if use_hybrid_detector:
            self.detector = HybridTableDetector(config=self.detector_config)
        else:
            self.detector = TableDetector(config=self.detector_config)

        self.use_hybrid = use_hybrid_detector
        self.ocr = OCREngine(languages=languages, tessdata_path=tessdata_path)
        self.preprocessor = ImagePreprocessor()
        self.use_multi_ocr = use_multi_ocr and EASYOCR_AVAILABLE

        # Initialize multi-OCR if available
        if self.use_multi_ocr:
            # Convert tesseract format to list
            lang_list = languages.replace("+", ",").split(",")
            lang_map = {"rus": "ru", "eng": "en", "deu": "de", "fra": "fr"}
            lang_list = [lang_map.get(l.strip(), l.strip()) for l in lang_list]
            self.multi_ocr = MultiOCREngine(languages=lang_list)
        else:
            self.multi_ocr = None

    def extract(
        self,
        image: np.ndarray,
        detect_lineless: bool = True,
        use_yolo: Optional[bool] = None,
    ) -> List[Table]:
        """
        Extract all tables from image.

        Args:
            image: Input image as numpy array
            detect_lineless: Whether to detect tables without visible lines
            use_yolo: Override YOLO usage (None = use config setting)

        Returns:
            List of extracted Table objects with OCR'd content.
        """
        tables = []

        # Detect tables with lines (uses hybrid approach if enabled)
        if self.use_hybrid and hasattr(self.detector, 'detect'):
            detected = self.detector.detect(image, use_yolo=use_yolo)
        else:
            detected = self.detector.detect(image)

        # If no lined tables found and lineless detection is enabled
        if not detected and detect_lineless:
            text_blocks = self.ocr.get_text_blocks(image)
            if self.use_hybrid and hasattr(self.detector, 'detect_lineless'):
                # Use improved graph-based detection
                detected = self.detector.detect_lineless(
                    image, text_blocks, use_improved=True
                )
            else:
                detected = self.detector.detect_lineless(image, text_blocks)

        # Process each detected table
        for i, det_table in enumerate(detected):
            table = self._process_detected_table(image, det_table, i + 1)
            if table:
                tables.append(table)

        return tables

    def _process_detected_table(
        self, image: np.ndarray, det_table: DetectedTable, index: int
    ) -> Optional[Table]:
        """Process a detected table region with OCR."""
        if det_table.rows < 2 or det_table.cols < 2:
            return None

        # Create table model
        table = Table(
            name=f"Table_{index}",
            rows=det_table.rows,
            cols=det_table.cols,
            bbox=det_table.bbox,
        )

        # OCR each cell
        cells = []
        has_merged = False
        total_confidence = 0.0
        needs_review_count = 0

        for i, cell_bbox in enumerate(det_table.cell_bboxes):
            row = i // det_table.cols
            col = i % det_table.cols

            # Check for potential merged cell
            is_merged = self._check_merged_cell(
                det_table.cell_bboxes, i, det_table.rows, det_table.cols
            )
            if is_merged:
                has_merged = True

            # Run OCR on cell (multi-OCR or single)
            if self.multi_ocr:
                result = self.multi_ocr.recognize(image, cell_bbox)
                cell = Cell(
                    row=row,
                    col=col,
                    text=result.text,
                    confidence=result.confidence,
                    bbox=cell_bbox,
                    is_merged=is_merged,
                    alternatives=result.alternatives,
                    needs_review=result.needs_review,
                )
                if result.needs_review:
                    needs_review_count += 1
            else:
                result = self.ocr.recognize_table_cell(image, cell_bbox)
                cell = Cell(
                    row=row,
                    col=col,
                    text=result.text,
                    confidence=result.confidence,
                    bbox=cell_bbox,
                    is_merged=is_merged,
                )

            cells.append(cell)
            total_confidence += cell.confidence

        table.cells = cells
        table.avg_confidence = total_confidence / len(cells) if cells else 0.0
        table.has_merged_cells = has_merged

        # Add warnings
        if has_merged:
            table.warnings.append("Merged cells detected - data may need manual review")
        if table.avg_confidence < 0.6:
            table.warnings.append("Low OCR confidence - verify extracted text")
        if not det_table.has_lines:
            table.warnings.append("Table detected without visible lines - structure may need verification")
        if needs_review_count > 0:
            table.warnings.append(f"{needs_review_count} cells need review - OCR engines disagreed")

        return table

    def _check_merged_cell(
        self,
        cell_bboxes: List,
        index: int,
        rows: int,
        cols: int,
    ) -> bool:
        """Check if a cell might be merged based on its size."""
        if index >= len(cell_bboxes):
            return False

        current_bbox = cell_bboxes[index]
        current_area = current_bbox[2] * current_bbox[3]

        # Calculate average cell area
        areas = [bbox[2] * bbox[3] for bbox in cell_bboxes]
        avg_area = sum(areas) / len(areas) if areas else 0

        # If cell is significantly larger than average, it might be merged
        return current_area > avg_area * 1.5

    def extract_from_file(
        self, file_path: str, detect_lineless: bool = True
    ) -> List[Table]:
        """Extract tables from image file."""
        image = ImagePreprocessor.load_image(file_path)
        return self.extract(image, detect_lineless)

    def set_languages(self, languages: str):
        """Set OCR languages."""
        self.ocr.set_languages(languages)


class TableProcessor:
    """Processes extracted tables for export."""

    @staticmethod
    def expand_merged_cells(table: Table) -> Table:
        """Expand merged cells by duplicating content to all covered cells."""
        # This is a simplified implementation
        # Full implementation would analyze cell spans
        for cell in table.cells:
            if cell.is_merged and cell.merge_span != (1, 1):
                row_span, col_span = cell.merge_span
                for dr in range(row_span):
                    for dc in range(col_span):
                        if dr == 0 and dc == 0:
                            continue
                        # Create duplicate cells
                        new_cell = Cell(
                            row=cell.row + dr,
                            col=cell.col + dc,
                            text=cell.text,
                            confidence=cell.confidence,
                            is_merged=True,
                        )
                        table.cells.append(new_cell)
        return table

    @staticmethod
    def filter_empty_rows(table: Table) -> Table:
        """Remove rows that are completely empty."""
        non_empty_rows = set()
        for cell in table.cells:
            if cell.text.strip():
                non_empty_rows.add(cell.row)

        if len(non_empty_rows) == table.rows:
            return table

        # Remap rows
        row_mapping = {old: new for new, old in enumerate(sorted(non_empty_rows))}

        new_cells = []
        for cell in table.cells:
            if cell.row in row_mapping:
                cell.row = row_mapping[cell.row]
                new_cells.append(cell)

        table.cells = new_cells
        table.rows = len(non_empty_rows)

        return table

    @staticmethod
    def filter_empty_columns(table: Table) -> Table:
        """Remove columns that are completely empty."""
        non_empty_cols = set()
        for cell in table.cells:
            if cell.text.strip():
                non_empty_cols.add(cell.col)

        if len(non_empty_cols) == table.cols:
            return table

        # Remap columns
        col_mapping = {old: new for new, old in enumerate(sorted(non_empty_cols))}

        new_cells = []
        for cell in table.cells:
            if cell.col in col_mapping:
                cell.col = col_mapping[cell.col]
                new_cells.append(cell)

        table.cells = new_cells
        table.cols = len(non_empty_cols)

        # Update column definitions
        table.columns = [
            col for col in table.columns
            if col.index in col_mapping
        ]
        for col in table.columns:
            col.index = col_mapping.get(col.index, col.index)

        return table
