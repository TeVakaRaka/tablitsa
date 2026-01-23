from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


@dataclass
class OCRAlternative:
    """Alternative OCR reading for a cell."""
    text: str
    confidence: float
    source: str  # "tesseract", "easyocr", etc.

    def to_dict(self) -> dict:
        return {"text": self.text, "confidence": self.confidence, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict) -> "OCRAlternative":
        return cls(text=data["text"], confidence=data["confidence"], source=data["source"])


@dataclass
class Cell:
    """Represents a single cell in a table."""

    row: int
    col: int
    text: str = ""
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    is_merged: bool = False
    merge_span: Tuple[int, int] = (1, 1)  # rowspan, colspan
    original_text: str = ""  # Text before user edits
    alternatives: List[OCRAlternative] = field(default_factory=list)  # Alternative OCR results
    needs_review: bool = False  # Flag for cells needing manual review

    def __post_init__(self):
        if not self.original_text:
            self.original_text = self.text

    @property
    def is_edited(self) -> bool:
        """Check if the cell was edited by user."""
        return self.text != self.original_text

    @property
    def has_low_confidence(self) -> bool:
        """Check if OCR confidence is below threshold."""
        return self.confidence < 0.6

    def reset(self):
        """Reset cell text to original OCR result."""
        self.text = self.original_text

    @property
    def has_alternatives(self) -> bool:
        """Check if cell has alternative OCR readings."""
        return len(self.alternatives) > 1

    def get_alternatives_text(self) -> List[str]:
        """Get list of alternative texts."""
        return [alt.text for alt in self.alternatives if alt.text != self.text]

    def to_dict(self) -> dict:
        """Serialize cell to dictionary."""
        return {
            "row": self.row,
            "col": self.col,
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "is_merged": self.is_merged,
            "merge_span": self.merge_span,
            "original_text": self.original_text,
            "alternatives": [alt.to_dict() for alt in self.alternatives],
            "needs_review": self.needs_review,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Cell":
        """Deserialize cell from dictionary."""
        cell = cls(
            row=data["row"],
            col=data["col"],
            text=data.get("text", ""),
            confidence=data.get("confidence", 0.0),
            bbox=tuple(data["bbox"]) if data.get("bbox") else None,
            is_merged=data.get("is_merged", False),
            merge_span=tuple(data.get("merge_span", (1, 1))),
            original_text=data.get("original_text", ""),
            needs_review=data.get("needs_review", False),
        )
        cell.alternatives = [
            OCRAlternative.from_dict(alt) for alt in data.get("alternatives", [])
        ]
        return cell
