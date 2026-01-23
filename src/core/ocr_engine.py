import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


def find_bundled_tesseract() -> Tuple[Optional[str], Optional[str]]:
    """Find bundled Tesseract executable and tessdata path."""
    possible_roots = []

    # For PyInstaller frozen app
    if getattr(sys, 'frozen', False):
        # sys.executable is the path to the .exe file
        exe_dir = Path(sys.executable).parent
        possible_roots.append(exe_dir)
        # Also check _MEIPASS (PyInstaller temp folder)
        if hasattr(sys, '_MEIPASS'):
            possible_roots.append(Path(sys._MEIPASS))
    else:
        # Development mode
        possible_roots.append(Path(__file__).parent.parent.parent)  # src/core -> project root
        possible_roots.append(Path(__file__).parent.parent.parent.parent)

    # Always check current working directory
    possible_roots.append(Path.cwd())

    # Check each possible location
    for root in possible_roots:
        if root is None:
            continue

        tesseract_dir = root / "tesseract"
        tesseract_exe = tesseract_dir / "tesseract.exe"
        tessdata_dir = tesseract_dir / "tessdata"

        if tesseract_exe.exists():
            return str(tesseract_exe), str(tessdata_dir) if tessdata_dir.exists() else None

    return None, None


def setup_bundled_tesseract():
    """Configure pytesseract to use bundled Tesseract."""
    if not TESSERACT_AVAILABLE:
        return False

    tesseract_exe, tessdata_path = find_bundled_tesseract()

    if tesseract_exe:
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe
        print(f"Using bundled Tesseract: {tesseract_exe}")

        if tessdata_path:
            os.environ["TESSDATA_PREFIX"] = tessdata_path
            print(f"Using tessdata: {tessdata_path}")

        return True

    return False


# Try to setup bundled Tesseract on module load
_bundled_tesseract_setup = setup_bundled_tesseract()


@dataclass
class OCRResult:
    """Result of OCR on a region."""

    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h


class OCREngine:
    """Wrapper for Tesseract OCR."""

    def __init__(
        self,
        languages: str = "rus+eng",
        tessdata_path: Optional[str] = None,
    ):
        if not TESSERACT_AVAILABLE:
            raise ImportError("pytesseract is not installed")

        self.languages = languages
        self.tessdata_path = tessdata_path

        # Try bundled tesseract first, then provided path
        if not _bundled_tesseract_setup:
            if tessdata_path and os.path.exists(tessdata_path):
                os.environ["TESSDATA_PREFIX"] = tessdata_path

        # Configure Tesseract
        self.config = "--psm 6"  # Assume uniform block of text

    def set_languages(self, languages: str):
        """Set OCR languages (e.g., 'rus+eng')."""
        self.languages = languages

    def recognize_text(
        self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> OCRResult:
        """Recognize text in image or region."""
        if bbox:
            x, y, w, h = bbox
            roi = image[y:y+h, x:x+w]
        else:
            roi = image

        # Ensure image is valid
        if roi.size == 0:
            return OCRResult(text="", confidence=0.0, bbox=bbox)

        # Preprocess for better OCR
        processed = self._preprocess_for_ocr(roi)

        try:
            # Get OCR data with confidence
            data = pytesseract.image_to_data(
                processed,
                lang=self.languages,
                config=self.config,
                output_type=Output.DICT
            )

            # Extract text and average confidence
            texts = []
            confidences = []

            for i, conf in enumerate(data["conf"]):
                if conf > 0:  # Valid detection
                    text = data["text"][i].strip()
                    if text:
                        texts.append(text)
                        confidences.append(conf)

            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(
                text=combined_text,
                confidence=avg_confidence / 100.0,  # Normalize to 0-1
                bbox=bbox,
            )

        except Exception as e:
            print(f"OCR error: {e}")
            return OCRResult(text="", confidence=0.0, bbox=bbox)

    def recognize_table_cell(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> OCRResult:
        """Recognize text in a table cell with cell-specific preprocessing."""
        x, y, w, h = bbox

        # Add padding to avoid cutting text
        padding = 2
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        roi = image[y:y+h, x:x+w]

        if roi.size == 0:
            return OCRResult(text="", confidence=0.0, bbox=bbox)

        # Preprocess
        processed = self._preprocess_for_ocr(roi)

        # Use PSM 7 for single line
        cell_config = "--psm 7" if h < 50 else "--psm 6"

        try:
            data = pytesseract.image_to_data(
                processed,
                lang=self.languages,
                config=cell_config,
                output_type=Output.DICT
            )

            texts = []
            confidences = []

            for i, conf in enumerate(data["conf"]):
                if conf > 0:
                    text = data["text"][i].strip()
                    if text:
                        texts.append(text)
                        confidences.append(conf)

            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(
                text=combined_text,
                confidence=avg_confidence / 100.0,
                bbox=bbox,
            )

        except Exception:
            return OCRResult(text="", confidence=0.0, bbox=bbox)

    def get_text_blocks(self, image: np.ndarray) -> List[Dict]:
        """Get all text blocks with their bounding boxes."""
        processed = self._preprocess_for_ocr(image)

        try:
            data = pytesseract.image_to_data(
                processed,
                lang=self.languages,
                config="--psm 11",  # Sparse text mode
                output_type=Output.DICT
            )

            blocks = []
            n_boxes = len(data["text"])

            for i in range(n_boxes):
                conf = data["conf"][i]
                text = data["text"][i].strip()

                if conf > 30 and text:  # Confidence threshold
                    block = {
                        "text": text,
                        "confidence": conf / 100.0,
                        "bbox": (
                            data["left"][i],
                            data["top"][i],
                            data["width"][i],
                            data["height"][i],
                        ),
                        "level": data["level"][i],
                        "block_num": data["block_num"][i],
                        "line_num": data["line_num"][i],
                        "word_num": data["word_num"][i],
                    }
                    blocks.append(block)

            return blocks

        except Exception as e:
            print(f"Error getting text blocks: {e}")
            return []

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize if too small
        h, w = gray.shape[:2]
        if h < 30:
            scale = 30 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

        # Apply thresholding
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Invert if needed (text should be dark on light background)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        return binary

    @staticmethod
    def is_available() -> bool:
        """Check if Tesseract is available."""
        if not TESSERACT_AVAILABLE:
            return False

        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    @staticmethod
    def get_available_languages() -> List[str]:
        """Get list of available Tesseract languages."""
        if not TESSERACT_AVAILABLE:
            return []

        try:
            langs = pytesseract.get_languages()
            return [lang for lang in langs if lang != "osd"]
        except Exception:
            return []
