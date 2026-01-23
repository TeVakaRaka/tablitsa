"""
YOLO-based table detection module.

Uses YOLOv8 for fast and accurate table region detection.
Falls back gracefully when YOLO is not available.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import ultralytics (YOLO)
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.info("ultralytics not installed. YOLO detection disabled.")
    YOLO = None


@dataclass
class TableRegion:
    """Represents a detected table region from YOLO."""

    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    class_name: str = "table"


class YOLOTableDetector:
    """
    YOLO-based table detector.

    Uses YOLOv8 to detect table regions in images.
    Supports both pre-trained models and custom fine-tuned models.
    """

    # Table class IDs for common YOLO models
    # For COCO: no table class, need custom model
    # For custom table detection models: typically class 0
    DEFAULT_TABLE_CLASSES = [0]

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Initialize YOLO table detector.

        Args:
            model_path: Path to YOLO model weights or model name
            confidence_threshold: Minimum confidence for detections
            device: Device to run on ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model: Optional[YOLO] = None
        self._loaded = False
        self._table_classes = self.DEFAULT_TABLE_CLASSES

    def is_available(self) -> bool:
        """Check if YOLO is available."""
        return YOLO_AVAILABLE

    def load_model(self) -> bool:
        """
        Load the YOLO model.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if not YOLO_AVAILABLE:
            logger.warning("YOLO not available. Install with: pip install ultralytics")
            return False

        if self._loaded:
            return True

        try:
            self.model = YOLO(self.model_path)
            if self.device:
                self.model.to(self.device)
            self._loaded = True
            logger.info(f"YOLO model loaded: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False

    def set_table_classes(self, classes: List[int]):
        """Set which class IDs correspond to tables."""
        self._table_classes = classes

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[TableRegion]:
        """
        Detect table regions in image.

        Args:
            image: Input image as numpy array (BGR or RGB)
            confidence_threshold: Override default confidence threshold

        Returns:
            List of detected table regions.
        """
        if not self._loaded:
            if not self.load_model():
                return []

        if self.model is None:
            return []

        conf_thresh = confidence_threshold or self.confidence_threshold

        try:
            # Run YOLO inference
            results = self.model.predict(
                image,
                conf=conf_thresh,
                verbose=False,
            )

            tables = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for i, box in enumerate(boxes):
                    # Get class ID
                    cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], 'item') else int(box.cls[0])

                    # Check if this is a table class
                    # For general object detection, we look for table-like objects
                    # For custom models, check configured classes
                    class_name = result.names.get(cls_id, "unknown")

                    # Accept if class name contains 'table' or is in our table classes
                    is_table = (
                        "table" in class_name.lower() or
                        cls_id in self._table_classes
                    )

                    if not is_table:
                        continue

                    # Get bounding box (xyxy format -> xywh)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = (
                        int(x1),
                        int(y1),
                        int(x2 - x1),
                        int(y2 - y1),
                    )

                    # Get confidence
                    conf = float(box.conf[0].item()) if hasattr(box.conf[0], 'item') else float(box.conf[0])

                    tables.append(TableRegion(
                        bbox=bbox,
                        confidence=conf,
                        class_name=class_name,
                    ))

            return tables

        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

    def detect_all_objects(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[TableRegion]:
        """
        Detect all objects (not just tables) - useful for debugging.

        Args:
            image: Input image as numpy array
            confidence_threshold: Override default confidence threshold

        Returns:
            List of all detected regions.
        """
        if not self._loaded:
            if not self.load_model():
                return []

        if self.model is None:
            return []

        conf_thresh = confidence_threshold or self.confidence_threshold

        try:
            results = self.model.predict(
                image,
                conf=conf_thresh,
                verbose=False,
            )

            regions = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], 'item') else int(box.cls[0])
                    class_name = result.names.get(cls_id, "unknown")

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    conf = float(box.conf[0].item()) if hasattr(box.conf[0], 'item') else float(box.conf[0])

                    regions.append(TableRegion(
                        bbox=bbox,
                        confidence=conf,
                        class_name=class_name,
                    ))

            return regions

        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

    def unload_model(self):
        """Unload model to free memory."""
        self.model = None
        self._loaded = False


# Convenience function for checking YOLO availability
def is_yolo_available() -> bool:
    """Check if YOLO is available for use."""
    return YOLO_AVAILABLE
