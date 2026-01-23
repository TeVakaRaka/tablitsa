from PySide2.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea,
    QSizePolicy
)
from PySide2.QtCore import Qt, QSize, Signal, QPoint, QPointF, QRectF
from PySide2.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QColor, QPolygonF, QPainterPath

from typing import List, Tuple, Optional


class ROIOverlay(QWidget):
    """Overlay widget for ROI selection on image."""

    roi_changed = Signal()

    CORNER_RADIUS = 8
    CORNER_HIT_RADIUS = 15

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)  # Enable transparency
        self.setMouseTracking(True)

        self._selection_mode = False
        self._corners: List[QPointF] = []  # 4 corners: TL, TR, BR, BL
        self._dragging_corner = -1
        self._image_rect = QRectF()

    def set_selection_mode(self, enabled: bool):
        """Enable or disable selection mode."""
        self._selection_mode = enabled
        if not enabled:
            self._corners = []
        self.update()

    def set_image_rect(self, rect: QRectF):
        """Set the image display rect for coordinate conversion."""
        self._image_rect = rect

        # Initialize corners to cover full image if not set
        if self._selection_mode and not self._corners and not rect.isEmpty():
            margin = 20
            self._corners = [
                QPointF(rect.left() + margin, rect.top() + margin),
                QPointF(rect.right() - margin, rect.top() + margin),
                QPointF(rect.right() - margin, rect.bottom() - margin),
                QPointF(rect.left() + margin, rect.bottom() - margin),
            ]

        self.update()

    def get_roi_normalized(self) -> Optional[List[Tuple[float, float]]]:
        """Get ROI corners as normalized coordinates (0-1)."""
        if not self._corners or len(self._corners) != 4:
            return None
        if self._image_rect.isEmpty():
            return None

        result = []
        for corner in self._corners:
            x = (corner.x() - self._image_rect.left()) / self._image_rect.width()
            y = (corner.y() - self._image_rect.top()) / self._image_rect.height()
            # Clamp to 0-1
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            result.append((x, y))
        return result

    def clear_roi(self):
        """Clear ROI selection."""
        self._corners = []
        self.update()
        self.roi_changed.emit()

    def has_roi(self) -> bool:
        """Check if ROI is defined."""
        return len(self._corners) == 4

    def _find_nearest_corner(self, pos: QPointF) -> int:
        """Find nearest corner to position, returns index or -1."""
        if not self._corners:
            return -1

        for i, corner in enumerate(self._corners):
            dist = (corner - pos).manhattanLength()
            if dist < self.CORNER_HIT_RADIUS:
                return i
        return -1

    def mousePressEvent(self, event):
        if not self._selection_mode:
            event.ignore()
            return

        pos = QPointF(event.pos())

        # Check if clicking on a corner
        corner_idx = self._find_nearest_corner(pos)
        if corner_idx >= 0:
            self._dragging_corner = corner_idx
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if not self._selection_mode:
            event.ignore()
            return

        pos = QPointF(event.pos())

        # Update cursor based on hover
        corner_idx = self._find_nearest_corner(pos)
        if corner_idx >= 0 or self._dragging_corner >= 0:
            self.setCursor(Qt.SizeAllCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        # Drag corner if active
        if self._dragging_corner >= 0:
            # Constrain to image rect
            new_pos = QPointF(
                max(self._image_rect.left(), min(self._image_rect.right(), pos.x())),
                max(self._image_rect.top(), min(self._image_rect.bottom(), pos.y()))
            )
            self._corners[self._dragging_corner] = new_pos
            self.update()
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        if self._dragging_corner >= 0:
            self._dragging_corner = -1
            self.roi_changed.emit()
            event.accept()
        else:
            event.ignore()

    def paintEvent(self, event):
        if not self._selection_mode or not self._corners or len(self._corners) != 4:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Create polygon from corners
        polygon = QPolygonF()
        for corner in self._corners:
            polygon.append(corner)

        full_rect = QRectF(self.rect())

        # Draw dimmed overlay OUTSIDE selection using 4 rectangles method
        # This is more reliable than path subtraction
        overlay_color = QColor(0, 0, 0, 120)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(overlay_color))

        # Get bounding box of selection polygon
        poly_rect = polygon.boundingRect()

        # Top rectangle (above selection)
        painter.drawRect(QRectF(
            full_rect.left(), full_rect.top(),
            full_rect.width(), poly_rect.top() - full_rect.top()
        ))

        # Bottom rectangle (below selection)
        painter.drawRect(QRectF(
            full_rect.left(), poly_rect.bottom(),
            full_rect.width(), full_rect.bottom() - poly_rect.bottom()
        ))

        # Left rectangle (left of selection, between top and bottom)
        painter.drawRect(QRectF(
            full_rect.left(), poly_rect.top(),
            poly_rect.left() - full_rect.left(), poly_rect.height()
        ))

        # Right rectangle (right of selection, between top and bottom)
        painter.drawRect(QRectF(
            poly_rect.right(), poly_rect.top(),
            full_rect.right() - poly_rect.right(), poly_rect.height()
        ))

        # Draw selection border
        border_pen = QPen(QColor("#2196f3"), 2, Qt.SolidLine)
        painter.setPen(border_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPolygon(polygon)

        # Draw corner handles
        for i, corner in enumerate(self._corners):
            # Outer circle
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.setBrush(QBrush(QColor("#2196f3")))
            painter.drawEllipse(
                corner,
                self.CORNER_RADIUS,
                self.CORNER_RADIUS
            )

            # Inner dot
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.drawEllipse(corner, 3, 3)

        # Draw edge midpoint markers (smaller)
        painter.setPen(QPen(QColor("#2196f3"), 1))
        painter.setBrush(QBrush(QColor("#ffffff")))
        for i in range(4):
            p1 = self._corners[i]
            p2 = self._corners[(i + 1) % 4]
            mid = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)
            painter.drawEllipse(mid, 4, 4)

        painter.end()


class ImagePreview(QScrollArea):
    """Widget for displaying image with zoom, pan, and ROI selection."""

    roi_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_level = 1.0
        self.original_pixmap = None
        self._selection_mode = False
        self._setup_ui()

    def _setup_ui(self):
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QScrollArea {
                background-color: #e0e0e0;
                border: 1px solid #ccc;
            }
        """)

        # Container widget
        self.container = QWidget()
        self.setWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)

        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label)

        # ROI overlay (positioned over image_label)
        self.roi_overlay = ROIOverlay(self.image_label)
        self.roi_overlay.roi_changed.connect(self._on_roi_changed)

        # Placeholder
        self.placeholder = QLabel("No image loaded")
        self.placeholder.setStyleSheet("color: #999; font-size: 14px;")
        self.placeholder.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.placeholder)

    def load_image(self, path: str):
        """Load image from file path."""
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.original_pixmap = pixmap
            self._update_display()
            self.placeholder.setVisible(False)
            self.image_label.setVisible(True)

    def load_from_bytes(self, data: bytes):
        """Load image from bytes."""
        image = QImage()
        if image.loadFromData(data):
            pixmap = QPixmap.fromImage(image)
            self.original_pixmap = pixmap
            self._update_display()
            self.placeholder.setVisible(False)
            self.image_label.setVisible(True)

    def load_from_qimage(self, image: QImage):
        """Load from QImage."""
        pixmap = QPixmap.fromImage(image)
        self.original_pixmap = pixmap
        self._update_display()
        self.placeholder.setVisible(False)
        self.image_label.setVisible(True)

    def clear(self):
        """Clear the preview."""
        self.original_pixmap = None
        self.image_label.clear()
        self.image_label.setVisible(False)
        self.placeholder.setVisible(True)

    def _update_display(self):
        """Update displayed image with current zoom."""
        if self.original_pixmap:
            # Get available size
            available = self.viewport().size()

            # Calculate scaled size maintaining aspect ratio
            scaled_size = self.original_pixmap.size()
            scaled_size.scale(
                int(available.width() * self.zoom_level * 0.95),
                int(available.height() * self.zoom_level * 0.95),
                Qt.KeepAspectRatio
            )

            scaled_pixmap = self.original_pixmap.scaled(
                scaled_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.image_label.setPixmap(scaled_pixmap)

            # Update ROI overlay size and position
            self._update_roi_overlay()

    def set_zoom(self, level: float):
        """Set zoom level (1.0 = fit to view)."""
        self.zoom_level = max(0.1, min(5.0, level))
        self._update_display()

    def zoom_in(self):
        """Zoom in by 25%."""
        self.set_zoom(self.zoom_level * 1.25)

    def zoom_out(self):
        """Zoom out by 25%."""
        self.set_zoom(self.zoom_level / 1.25)

    def fit_to_view(self):
        """Fit image to view."""
        self.set_zoom(1.0)

    def resizeEvent(self, event):
        """Handle resize."""
        super().resizeEvent(event)
        self._update_display()

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        if event.modifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    # ROI Selection Methods

    def set_selection_mode(self, enabled: bool):
        """Enable or disable ROI selection mode."""
        self._selection_mode = enabled
        self.roi_overlay.set_selection_mode(enabled)
        self._update_roi_overlay()

    def is_selection_mode(self) -> bool:
        """Check if selection mode is active."""
        return self._selection_mode

    def get_roi(self) -> Optional[List[Tuple[float, float]]]:
        """Get ROI as normalized coordinates (0-1) for each corner.

        Returns list of 4 tuples [(x, y), ...] in order: TL, TR, BR, BL
        or None if no ROI is set.
        """
        if not self._selection_mode:
            return None
        return self.roi_overlay.get_roi_normalized()

    def get_roi_pixels(self, image_width: int, image_height: int) -> Optional[List[Tuple[int, int]]]:
        """Get ROI as pixel coordinates for given image dimensions.

        Args:
            image_width: Original image width
            image_height: Original image height

        Returns:
            List of 4 tuples [(x, y), ...] in pixel coordinates, or None.
        """
        normalized = self.get_roi()
        if not normalized:
            return None

        return [
            (int(x * image_width), int(y * image_height))
            for x, y in normalized
        ]

    def clear_roi(self):
        """Clear the ROI selection."""
        self.roi_overlay.clear_roi()

    def has_roi(self) -> bool:
        """Check if ROI is defined."""
        return self._selection_mode and self.roi_overlay.has_roi()

    def _update_roi_overlay(self):
        """Update ROI overlay position and size."""
        if not self.original_pixmap:
            return

        # Get the image label's pixmap rect
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return

        # Calculate the rect where the image is displayed
        label_size = self.image_label.size()
        pixmap_size = pixmap.size()

        # Center the pixmap rect within label
        x = (label_size.width() - pixmap_size.width()) / 2
        y = (label_size.height() - pixmap_size.height()) / 2

        image_rect = QRectF(x, y, pixmap_size.width(), pixmap_size.height())

        # Resize overlay to match image_label
        self.roi_overlay.setGeometry(self.image_label.geometry())
        self.roi_overlay.set_image_rect(image_rect)
        self.roi_overlay.raise_()

    def _on_roi_changed(self):
        """Handle ROI change from overlay."""
        self.roi_changed.emit()
