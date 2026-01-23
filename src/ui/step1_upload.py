import os
import cv2
import numpy as np
from PySide2.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QComboBox, QSlider,
    QGroupBox, QCheckBox, QFileDialog, QSplitter,
    QScrollArea
)
from PySide2.QtCore import Qt, Signal, QThread
from PySide2.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent

from ..models import Project, PreprocessingProfile
from ..core import ImagePreprocessor, TableExtractor
from .widgets.image_preview import ImagePreview


class ProcessingThread(QThread):
    """Background thread for image processing and table extraction."""

    progress = Signal(int)
    finished = Signal(Project)
    error = Signal(str)

    def __init__(self, image_path: str, profile: PreprocessingProfile, roi=None):
        super().__init__()
        self.image_path = image_path
        self.profile = profile
        self.roi = roi  # Optional: list of 4 normalized points [(x,y), ...]

    def run(self):
        try:
            self.progress.emit(10)

            # Load image
            image = cv2.imread(self.image_path)
            if image is None:
                self.error.emit(f"Could not load image: {self.image_path}")
                return

            self.progress.emit(15)

            # Apply ROI crop if specified
            if self.roi and len(self.roi) == 4:
                image = self._apply_roi_crop(image)

            self.progress.emit(20)

            # Preprocess
            preprocessor = ImagePreprocessor(self.profile)
            processed = preprocessor.process(image)

            self.progress.emit(40)

            # Extract tables
            extractor = TableExtractor()
            tables = extractor.extract(processed)

            self.progress.emit(80)

            # Create project
            project = Project(
                name=os.path.splitext(os.path.basename(self.image_path))[0],
                image_path=self.image_path,
                profile=self.profile,
                tables=tables,
                roi=self.roi,
            )

            # Store image data
            with open(self.image_path, "rb") as f:
                project.image_data = f.read()

            # Store processed image
            _, buffer = cv2.imencode(".png", processed)
            project.processed_image_data = buffer.tobytes()

            self.progress.emit(100)
            self.finished.emit(project)

        except Exception as e:
            self.error.emit(str(e))

    def _apply_roi_crop(self, image: np.ndarray) -> np.ndarray:
        """Apply perspective transform to extract ROI from image."""
        h, w = image.shape[:2]

        # Convert normalized coords to pixel coords
        src_points = np.array([
            [int(pt[0] * w), int(pt[1] * h)]
            for pt in self.roi
        ], dtype=np.float32)

        # Order points: TL, TR, BR, BL
        # Calculate output dimensions
        width_top = np.linalg.norm(src_points[1] - src_points[0])
        width_bottom = np.linalg.norm(src_points[2] - src_points[3])
        height_left = np.linalg.norm(src_points[3] - src_points[0])
        height_right = np.linalg.norm(src_points[2] - src_points[1])

        out_w = int(max(width_top, width_bottom))
        out_h = int(max(height_left, height_right))

        # Destination points (rectangle)
        dst_points = np.array([
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1]
        ], dtype=np.float32)

        # Perspective transform
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        cropped = cv2.warpPerspective(image, matrix, (out_w, out_h))

        return cropped


class DropZone(QFrame):
    """Drag and drop zone for images."""

    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._setup_ui()

    def _setup_ui(self):
        self.setMinimumSize(300, 200)
        self.setStyleSheet("""
            DropZone {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f9f9f9;
            }
            DropZone:hover {
                border-color: #2196F3;
                background-color: #e3f2fd;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        icon_label = QLabel("📄")
        icon_label.setStyleSheet("font-size: 48px;")
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        text_label = QLabel("Drag & drop image here\nor click to browse")
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(text_label)

        self.format_label = QLabel("Supported: PNG, JPG, JPEG, BMP, TIFF")
        self.format_label.setAlignment(Qt.AlignCenter)
        self.format_label.setStyleSheet("color: #999; font-size: 11px;")
        layout.addWidget(self.format_label)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._browse_file()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self._is_valid_image(file_path):
                self.file_dropped.emit(file_path)

    def _browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        if file_path:
            self.file_dropped.emit(file_path)

    def _is_valid_image(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]


class Step1Upload(QWidget):
    """Step 1: Upload and preprocess image."""

    processing_started = Signal()
    processing_finished = Signal(Project)
    processing_progress = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.current_image_path = None
        self.processing_thread = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Upload and settings
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Drop zone
        self.drop_zone = DropZone()
        left_layout.addWidget(self.drop_zone)

        # File info
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: #666; padding: 10px;")
        left_layout.addWidget(self.file_label)

        # Profile selection
        profile_group = QGroupBox("Preprocessing Profile")
        profile_layout = QVBoxLayout(profile_group)

        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["Scan (Document)", "Photo"])
        profile_layout.addWidget(self.profile_combo)

        # Profile options
        self.chk_deskew = QCheckBox("Deskew (straighten)")
        self.chk_deskew.setChecked(True)
        profile_layout.addWidget(self.chk_deskew)

        self.chk_denoise = QCheckBox("Denoise")
        self.chk_denoise.setChecked(True)
        profile_layout.addWidget(self.chk_denoise)

        self.chk_contrast = QCheckBox("Enhance contrast")
        self.chk_contrast.setChecked(True)
        profile_layout.addWidget(self.chk_contrast)

        self.chk_lines = QCheckBox("Enhance table lines")
        self.chk_lines.setChecked(True)
        profile_layout.addWidget(self.chk_lines)

        # Photo-specific options
        self.chk_perspective = QCheckBox("Correct perspective")
        self.chk_perspective.setChecked(False)
        profile_layout.addWidget(self.chk_perspective)

        self.chk_shadows = QCheckBox("Remove shadows")
        self.chk_shadows.setChecked(False)
        profile_layout.addWidget(self.chk_shadows)

        left_layout.addWidget(profile_group)

        # ROI Selection group
        roi_group = QGroupBox("Area Selection")
        roi_layout = QVBoxLayout(roi_group)

        self.chk_use_roi = QCheckBox("Process selected area only")
        self.chk_use_roi.setChecked(False)
        self.chk_use_roi.setEnabled(False)
        roi_layout.addWidget(self.chk_use_roi)

        roi_btn_layout = QHBoxLayout()
        self.btn_select_area = QPushButton("Select Area")
        self.btn_select_area.setEnabled(False)
        self.btn_select_area.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
            QPushButton:checked {
                background-color: #1565c0;
            }
        """)
        self.btn_select_area.setCheckable(True)
        roi_btn_layout.addWidget(self.btn_select_area)

        self.btn_clear_area = QPushButton("Clear")
        self.btn_clear_area.setEnabled(False)
        self.btn_clear_area.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                color: #333;
                border: 1px solid #ddd;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #999;
            }
        """)
        roi_btn_layout.addWidget(self.btn_clear_area)
        roi_btn_layout.addStretch()

        roi_layout.addLayout(roi_btn_layout)

        self.roi_status_label = QLabel("")
        self.roi_status_label.setStyleSheet("color: #666; font-size: 11px;")
        roi_layout.addWidget(self.roi_status_label)

        left_layout.addWidget(roi_group)

        # Process button
        self.btn_process = QPushButton("Process Image")
        self.btn_process.setEnabled(False)
        self.btn_process.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #43A047;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        left_layout.addWidget(self.btn_process)

        # Results summary
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("padding: 10px; color: #333;")
        left_layout.addWidget(self.results_label)

        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # Right panel - Preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        preview_label = QLabel("Preview")
        preview_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(preview_label)

        # Preview toggle
        preview_toggle = QHBoxLayout()
        self.btn_original = QPushButton("Original")
        self.btn_original.setCheckable(True)
        self.btn_original.setChecked(True)
        self.btn_processed = QPushButton("Processed")
        self.btn_processed.setCheckable(True)
        preview_toggle.addWidget(self.btn_original)
        preview_toggle.addWidget(self.btn_processed)
        preview_toggle.addStretch()
        right_layout.addLayout(preview_toggle)

        # Image preview
        self.image_preview = ImagePreview()
        right_layout.addWidget(self.image_preview, 1)

        splitter.addWidget(right_panel)
        splitter.setSizes([350, 600])

    def _connect_signals(self):
        self.drop_zone.file_dropped.connect(self._on_file_dropped)
        self.btn_process.clicked.connect(self._process_image)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        self.btn_original.clicked.connect(lambda: self._show_preview("original"))
        self.btn_processed.clicked.connect(lambda: self._show_preview("processed"))

        # ROI selection signals
        self.btn_select_area.toggled.connect(self._on_select_area_toggled)
        self.btn_clear_area.clicked.connect(self._on_clear_area)
        self.image_preview.roi_changed.connect(self._on_roi_changed)

    def _on_file_dropped(self, file_path: str):
        self.current_image_path = file_path
        self.file_label.setText(f"File: {os.path.basename(file_path)}")
        self.btn_process.setEnabled(True)
        self.results_label.setText("")

        # Show preview
        self.image_preview.load_image(file_path)
        self.btn_original.setChecked(True)
        self.btn_processed.setChecked(False)

        # Enable ROI selection
        self.btn_select_area.setEnabled(True)
        self.btn_select_area.setChecked(False)
        self.btn_clear_area.setEnabled(False)
        self.chk_use_roi.setEnabled(False)
        self.chk_use_roi.setChecked(False)
        self.roi_status_label.setText("")
        self.image_preview.set_selection_mode(False)
        self.image_preview.clear_roi()

    def _on_profile_changed(self, index: int):
        is_photo = index == 1
        self.chk_perspective.setVisible(is_photo)
        self.chk_shadows.setVisible(is_photo)
        self.chk_perspective.setChecked(is_photo)
        self.chk_shadows.setChecked(is_photo)

    def _on_select_area_toggled(self, checked: bool):
        """Toggle area selection mode."""
        self.image_preview.set_selection_mode(checked)

        if checked:
            self.btn_select_area.setText("Done Selecting")
            self.roi_status_label.setText("Drag corners to adjust selection area")
        else:
            self.btn_select_area.setText("Select Area")
            if self.image_preview.has_roi():
                self.roi_status_label.setText("Area selected")
                self.btn_clear_area.setEnabled(True)
                self.chk_use_roi.setEnabled(True)
                self.chk_use_roi.setChecked(True)
            else:
                self.roi_status_label.setText("")

    def _on_clear_area(self):
        """Clear ROI selection."""
        self.image_preview.clear_roi()
        self.image_preview.set_selection_mode(False)
        self.btn_select_area.setChecked(False)
        self.btn_select_area.setText("Select Area")
        self.btn_clear_area.setEnabled(False)
        self.chk_use_roi.setEnabled(False)
        self.chk_use_roi.setChecked(False)
        self.roi_status_label.setText("")

    def _on_roi_changed(self):
        """Handle ROI change from image preview."""
        if self.image_preview.has_roi():
            self.btn_clear_area.setEnabled(True)
            self.chk_use_roi.setEnabled(True)
        else:
            self.btn_clear_area.setEnabled(False)
            self.chk_use_roi.setEnabled(False)
            self.chk_use_roi.setChecked(False)

    def _get_current_profile(self) -> PreprocessingProfile:
        is_photo = self.profile_combo.currentIndex() == 1
        return PreprocessingProfile(
            name="photo" if is_photo else "scan",
            deskew=self.chk_deskew.isChecked(),
            denoise=self.chk_denoise.isChecked(),
            contrast_enhance=self.chk_contrast.isChecked(),
            line_enhancement=self.chk_lines.isChecked(),
            perspective_correction=self.chk_perspective.isChecked() if is_photo else False,
            shadow_removal=self.chk_shadows.isChecked() if is_photo else False,
        )

    def _process_image(self):
        if not self.current_image_path:
            return

        profile = self._get_current_profile()

        # Get ROI if enabled
        roi = None
        if self.chk_use_roi.isChecked() and self.image_preview.has_roi():
            roi = self.image_preview.get_roi()

        self.btn_process.setEnabled(False)
        self.processing_started.emit()

        self.processing_thread = ProcessingThread(self.current_image_path, profile, roi)
        self.processing_thread.progress.connect(self._on_progress)
        self.processing_thread.finished.connect(self._on_finished)
        self.processing_thread.error.connect(self._on_error)
        self.processing_thread.start()

    def _on_progress(self, value: int):
        self.processing_progress.emit(value)

    def _on_finished(self, project: Project):
        self.project = project
        self.btn_process.setEnabled(True)

        # Update results
        num_tables = len(project.tables)
        if num_tables == 0:
            self.results_label.setText(
                "⚠️ No tables detected. Try adjusting preprocessing settings."
            )
        else:
            self.results_label.setText(
                f"✅ Found {num_tables} table(s). Click 'Next' to continue."
            )

        # Show processed preview
        if project.processed_image_data:
            self.image_preview.load_from_bytes(project.processed_image_data)
            self.btn_original.setChecked(False)
            self.btn_processed.setChecked(True)

        self.processing_finished.emit(project)

    def _on_error(self, error_msg: str):
        self.btn_process.setEnabled(True)
        self.results_label.setText(f"❌ Error: {error_msg}")
        self.processing_progress.emit(0)

    def _show_preview(self, mode: str):
        if mode == "original":
            self.btn_original.setChecked(True)
            self.btn_processed.setChecked(False)
            if self.current_image_path:
                self.image_preview.load_image(self.current_image_path)
        else:
            self.btn_original.setChecked(False)
            self.btn_processed.setChecked(True)
            if self.project and self.project.processed_image_data:
                self.image_preview.load_from_bytes(self.project.processed_image_data)

    def set_project(self, project: Project):
        """Set project data."""
        self.project = project
        if project.image_path:
            self.current_image_path = project.image_path
            self.file_label.setText(f"File: {os.path.basename(project.image_path)}")
            self.btn_process.setEnabled(True)

            if project.image_data:
                self.image_preview.load_from_bytes(project.image_data)

    def reset(self):
        """Reset to initial state."""
        self.project = None
        self.current_image_path = None
        self.file_label.setText("No file selected")
        self.btn_process.setEnabled(False)
        self.results_label.setText("")
        self.image_preview.clear()

        # Reset ROI state
        self.btn_select_area.setEnabled(False)
        self.btn_select_area.setChecked(False)
        self.btn_clear_area.setEnabled(False)
        self.chk_use_roi.setEnabled(False)
        self.chk_use_roi.setChecked(False)
        self.roi_status_label.setText("")
        self.image_preview.set_selection_mode(False)
