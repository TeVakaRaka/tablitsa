from PySide2.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QPushButton, QLabel, QFrame,
    QMessageBox, QFileDialog, QProgressBar, QShortcut
)
from PySide2.QtCore import Qt, Signal
from PySide2.QtGui import QFont, QKeySequence

from ..models import Project
from .step1_upload import Step1Upload
from .step2_edit import Step2Edit
from .step3_export import Step3Export


class StepIndicator(QFrame):
    """Visual step indicator for the wizard."""

    def __init__(self, steps: list, parent=None):
        super().__init__(parent)
        self.steps = steps
        self.current_step = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)

        self.step_labels = []
        self.step_circles = []

        for i, step_name in enumerate(self.steps):
            if i > 0:
                # Connector line
                line = QFrame()
                line.setFrameShape(QFrame.HLine)
                line.setFixedHeight(2)
                line.setStyleSheet("background-color: #ccc;")
                layout.addWidget(line)

            # Step container
            step_widget = QWidget()
            step_layout = QVBoxLayout(step_widget)
            step_layout.setAlignment(Qt.AlignCenter)
            step_layout.setSpacing(5)

            # Circle with number
            circle = QLabel(str(i + 1))
            circle.setFixedSize(30, 30)
            circle.setAlignment(Qt.AlignCenter)
            circle.setStyleSheet("""
                QLabel {
                    background-color: #ccc;
                    border-radius: 15px;
                    color: white;
                    font-weight: bold;
                }
            """)
            step_layout.addWidget(circle, alignment=Qt.AlignCenter)
            self.step_circles.append(circle)

            # Step name
            label = QLabel(step_name)
            label.setAlignment(Qt.AlignCenter)
            font = label.font()
            font.setPointSize(9)
            label.setFont(font)
            step_layout.addWidget(label)
            self.step_labels.append(label)

            layout.addWidget(step_widget)

        self._update_styles()

    def set_current_step(self, step: int):
        self.current_step = step
        self._update_styles()

    def _update_styles(self):
        for i, (circle, label) in enumerate(zip(self.step_circles, self.step_labels)):
            if i < self.current_step:
                # Completed
                circle.setStyleSheet("""
                    QLabel {
                        background-color: #4CAF50;
                        border-radius: 15px;
                        color: white;
                        font-weight: bold;
                    }
                """)
            elif i == self.current_step:
                # Current
                circle.setStyleSheet("""
                    QLabel {
                        background-color: #2196F3;
                        border-radius: 15px;
                        color: white;
                        font-weight: bold;
                    }
                """)
            else:
                # Future
                circle.setStyleSheet("""
                    QLabel {
                        background-color: #ccc;
                        border-radius: 15px;
                        color: white;
                        font-weight: bold;
                    }
                """)


class MainWindow(QMainWindow):
    """Main application window with step-based navigation."""

    def __init__(self):
        super().__init__()
        self.project = Project()
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        self.setWindowTitle("Tablitsa - Table Extractor")
        self.setMinimumSize(1000, 700)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Step indicator
        self.step_indicator = StepIndicator([
            "Upload",
            "Edit",
            "Export"
        ])
        main_layout.addWidget(self.step_indicator)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #ddd;")
        main_layout.addWidget(separator)

        # Stacked widget for steps
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, 1)

        # Create step widgets
        self.step1 = Step1Upload()
        self.step2 = Step2Edit()
        self.step3 = Step3Export()

        self.stack.addWidget(self.step1)
        self.stack.addWidget(self.step2)
        self.stack.addWidget(self.step3)

        # Bottom navigation
        nav_frame = QFrame()
        nav_frame.setStyleSheet("background-color: #f5f5f5;")
        nav_layout = QHBoxLayout(nav_frame)
        nav_layout.setContentsMargins(20, 10, 20, 10)

        self.btn_back = QPushButton("\u2190 Back")
        self.btn_back.setEnabled(False)
        self.btn_back.setFixedWidth(110)
        self.btn_back.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                color: #333;
                border: 1px solid #ccc;
                padding: 10px 16px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover:enabled {
                background-color: #d0d0d0;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #aaa;
                border-color: #ddd;
            }
        """)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(20)

        self.btn_next = QPushButton("Next \u2192")
        self.btn_next.setFixedWidth(110)
        self.btn_next.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 16px;
                border-radius: 4px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)

        # Shortcut hints
        back_hint = QLabel("Alt+\u2190")
        back_hint.setStyleSheet("color: #999; font-size: 10px;")
        next_hint = QLabel("Alt+\u2192")
        next_hint.setStyleSheet("color: #999; font-size: 10px;")

        nav_layout.addWidget(self.btn_back)
        nav_layout.addWidget(back_hint)
        nav_layout.addWidget(self.progress_bar, 1)
        nav_layout.addStretch()
        nav_layout.addWidget(next_hint)
        nav_layout.addWidget(self.btn_next)

        main_layout.addWidget(nav_frame)

        # Menu bar
        self._setup_menu()

    def _setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        new_action = file_menu.addAction("New Project")
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new_project)

        open_action = file_menu.addAction("Open Project...")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_project)

        save_action = file_menu.addAction("Save Project")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_project)

        save_as_action = file_menu.addAction("Save Project As...")
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._save_project_as)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self._show_about)

    def _connect_signals(self):
        self.btn_back.clicked.connect(self._go_back)
        self.btn_next.clicked.connect(self._go_next)

        # Keyboard shortcuts for navigation
        self.shortcut_back = QShortcut(QKeySequence("Alt+Left"), self)
        self.shortcut_back.activated.connect(self._go_back)

        self.shortcut_next = QShortcut(QKeySequence("Alt+Right"), self)
        self.shortcut_next.activated.connect(self._go_next)

        # Connect step signals
        self.step1.processing_started.connect(self._on_processing_started)
        self.step1.processing_finished.connect(self._on_processing_finished)
        self.step1.processing_progress.connect(self._on_processing_progress)

    def _go_back(self):
        current = self.stack.currentIndex()
        if current > 0:
            self.stack.setCurrentIndex(current - 1)
            self.step_indicator.set_current_step(current - 1)
            self._update_nav_buttons()

    def _go_next(self):
        current = self.stack.currentIndex()

        # Validate current step
        if current == 0:
            if not self._validate_step1():
                return
            # Pass data to step 2
            self.step2.set_project(self.project)
        elif current == 1:
            if not self._validate_step2():
                return
            # Pass data to step 3
            self.step3.set_project(self.project)

        if current < 2:
            self.stack.setCurrentIndex(current + 1)
            self.step_indicator.set_current_step(current + 1)
            self._update_nav_buttons()

    def _validate_step1(self) -> bool:
        """Validate step 1 (upload and processing)."""
        if not self.project.tables:
            QMessageBox.warning(
                self, "No Tables",
                "No tables were detected. Please upload an image with tables."
            )
            return False
        return True

    def _validate_step2(self) -> bool:
        """Validate step 2 (editing)."""
        enabled_tables = self.project.get_enabled_tables()
        if not enabled_tables:
            QMessageBox.warning(
                self, "No Tables Selected",
                "Please select at least one table to export."
            )
            return False
        return True

    def _update_nav_buttons(self):
        current = self.stack.currentIndex()
        self.btn_back.setEnabled(current > 0)

        if current == 2:
            self.btn_next.setText("Export \u2713")
            self.btn_next.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 10px 16px;
                    border-radius: 4px;
                    font-size: 13px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #43A047;
                }
                QPushButton:disabled {
                    background-color: #ccc;
                }
            """)
        else:
            self.btn_next.setText("Next \u2192")
            self.btn_next.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    padding: 10px 16px;
                    border-radius: 4px;
                    font-size: 13px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
                QPushButton:disabled {
                    background-color: #ccc;
                }
            """)

    def _on_processing_started(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_next.setEnabled(False)

    def _on_processing_finished(self, project: Project):
        self.progress_bar.setVisible(False)
        self.btn_next.setEnabled(True)
        self.project = project

    def _on_processing_progress(self, value: int):
        self.progress_bar.setValue(value)

    def _new_project(self):
        if self._confirm_discard():
            self.project = Project()
            self.stack.setCurrentIndex(0)
            self.step_indicator.set_current_step(0)
            self.step1.reset()
            self._update_nav_buttons()

    def _open_project(self):
        if not self._confirm_discard():
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project",
            "", "Tablitsa Project (*.tablitsa);;JSON (*.json)"
        )

        if file_path:
            try:
                self.project = Project.load(file_path)
                self.step1.set_project(self.project)
                if self.project.tables:
                    self.step2.set_project(self.project)
                    self.stack.setCurrentIndex(1)
                    self.step_indicator.set_current_step(1)
                self._update_nav_buttons()
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to open project: {str(e)}"
                )

    def _save_project(self):
        if self.project.file_path:
            try:
                self.project.save()
                self.statusBar().showMessage("Project saved", 3000)
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to save project: {str(e)}"
                )
        else:
            self._save_project_as()

    def _save_project_as(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As",
            f"{self.project.name}.tablitsa",
            "Tablitsa Project (*.tablitsa);;JSON (*.json)"
        )

        if file_path:
            try:
                self.project.save(file_path)
                self.statusBar().showMessage("Project saved", 3000)
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to save project: {str(e)}"
                )

    def _confirm_discard(self) -> bool:
        if self.project.tables:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "Do you want to discard current project?",
                QMessageBox.Yes | QMessageBox.No
            )
            return reply == QMessageBox.Yes
        return True

    def _show_about(self):
        QMessageBox.about(
            self, "About Tablitsa",
            "Tablitsa - Table Extractor\n\n"
            "Extract tables from scanned documents and photos.\n\n"
            "Version 2.0.0\n\n"
            "Features:\n"
            "- Triple OCR (PaddleOCR + EasyOCR + Tesseract)\n"
            "- Interactive ROI selection\n"
            "- Cell Inspector with OCR alternatives\n"
            "- Photo preprocessing enhancements"
        )

    def set_project(self, project: Project):
        """Set current project."""
        self.project = project
        self.step1.set_project(project)
