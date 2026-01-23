import os
from PySide2.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QRadioButton,
    QLineEdit, QFileDialog, QMessageBox,
    QProgressBar, QCheckBox, QButtonGroup
)
from PySide2.QtCore import Qt, Signal, QThread

from ..models import Project
from ..export import XLSXExporter, CSVExporter


class ExportThread(QThread):
    """Background thread for export."""

    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, project: Project, format_type: str, output_path: str, options: dict):
        super().__init__()
        self.project = project
        self.format_type = format_type
        self.output_path = output_path
        self.options = options

    def run(self):
        try:
            tables = self.project.get_enabled_tables()

            if self.format_type == "xlsx":
                exporter = XLSXExporter()
                self.progress.emit(50)
                exporter.export(tables, self.output_path, **self.options)
                self.progress.emit(100)
                self.finished.emit(self.output_path)

            elif self.format_type == "csv":
                exporter = CSVExporter()
                self.progress.emit(30)

                # Export each table to separate file
                output_dir = os.path.dirname(self.output_path)
                base_name = os.path.splitext(os.path.basename(self.output_path))[0]

                exported_files = []
                for i, table in enumerate(tables):
                    file_path = os.path.join(
                        output_dir,
                        f"{base_name}_{table.name}.csv"
                    )
                    exporter.export([table], file_path, **self.options)
                    exported_files.append(file_path)
                    self.progress.emit(30 + int(70 * (i + 1) / len(tables)))

                self.finished.emit(output_dir)

        except Exception as e:
            self.error.emit(str(e))


class Step3Export(QWidget):
    """Step 3: Export tables."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.export_thread = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Summary
        self.summary_label = QLabel("Ready to export")
        self.summary_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.summary_label)

        self.tables_info = QLabel("")
        self.tables_info.setStyleSheet("color: #666;")
        layout.addWidget(self.tables_info)

        # Format selection
        format_group = QGroupBox("Export Format")
        format_layout = QVBoxLayout(format_group)

        self.format_button_group = QButtonGroup(self)

        self.radio_xlsx = QRadioButton("Excel (XLSX)")
        self.radio_xlsx.setChecked(True)
        self.format_button_group.addButton(self.radio_xlsx)
        format_layout.addWidget(self.radio_xlsx)

        xlsx_desc = QLabel("  All tables in one file, each table on a separate sheet")
        xlsx_desc.setStyleSheet("color: #666; font-size: 11px; margin-left: 20px;")
        format_layout.addWidget(xlsx_desc)

        self.radio_csv = QRadioButton("CSV (UTF-8)")
        self.format_button_group.addButton(self.radio_csv)
        format_layout.addWidget(self.radio_csv)

        csv_desc = QLabel("  Separate file for each table, semicolon delimiter")
        csv_desc.setStyleSheet("color: #666; font-size: 11px; margin-left: 20px;")
        format_layout.addWidget(csv_desc)

        layout.addWidget(format_group)

        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)

        self.chk_include_headers = QCheckBox("Include column headers")
        self.chk_include_headers.setChecked(True)
        options_layout.addWidget(self.chk_include_headers)

        self.chk_skip_empty = QCheckBox("Skip empty rows")
        self.chk_skip_empty.setChecked(False)
        options_layout.addWidget(self.chk_skip_empty)

        layout.addWidget(options_group)

        # Output path
        path_group = QGroupBox("Output Location")
        path_layout = QHBoxLayout(path_group)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select output file or folder...")
        path_layout.addWidget(self.path_edit)

        self.btn_browse = QPushButton("Browse...")
        path_layout.addWidget(self.btn_browse)

        layout.addWidget(path_group)

        # Export button
        export_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        export_layout.addWidget(self.progress_bar, 1)

        self.btn_export = QPushButton("Export")
        self.btn_export.setFixedWidth(120)
        self.btn_export.setStyleSheet("""
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
        export_layout.addWidget(self.btn_export)

        layout.addLayout(export_layout)

        # Status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def _connect_signals(self):
        self.btn_browse.clicked.connect(self._browse_output)
        self.btn_export.clicked.connect(self._export)
        self.radio_xlsx.toggled.connect(self._on_format_changed)
        self.radio_csv.toggled.connect(self._on_format_changed)

    def set_project(self, project: Project):
        """Set project data."""
        self.project = project

        # Update summary
        enabled_tables = project.get_enabled_tables()
        self.tables_info.setText(
            f"{len(enabled_tables)} table(s) selected for export"
        )

        # Set default output path
        if project.image_path:
            base_name = os.path.splitext(project.image_path)[0]
            self.path_edit.setText(f"{base_name}.xlsx")

    def _on_format_changed(self):
        """Handle format change."""
        if self.path_edit.text():
            path = self.path_edit.text()
            base = os.path.splitext(path)[0]
            if self.radio_xlsx.isChecked():
                self.path_edit.setText(f"{base}.xlsx")
            else:
                self.path_edit.setText(f"{base}.csv")

    def _browse_output(self):
        """Browse for output file."""
        if self.radio_xlsx.isChecked():
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Excel File",
                self.path_edit.text() or "",
                "Excel Files (*.xlsx)"
            )
        else:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV File",
                self.path_edit.text() or "",
                "CSV Files (*.csv)"
            )

        if file_path:
            self.path_edit.setText(file_path)

    def _export(self):
        """Start export."""
        if not self.project:
            return

        output_path = self.path_edit.text().strip()
        if not output_path:
            QMessageBox.warning(self, "Error", "Please select output location")
            return

        enabled_tables = self.project.get_enabled_tables()
        if not enabled_tables:
            QMessageBox.warning(self, "Error", "No tables selected for export")
            return

        format_type = "xlsx" if self.radio_xlsx.isChecked() else "csv"
        options = {
            "include_headers": self.chk_include_headers.isChecked(),
            "skip_empty": self.chk_skip_empty.isChecked(),
        }

        self.btn_export.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Exporting...")

        self.export_thread = ExportThread(self.project, format_type, output_path, options)
        self.export_thread.progress.connect(self._on_progress)
        self.export_thread.finished.connect(self._on_finished)
        self.export_thread.error.connect(self._on_error)
        self.export_thread.start()

    def _on_progress(self, value: int):
        self.progress_bar.setValue(value)

    def _on_finished(self, path: str):
        self.btn_export.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"✅ Export complete: {path}")

        reply = QMessageBox.question(
            self, "Export Complete",
            "Export successful! Open the exported file?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            import subprocess
            import sys
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])

    def _on_error(self, error_msg: str):
        self.btn_export.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"❌ Error: {error_msg}")
        QMessageBox.critical(self, "Export Error", f"Failed to export: {error_msg}")
