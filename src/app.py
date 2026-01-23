import sys
from PySide2.QtWidgets import QApplication
from PySide2.QtCore import Qt
from PySide2.QtGui import QFont, QPalette, QColor

from .ui import MainWindow
from .utils import get_config


def create_application() -> QApplication:
    """Create and configure the Qt application."""
    # High DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Tablitsa")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Tablitsa")

    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Apply style
    app.setStyle("Fusion")

    # Light palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(250, 250, 250))
    palette.setColor(QPalette.WindowText, QColor(33, 33, 33))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(33, 33, 33))
    palette.setColor(QPalette.Text, QColor(33, 33, 33))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(33, 33, 33))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(33, 150, 243))
    palette.setColor(QPalette.Highlight, QColor(33, 150, 243))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    app.setPalette(palette)

    return app


def run_app():
    """Run the application."""
    app = create_application()

    # Load config
    config = get_config()

    # Create main window
    window = MainWindow()
    window.resize(config.window_width, config.window_height)
    window.show()

    # Run event loop
    return app.exec_()
