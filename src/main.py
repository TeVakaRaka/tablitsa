#!/usr/bin/env python3
"""
Tablitsa - Table Extractor

A Windows application for extracting tabular data from scanned documents and photos.
"""

import sys
import os

# CRITICAL: Import PyTorch/PaddleOCR BEFORE PySide2/Qt to avoid DLL initialization errors on Windows
# This fixes: OSError: [WinError 1114] c10.dll initialization failed
try:
    import torch
except Exception:
    pass

try:
    import easyocr
except Exception:
    pass

try:
    from paddleocr import PaddleOCR
except Exception:
    pass

# Add src to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_dependencies():
    """Check if all required dependencies are available."""
    missing = []

    try:
        import PySide2
    except ImportError:
        missing.append("PySide2")

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        import pytesseract
    except ImportError:
        missing.append("pytesseract")

    try:
        import PIL
    except ImportError:
        missing.append("Pillow")

    try:
        import openpyxl
    except ImportError:
        missing.append("openpyxl")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install -r requirements.txt")
        return False

    return True


def check_tesseract():
    """Check if Tesseract is installed and accessible."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        print("Warning: Tesseract OCR not found or not configured.")
        print("Please install Tesseract and add it to PATH.")
        print("Download: https://github.com/UB-Mannheim/tesseract/wiki")
        return False


def main():
    """Main entry point."""
    if not check_dependencies():
        sys.exit(1)

    if not check_tesseract():
        # Continue with warning, OCR will fail but app can still run
        pass

    from src.app import run_app
    sys.exit(run_app())


if __name__ == "__main__":
    main()
