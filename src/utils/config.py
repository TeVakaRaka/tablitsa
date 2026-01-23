import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class DetectorConfig:
    """Configuration for table detection parameters."""

    # Line detection parameters
    angle_tolerance: float = 10.0       # ±degrees for horizontal/vertical line detection
    cluster_tolerance: int = 20         # pixels for coordinate clustering
    min_line_length: int = 50           # minimum line length in pixels
    hough_threshold: int = 80           # Hough transform threshold

    # Table validation
    min_table_area_ratio: float = 0.01  # minimum 1% of image area
    min_rows: int = 2
    min_cols: int = 2
    min_cell_area: int = 100            # minimum cell area in pixels

    # Grid regularity
    max_row_variance: float = 0.5       # maximum coefficient of variation for row heights
    max_col_variance: float = 0.5       # maximum coefficient of variation for column widths

    # CLAHE preprocessing
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8

    # Multiple thresholds
    use_multiple_thresholds: bool = True
    adaptive_block_sizes: tuple = (11, 15, 21)

    # Lineless detection
    lineless_min_blocks: int = 4        # minimum text blocks for lineless detection
    adjacency_distance: int = 50        # max distance for adjacency graph edges

    # YOLO integration
    use_yolo: bool = True
    yolo_model_path: str = "yolov8m.pt"
    yolo_confidence_threshold: float = 0.5
    yolo_fallback_threshold: float = 0.7  # use YOLO if OpenCV confidence < this


@dataclass
class AppConfig:
    """Application configuration."""

    # OCR settings
    ocr_languages: str = "rus+eng"
    tessdata_path: Optional[str] = None

    # Detection settings
    min_table_area: int = 5000
    min_table_rows: int = 2
    min_table_cols: int = 2
    detect_lineless_tables: bool = True

    # Export settings
    default_export_format: str = "xlsx"
    csv_delimiter: str = ";"
    csv_encoding: str = "utf-8-sig"
    include_headers_default: bool = True

    # UI settings
    window_width: int = 1000
    window_height: int = 700
    recent_files: list = field(default_factory=list)
    max_recent_files: int = 10

    # Paths
    last_open_dir: str = ""
    last_save_dir: str = ""

    @classmethod
    def get_config_path(cls) -> Path:
        """Get path to config file."""
        if os.name == "nt":  # Windows
            config_dir = Path(os.environ.get("APPDATA", "")) / "Tablitsa"
        else:  # macOS/Linux
            config_dir = Path.home() / ".config" / "tablitsa"

        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "config.json"

    @classmethod
    def get_tessdata_path(cls) -> Optional[str]:
        """Get path to bundled tessdata."""
        # Check relative to executable/script
        possible_paths = [
            Path(__file__).parent.parent.parent / "resources" / "tessdata",
            Path.cwd() / "resources" / "tessdata",
            Path(__file__).parent.parent.parent.parent / "resources" / "tessdata",
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        return None

    def save(self):
        """Save configuration to file."""
        config_path = self.get_config_path()
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls) -> "AppConfig":
        """Load configuration from file."""
        config_path = cls.get_config_path()

        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(**data)
            except (json.JSONDecodeError, TypeError):
                pass

        return cls()

    def add_recent_file(self, file_path: str):
        """Add file to recent files list."""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)

        self.recent_files.insert(0, file_path)

        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[:self.max_recent_files]

        self.save()


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = AppConfig.load()
    return _config


def save_config():
    """Save global config."""
    global _config
    if _config:
        _config.save()


# Global detector config instance
_detector_config: Optional[DetectorConfig] = None


def get_detector_config() -> DetectorConfig:
    """Get global detector config instance."""
    global _detector_config
    if _detector_config is None:
        _detector_config = DetectorConfig()
    return _detector_config


def set_detector_config(config: DetectorConfig):
    """Set global detector config."""
    global _detector_config
    _detector_config = config
