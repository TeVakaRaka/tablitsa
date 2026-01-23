import json
import os
import base64
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

from .table import Table
from typing import Tuple


@dataclass
class PreprocessingProfile:
    """Preprocessing settings for an image."""

    name: str = "scan"  # "scan" or "photo"
    deskew: bool = True
    denoise: bool = True
    contrast_enhance: bool = True
    threshold_value: int = 127
    line_enhancement: bool = True
    perspective_correction: bool = False  # For photo mode
    shadow_removal: bool = False  # For photo mode

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "deskew": self.deskew,
            "denoise": self.denoise,
            "contrast_enhance": self.contrast_enhance,
            "threshold_value": self.threshold_value,
            "line_enhancement": self.line_enhancement,
            "perspective_correction": self.perspective_correction,
            "shadow_removal": self.shadow_removal,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PreprocessingProfile":
        return cls(**data)

    @classmethod
    def scan_profile(cls) -> "PreprocessingProfile":
        """Default profile for scanned documents."""
        return cls(
            name="scan",
            deskew=True,
            denoise=True,
            contrast_enhance=True,
            line_enhancement=True,
        )

    @classmethod
    def photo_profile(cls) -> "PreprocessingProfile":
        """Default profile for photos."""
        return cls(
            name="photo",
            deskew=True,
            denoise=True,
            contrast_enhance=True,
            line_enhancement=True,
            perspective_correction=True,
            shadow_removal=True,
        )


@dataclass
class Project:
    """Represents a project with image and extracted tables."""

    name: str = "Untitled"
    image_path: Optional[str] = None
    image_data: Optional[bytes] = None  # For embedded storage
    processed_image_data: Optional[bytes] = None
    profile: PreprocessingProfile = field(default_factory=PreprocessingProfile)
    tables: List[Table] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    file_path: Optional[str] = None  # Path to saved .tablitsa file
    # ROI: 4 corner points as normalized coords [(x,y), ...] - TL, TR, BR, BL
    roi: Optional[List[Tuple[float, float]]] = None

    def add_table(self, table: Table):
        """Add a table to the project."""
        self.tables.append(table)
        self.modified_at = datetime.now().isoformat()

    def remove_table(self, table_id: str):
        """Remove a table by ID."""
        self.tables = [t for t in self.tables if t.id != table_id]
        self.modified_at = datetime.now().isoformat()

    def get_table(self, table_id: str) -> Optional[Table]:
        """Get table by ID."""
        for table in self.tables:
            if table.id == table_id:
                return table
        return None

    def get_enabled_tables(self) -> List[Table]:
        """Get only enabled tables."""
        return [t for t in self.tables if t.enabled]

    def save(self, file_path: Optional[str] = None) -> str:
        """Save project to file."""
        if file_path:
            self.file_path = file_path
        if not self.file_path:
            raise ValueError("No file path specified")

        self.modified_at = datetime.now().isoformat()
        data = self.to_dict()

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return self.file_path

    @classmethod
    def load(cls, file_path: str) -> "Project":
        """Load project from file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        project = cls.from_dict(data)
        project.file_path = file_path
        return project

    def to_dict(self) -> dict:
        """Serialize project to dictionary."""
        data = {
            "name": self.name,
            "image_path": self.image_path,
            "profile": self.profile.to_dict(),
            "tables": [t.to_dict() for t in self.tables],
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }

        # Embed image data as base64
        if self.image_data:
            data["image_data"] = base64.b64encode(self.image_data).decode("utf-8")
        if self.processed_image_data:
            data["processed_image_data"] = base64.b64encode(
                self.processed_image_data
            ).decode("utf-8")

        # Save ROI if defined
        if self.roi:
            data["roi"] = self.roi

        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        """Deserialize project from dictionary."""
        project = cls(
            name=data.get("name", "Untitled"),
            image_path=data.get("image_path"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            modified_at=data.get("modified_at", datetime.now().isoformat()),
        )

        if data.get("profile"):
            project.profile = PreprocessingProfile.from_dict(data["profile"])

        if data.get("image_data"):
            project.image_data = base64.b64decode(data["image_data"])
        if data.get("processed_image_data"):
            project.processed_image_data = base64.b64decode(
                data["processed_image_data"]
            )

        project.tables = [Table.from_dict(t) for t in data.get("tables", [])]

        # Load ROI if available
        if data.get("roi"):
            project.roi = [tuple(pt) for pt in data["roi"]]

        return project
