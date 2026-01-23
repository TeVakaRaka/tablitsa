import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..utils.config import DetectorConfig, get_detector_config


@dataclass
class DetectedTable:
    """Represents a detected table region."""

    bbox: Tuple[int, int, int, int]  # x, y, width, height
    grid: Optional[np.ndarray] = None  # Grid structure
    rows: int = 0
    cols: int = 0
    cell_bboxes: List[Tuple[int, int, int, int]] = None  # List of cell bounding boxes
    confidence: float = 0.0
    has_lines: bool = True
    detection_method: str = "unknown"  # Track which method detected this table

    def __post_init__(self):
        if self.cell_bboxes is None:
            self.cell_bboxes = []


class TableDetector:
    """Detects tables in images using line detection and text clustering."""

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        min_area: int = 5000,
        min_rows: int = 2,
        min_cols: int = 2,
        line_threshold: int = 100,
    ):
        # Use provided config or get global config
        self.config = config or get_detector_config()

        # For backward compatibility, still accept direct parameters
        self.min_area = min_area
        self.min_rows = self.config.min_rows if config else min_rows
        self.min_cols = self.config.min_cols if config else min_cols
        self.line_threshold = self.config.hough_threshold if config else line_threshold

        # Adaptive parameters (set per image)
        self._kernel_h = 40
        self._kernel_v = 40
        self._image_area = 0

    def detect(self, image: np.ndarray) -> List[DetectedTable]:
        """Detect all tables in image."""
        # Calculate adaptive parameters based on image size
        self._calculate_adaptive_params(image)

        # Preprocess image for better detection
        preprocessed = self._preprocess_for_detection(image)

        # Combine results from multiple detection methods
        tables = self._combine_detection_methods(image, preprocessed)

        # Validate all detected tables
        tables = [t for t in tables if self._validate_table(t)]

        return tables

    def _preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image using CLAHE for better line detection."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.config.use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=(self.config.clahe_tile_size, self.config.clahe_tile_size)
            )
            gray = clahe.apply(gray)

        return gray

    def _validate_table(self, table: DetectedTable) -> bool:
        """Validate detected table structure."""
        # Check minimum dimensions
        if table.rows < self.min_rows or table.cols < self.min_cols:
            return False

        # Check minimum area
        _, _, w, h = table.bbox
        if w * h < self.min_area:
            return False

        # Check minimum area ratio (must be at least 1% of image)
        if self._image_area > 0:
            area_ratio = (w * h) / self._image_area
            if area_ratio < self.config.min_table_area_ratio:
                return False

        # Validate cell bboxes
        if table.cell_bboxes:
            valid_cells = 0
            for cx, cy, cw, ch in table.cell_bboxes:
                if cw > 0 and ch > 0 and cw * ch >= self.config.min_cell_area:
                    valid_cells += 1

            # At least 50% of cells should be valid
            expected_cells = table.rows * table.cols
            if expected_cells > 0 and valid_cells / expected_cells < 0.5:
                return False

        return True

    def _calculate_adaptive_params(self, image: np.ndarray):
        """Calculate adaptive parameters based on image size."""
        h, w = image.shape[:2]
        self._image_area = h * w

        # Minimum area as percentage of image
        self.min_area = int(self._image_area * self.config.min_table_area_ratio)

        # Kernel sizes for line detection (scale with image)
        self._kernel_h = max(w // 30, 10)
        self._kernel_v = max(h // 30, 10)

        # Adaptive line detection parameters
        self._min_line_length = max(self.config.min_line_length, min(w, h) // 20)

    def _combine_detection_methods(
        self, image: np.ndarray, preprocessed: np.ndarray
    ) -> List[DetectedTable]:
        """Combine results from different detection methods.

        Tables detected by multiple methods get higher confidence.
        """
        all_tables = []

        # Method 1: Line-based detection with multiple thresholds (primary)
        if self.config.use_multiple_thresholds:
            lined_tables = self._detect_with_multiple_thresholds(image, preprocessed)
        else:
            lined_tables = self._detect_lined_tables(image, preprocessed)

        for t in lined_tables:
            t.confidence = max(t.confidence, 0.8)
            t.detection_method = "morphology"
        all_tables.extend(lined_tables)

        # Method 2: Intersection-based detection (Hough transform)
        intersection_tables = self._detect_by_intersections(preprocessed)

        # Check for overlapping detections and boost confidence
        for int_table in intersection_tables:
            int_table.detection_method = "hough"
            overlap_found = False
            for lined_table in lined_tables:
                if self._tables_overlap(int_table, lined_table):
                    # Boost confidence of lined table (confirmed by intersection method)
                    lined_table.confidence = min(lined_table.confidence + 0.15, 1.0)
                    overlap_found = True
                    break

            # If no overlap, add as separate detection with lower confidence
            if not overlap_found:
                int_table.confidence = 0.6
                all_tables.append(int_table)

        # Remove duplicates (keep higher confidence)
        return self._remove_duplicate_tables(all_tables)

    def _detect_with_multiple_thresholds(
        self, image: np.ndarray, preprocessed: np.ndarray
    ) -> List[DetectedTable]:
        """Detect tables using multiple threshold methods for robustness."""
        all_results = []

        # Method 1: Otsu threshold
        _, binary_otsu = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        tables_otsu = self._detect_from_binary(image, binary_otsu)
        all_results.extend(tables_otsu)

        # Method 2: Adaptive threshold with different block sizes
        for block_size in self.config.adaptive_block_sizes:
            binary_adaptive = cv2.adaptiveThreshold(
                preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block_size, 5
            )
            tables_adaptive = self._detect_from_binary(image, binary_adaptive)
            all_results.extend(tables_adaptive)

        # Method 3: Combined (OR of Otsu and adaptive)
        binary_combined = cv2.bitwise_or(binary_otsu, cv2.adaptiveThreshold(
            preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        ))
        tables_combined = self._detect_from_binary(image, binary_combined)
        all_results.extend(tables_combined)

        # Merge overlapping results and keep best
        return self._merge_overlapping_tables(all_results)

    def _merge_overlapping_tables(
        self, tables: List[DetectedTable]
    ) -> List[DetectedTable]:
        """Merge overlapping table detections, keeping the best ones."""
        if not tables:
            return []

        # Sort by confidence and grid completeness (rows * cols)
        sorted_tables = sorted(
            tables,
            key=lambda t: (t.confidence, t.rows * t.cols),
            reverse=True
        )

        merged = []
        used = set()

        for i, table in enumerate(sorted_tables):
            if i in used:
                continue

            # Find all overlapping tables
            overlapping = [table]
            for j, other in enumerate(sorted_tables[i+1:], start=i+1):
                if j not in used and self._tables_overlap(table, other, threshold=0.3):
                    overlapping.append(other)
                    used.add(j)

            # Keep the one with highest score (already first due to sorting)
            # But boost confidence if multiple methods detected it
            if len(overlapping) > 1:
                table.confidence = min(table.confidence + 0.1 * (len(overlapping) - 1), 1.0)

            merged.append(table)
            used.add(i)

        return merged

    def _tables_overlap(self, t1: DetectedTable, t2: DetectedTable, threshold: float = 0.5) -> bool:
        """Check if two tables overlap significantly."""
        x1, y1, w1, h1 = t1.bbox
        x2, y2, w2, h2 = t2.bbox

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return False

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        iou = intersection / union if union > 0 else 0
        return iou > threshold

    def _remove_duplicate_tables(self, tables: List[DetectedTable]) -> List[DetectedTable]:
        """Remove duplicate table detections, keeping higher confidence ones."""
        if not tables:
            return []

        # Sort by confidence (highest first)
        sorted_tables = sorted(tables, key=lambda t: t.confidence, reverse=True)

        result = []
        for table in sorted_tables:
            is_duplicate = False
            for existing in result:
                if self._tables_overlap(table, existing, threshold=0.3):
                    is_duplicate = True
                    break
            if not is_duplicate:
                result.append(table)

        return result

    def _detect_by_intersections(self, preprocessed: np.ndarray) -> List[DetectedTable]:
        """Detect tables by finding line intersections using Hough transform."""
        # Edge detection
        edges = cv2.Canny(preprocessed, 50, 150, apertureSize=3)

        # Detect lines using Hough transform with config parameters
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=self.config.hough_threshold,
            minLineLength=self._min_line_length,
            maxLineGap=10
        )

        if lines is None or len(lines) < 4:
            return []

        # Separate horizontal and vertical lines based on angle tolerance
        horizontal = []
        vertical = []
        angle_tol = self.config.angle_tolerance

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < angle_tol or angle > (180 - angle_tol):  # Horizontal
                horizontal.append((min(y1, y2), x1, x2))
            elif (90 - angle_tol) < angle < (90 + angle_tol):  # Vertical
                vertical.append((min(x1, x2), y1, y2))

        if len(horizontal) < 2 or len(vertical) < 2:
            return []

        # Cluster lines to get unique positions using config tolerance
        h_positions = self._cluster_coordinates(
            [h[0] for h in horizontal],
            tolerance=self.config.cluster_tolerance
        )
        v_positions = self._cluster_coordinates(
            [v[0] for v in vertical],
            tolerance=self.config.cluster_tolerance
        )

        if len(h_positions) < 2 or len(v_positions) < 2:
            return []

        # Build grid from line positions
        rows = len(h_positions) - 1
        cols = len(v_positions) - 1

        if rows < self.min_rows or cols < self.min_cols:
            return []

        # Build cell bboxes
        cells = []
        for i in range(rows):
            for j in range(cols):
                cell_x = v_positions[j]
                cell_y = h_positions[i]
                cell_w = v_positions[j + 1] - v_positions[j]
                cell_h = h_positions[i + 1] - h_positions[i]
                cells.append((cell_x, cell_y, cell_w, cell_h))

        # Calculate bounding box
        bbox = (
            v_positions[0],
            h_positions[0],
            v_positions[-1] - v_positions[0],
            h_positions[-1] - h_positions[0]
        )

        table = DetectedTable(
            bbox=bbox,
            rows=rows,
            cols=cols,
            cell_bboxes=cells,
            confidence=0.7,
            has_lines=True,
        )

        return [table]

    def detect_lineless(
        self, image: np.ndarray, text_blocks: List[dict]
    ) -> List[DetectedTable]:
        """Detect tables without lines using text block clustering."""
        if not text_blocks:
            return []

        # Filter valid text blocks
        valid_blocks = [
            b for b in text_blocks
            if b.get("bbox") and len(b["bbox"]) == 4
        ]

        if len(valid_blocks) < 4:  # Need at least 2x2 for a table
            return []

        # Extract coordinates
        x_coords = []
        y_coords = []
        for block in valid_blocks:
            x, y, w, h = block["bbox"]
            x_coords.append(x + w // 2)  # Center x
            y_coords.append(y + h // 2)  # Center y

        # Cluster by x (columns) and y (rows)
        col_clusters = self._cluster_coordinates(x_coords)
        row_clusters = self._cluster_coordinates(y_coords)

        if len(col_clusters) < self.min_cols or len(row_clusters) < self.min_rows:
            return []

        # Check alignment regularity
        if not self._check_alignment_regularity(col_clusters, row_clusters):
            return []

        # Build table from clusters
        table = self._build_table_from_clusters(
            valid_blocks, col_clusters, row_clusters
        )

        if table:
            return [table]

        return []

    def _detect_lined_tables(
        self, image: np.ndarray, preprocessed: Optional[np.ndarray] = None
    ) -> List[DetectedTable]:
        """Detect tables with visible lines."""
        if preprocessed is None:
            preprocessed = self._preprocess_for_detection(image)

        # Denoise for better line detection
        denoised = cv2.fastNlMeansDenoising(preprocessed, None, h=10)

        # Binarize with adaptive threshold
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        )

        # Also try Otsu for comparison
        _, binary_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Combine both binarization results
        binary = cv2.bitwise_or(binary, binary_otsu)

        return self._detect_from_binary(image, binary)

    def _detect_from_binary(
        self, image: np.ndarray, binary: np.ndarray
    ) -> List[DetectedTable]:
        """Detect tables from a binary image using morphological operations."""
        # Use adaptive kernel sizes (already calculated)
        h_kernel_size = self._kernel_h
        v_kernel_size = self._kernel_v

        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        # Dilate to connect broken lines
        horizontal_lines = cv2.dilate(
            horizontal_lines,
            cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size // 2, 3))
        )

        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        # Dilate to connect broken lines
        vertical_lines = cv2.dilate(
            vertical_lines,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, v_kernel_size // 2))
        )

        # Combine lines
        table_mask = cv2.add(horizontal_lines, vertical_lines)

        # Find table contours
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area < self.min_area:
                continue

            # Extract table region
            table_region = table_mask[y:y+h, x:x+w]

            # Detect grid structure
            grid_info = self._detect_grid(table_region, horizontal_lines[y:y+h, x:x+w],
                                          vertical_lines[y:y+h, x:x+w])

            if grid_info and grid_info["rows"] >= self.min_rows and grid_info["cols"] >= self.min_cols:
                # Adjust cell bboxes to global coordinates
                cell_bboxes = [
                    (cx + x, cy + y, cw, ch)
                    for cx, cy, cw, ch in grid_info["cells"]
                ]

                table = DetectedTable(
                    bbox=(x, y, w, h),
                    rows=grid_info["rows"],
                    cols=grid_info["cols"],
                    cell_bboxes=cell_bboxes,
                    confidence=grid_info.get("score", 0.8),
                    has_lines=True,
                )
                tables.append(table)

        return tables

    def _detect_grid(
        self,
        table_mask: np.ndarray,
        h_lines: np.ndarray,
        v_lines: np.ndarray
    ) -> Optional[dict]:
        """Detect grid structure from line masks."""
        h, w = table_mask.shape[:2]

        # Find horizontal line positions
        h_profile = np.sum(h_lines, axis=1)
        h_peaks = self._find_line_positions(h_profile, threshold=w * 0.3)

        # Find vertical line positions
        v_profile = np.sum(v_lines, axis=0)
        v_peaks = self._find_line_positions(v_profile, threshold=h * 0.3)

        if len(h_peaks) < 2 or len(v_peaks) < 2:
            return None

        rows = len(h_peaks) - 1
        cols = len(v_peaks) - 1

        # Build cell bounding boxes
        cells = []
        for i in range(rows):
            for j in range(cols):
                cell_x = v_peaks[j]
                cell_y = h_peaks[i]
                cell_w = v_peaks[j + 1] - v_peaks[j]
                cell_h = h_peaks[i + 1] - h_peaks[i]
                cells.append((cell_x, cell_y, cell_w, cell_h))

        # Calculate grid score based on regularity
        score = self._calculate_grid_score(h_peaks, v_peaks)

        return {
            "rows": rows,
            "cols": cols,
            "cells": cells,
            "score": score,
        }

    def _find_line_positions(
        self, profile: np.ndarray, threshold: float
    ) -> List[int]:
        """Find line positions from projection profile."""
        # Normalize profile
        if profile.max() == 0:
            return []

        normalized = profile / profile.max()

        # Smooth the profile to reduce noise
        kernel_size = max(3, len(normalized) // 100)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = np.convolve(normalized, np.ones(kernel_size) / kernel_size, mode='same')

        # Find peaks using lower threshold for better detection
        peak_threshold = 0.15  # Lower threshold to catch faint lines
        peaks = []
        in_peak = False
        peak_start = 0
        peak_max_val = 0
        peak_max_pos = 0

        for i, val in enumerate(smoothed):
            if val > peak_threshold and not in_peak:
                in_peak = True
                peak_start = i
                peak_max_val = val
                peak_max_pos = i
            elif val > peak_threshold and in_peak:
                if val > peak_max_val:
                    peak_max_val = val
                    peak_max_pos = i
            elif val <= peak_threshold and in_peak:
                in_peak = False
                # Use the position of maximum value within the peak
                peaks.append(peak_max_pos)

        # Add last peak if still in one
        if in_peak:
            peaks.append(peak_max_pos)

        # Add boundaries if not detected
        if peaks and peaks[0] > 10:
            peaks.insert(0, 0)
        if peaks and peaks[-1] < len(profile) - 10:
            peaks.append(len(profile) - 1)

        # Remove duplicate or very close peaks
        if len(peaks) > 1:
            min_gap = len(profile) // 50  # Minimum 2% of dimension between lines
            filtered_peaks = [peaks[0]]
            for p in peaks[1:]:
                if p - filtered_peaks[-1] > min_gap:
                    filtered_peaks.append(p)
            peaks = filtered_peaks

        return peaks

    def _calculate_grid_score(
        self, h_peaks: List[int], v_peaks: List[int]
    ) -> float:
        """Calculate regularity score for grid."""
        if len(h_peaks) < 2 or len(v_peaks) < 2:
            return 0.0

        # Calculate row height variance
        row_heights = [h_peaks[i+1] - h_peaks[i] for i in range(len(h_peaks)-1)]
        row_variance = np.std(row_heights) / (np.mean(row_heights) + 1e-6)

        # Calculate column width variance
        col_widths = [v_peaks[i+1] - v_peaks[i] for i in range(len(v_peaks)-1)]
        col_variance = np.std(col_widths) / (np.mean(col_widths) + 1e-6)

        # Lower variance = higher score
        score = 1.0 - min((row_variance + col_variance) / 2, 1.0)

        return score

    def _cluster_coordinates(
        self, coords: List[int], tolerance: int = 20
    ) -> List[int]:
        """Cluster coordinates to find column/row positions."""
        if not coords:
            return []

        sorted_coords = sorted(coords)
        clusters = [[sorted_coords[0]]]

        for coord in sorted_coords[1:]:
            if coord - clusters[-1][-1] <= tolerance:
                clusters[-1].append(coord)
            else:
                clusters.append([coord])

        # Return cluster centers
        return [int(np.mean(cluster)) for cluster in clusters]

    def _check_alignment_regularity(
        self, col_clusters: List[int], row_clusters: List[int]
    ) -> bool:
        """Check if text blocks form a regular grid pattern."""
        if len(col_clusters) < 2 or len(row_clusters) < 2:
            return False

        # Check column spacing regularity
        col_gaps = [col_clusters[i+1] - col_clusters[i]
                    for i in range(len(col_clusters)-1)]
        if col_gaps:
            col_variance = np.std(col_gaps) / (np.mean(col_gaps) + 1e-6)
            if col_variance > 0.5:  # Allow some variance
                return False

        # Check row spacing regularity
        row_gaps = [row_clusters[i+1] - row_clusters[i]
                    for i in range(len(row_clusters)-1)]
        if row_gaps:
            row_variance = np.std(row_gaps) / (np.mean(row_gaps) + 1e-6)
            if row_variance > 0.5:
                return False

        return True

    def _build_table_from_clusters(
        self,
        text_blocks: List[dict],
        col_clusters: List[int],
        row_clusters: List[int],
    ) -> Optional[DetectedTable]:
        """Build table structure from text block clusters."""
        rows = len(row_clusters)
        cols = len(col_clusters)

        # Calculate bounding box
        all_x = []
        all_y = []
        for block in text_blocks:
            x, y, w, h = block["bbox"]
            all_x.extend([x, x + w])
            all_y.extend([y, y + h])

        if not all_x or not all_y:
            return None

        bbox = (min(all_x), min(all_y),
                max(all_x) - min(all_x),
                max(all_y) - min(all_y))

        # Build cell bboxes
        cell_bboxes = []
        for i in range(rows):
            for j in range(cols):
                # Find cell bounds
                if j < len(col_clusters) - 1:
                    cell_x = col_clusters[j] - 20
                    cell_w = col_clusters[j + 1] - col_clusters[j] + 40
                else:
                    cell_x = col_clusters[j] - 20
                    cell_w = 100

                if i < len(row_clusters) - 1:
                    cell_y = row_clusters[i] - 10
                    cell_h = row_clusters[i + 1] - row_clusters[i] + 20
                else:
                    cell_y = row_clusters[i] - 10
                    cell_h = 40

                cell_bboxes.append((cell_x, cell_y, cell_w, cell_h))

        return DetectedTable(
            bbox=bbox,
            rows=rows,
            cols=cols,
            cell_bboxes=cell_bboxes,
            confidence=0.6,  # Lower confidence for lineless detection
            has_lines=False,
            detection_method="lineless_clustering",
        )

    def detect_lineless_improved(
        self, image: np.ndarray, text_blocks: List[dict]
    ) -> List[DetectedTable]:
        """
        Improved lineless table detection using adjacency graph analysis.

        This method builds a graph of text block relationships and finds
        clusters that form table-like structures.
        """
        if not text_blocks:
            return []

        # Filter valid text blocks
        valid_blocks = [
            b for b in text_blocks
            if b.get("bbox") and len(b["bbox"]) == 4
        ]

        if len(valid_blocks) < self.config.lineless_min_blocks:
            return []

        # Build adjacency graph
        graph = self._build_adjacency_graph(valid_blocks)

        # Find grid-like clusters
        clusters = self._find_grid_clusters(graph, valid_blocks)

        # Convert clusters to tables
        tables = []
        for cluster in clusters:
            cluster_blocks = [valid_blocks[i] for i in cluster]
            if self._is_valid_table_structure(cluster_blocks):
                table = self._cluster_to_table(cluster_blocks)
                if table:
                    table.detection_method = "lineless_graph"
                    tables.append(table)

        return tables

    def _build_adjacency_graph(
        self, blocks: List[dict]
    ) -> List[List[Tuple[int, float]]]:
        """
        Build adjacency graph from text blocks.

        Each node is a text block. Edges connect blocks that are
        horizontally or vertically aligned within a threshold distance.

        Returns:
            Adjacency list: graph[i] = [(neighbor_idx, distance), ...]
        """
        n = len(blocks)
        graph = [[] for _ in range(n)]
        max_dist = self.config.adjacency_distance

        for i in range(n):
            x1, y1, w1, h1 = blocks[i]["bbox"]
            cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2

            for j in range(i + 1, n):
                x2, y2, w2, h2 = blocks[j]["bbox"]
                cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2

                # Calculate horizontal and vertical distances
                dx = abs(cx2 - cx1)
                dy = abs(cy2 - cy1)

                # Check if blocks are aligned (horizontally or vertically)
                # Horizontal alignment: similar y, different x
                h_aligned = dy < (h1 + h2) // 2 + 10
                # Vertical alignment: similar x, different y
                v_aligned = dx < (w1 + w2) // 2 + 10

                if h_aligned or v_aligned:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < max_dist:
                        # Store neighbor and whether it's horizontal or vertical
                        edge_type = "h" if h_aligned and not v_aligned else "v" if v_aligned and not h_aligned else "b"
                        graph[i].append((j, dist, edge_type))
                        graph[j].append((i, dist, edge_type))

        return graph

    def _find_grid_clusters(
        self, graph: List[List[Tuple[int, float]]], blocks: List[dict]
    ) -> List[List[int]]:
        """
        Find clusters of blocks that form grid-like structures.

        Uses connected component analysis with grid structure validation.
        """
        n = len(blocks)
        visited = [False] * n
        clusters = []

        def dfs(start: int) -> List[int]:
            """Depth-first search to find connected component."""
            stack = [start]
            component = []
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)
                for neighbor, dist, edge_type in graph[node]:
                    if not visited[neighbor]:
                        stack.append(neighbor)
            return component

        # Find all connected components
        for i in range(n):
            if not visited[i] and len(graph[i]) > 0:
                component = dfs(i)
                if len(component) >= self.config.lineless_min_blocks:
                    clusters.append(component)

        return clusters

    def _is_valid_table_structure(self, blocks: List[dict]) -> bool:
        """
        Check if a cluster of blocks forms a valid table structure.

        Validates:
        - Minimum 2x2 grid
        - Regular alignment of rows and columns
        - Consistent cell sizes
        """
        if len(blocks) < 4:
            return False

        # Extract coordinates
        x_coords = []
        y_coords = []
        for block in blocks:
            x, y, w, h = block["bbox"]
            x_coords.append(x + w // 2)
            y_coords.append(y + h // 2)

        # Cluster by position to find rows and columns
        col_clusters = self._cluster_coordinates(
            x_coords,
            tolerance=self.config.cluster_tolerance
        )
        row_clusters = self._cluster_coordinates(
            y_coords,
            tolerance=self.config.cluster_tolerance
        )

        # Must have at least 2 rows and 2 columns
        if len(col_clusters) < self.min_cols or len(row_clusters) < self.min_rows:
            return False

        # Check regularity using config thresholds
        if len(col_clusters) > 1:
            col_gaps = [col_clusters[i+1] - col_clusters[i]
                        for i in range(len(col_clusters)-1)]
            col_variance = np.std(col_gaps) / (np.mean(col_gaps) + 1e-6)
            if col_variance > self.config.max_col_variance:
                return False

        if len(row_clusters) > 1:
            row_gaps = [row_clusters[i+1] - row_clusters[i]
                        for i in range(len(row_clusters)-1)]
            row_variance = np.std(row_gaps) / (np.mean(row_gaps) + 1e-6)
            if row_variance > self.config.max_row_variance:
                return False

        return True

    def _cluster_to_table(self, blocks: List[dict]) -> Optional[DetectedTable]:
        """Convert a cluster of text blocks to a DetectedTable."""
        # Extract coordinates for clustering
        x_coords = [b["bbox"][0] + b["bbox"][2] // 2 for b in blocks]
        y_coords = [b["bbox"][1] + b["bbox"][3] // 2 for b in blocks]

        col_clusters = self._cluster_coordinates(
            x_coords,
            tolerance=self.config.cluster_tolerance
        )
        row_clusters = self._cluster_coordinates(
            y_coords,
            tolerance=self.config.cluster_tolerance
        )

        return self._build_table_from_clusters(blocks, col_clusters, row_clusters)


class HybridTableDetector:
    """
    Hybrid table detector combining OpenCV and YOLO methods.

    Strategy:
    1. Try fast OpenCV detection first
    2. If OpenCV confidence is low, use YOLO as fallback
    3. Merge results from both methods for best coverage
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize hybrid detector.

        Args:
            config: DetectorConfig instance. If None, uses global config.
        """
        self.config = config or get_detector_config()
        self.cv_detector = TableDetector(config=self.config)
        self._yolo_detector = None  # Lazy loaded
        self._yolo_load_attempted = False

    def _ensure_yolo_loaded(self) -> bool:
        """Lazy load YOLO detector."""
        if self._yolo_detector is not None:
            return True

        if self._yolo_load_attempted:
            return False

        self._yolo_load_attempted = True

        try:
            from .yolo_detector import YOLOTableDetector, is_yolo_available

            if not is_yolo_available():
                return False

            self._yolo_detector = YOLOTableDetector(
                model_path=self.config.yolo_model_path,
                confidence_threshold=self.config.yolo_confidence_threshold,
            )
            return self._yolo_detector.load_model()
        except ImportError:
            return False

    def detect(
        self,
        image: np.ndarray,
        use_yolo: Optional[bool] = None,
    ) -> List[DetectedTable]:
        """
        Detect tables using hybrid approach.

        Args:
            image: Input image as numpy array
            use_yolo: Override config.use_yolo setting

        Returns:
            List of detected tables.
        """
        should_use_yolo = use_yolo if use_yolo is not None else self.config.use_yolo

        # Step 1: OpenCV detection (fast)
        cv_tables = self.cv_detector.detect(image)

        # Check if OpenCV results are confident enough
        cv_confident = all(
            t.confidence >= self.config.yolo_fallback_threshold
            for t in cv_tables
        ) if cv_tables else False

        # Step 2: If OpenCV not confident and YOLO enabled, use YOLO
        if not cv_confident and should_use_yolo:
            if self._ensure_yolo_loaded() and self._yolo_detector is not None:
                yolo_regions = self._yolo_detector.detect(image)

                if yolo_regions:
                    # Convert YOLO regions to DetectedTable format
                    yolo_tables = self._process_yolo_regions(image, yolo_regions)

                    # Merge results
                    return self._merge_results(cv_tables, yolo_tables)

        return cv_tables

    def _process_yolo_regions(
        self,
        image: np.ndarray,
        regions: List,  # List[TableRegion]
    ) -> List[DetectedTable]:
        """
        Process YOLO detected regions to extract table structure.

        YOLO gives bounding boxes; we use OpenCV to find the grid within.
        """
        tables = []

        for region in regions:
            x, y, w, h = region.bbox

            # Extract region from image
            region_image = image[y:y+h, x:x+w]

            # Use OpenCV to detect grid structure within the region
            sub_tables = self.cv_detector.detect(region_image)

            if sub_tables:
                # Adjust coordinates to global
                for table in sub_tables:
                    tx, ty, tw, th = table.bbox
                    table.bbox = (tx + x, ty + y, tw, th)

                    # Adjust cell bboxes
                    table.cell_bboxes = [
                        (cx + x, cy + y, cw, ch)
                        for cx, cy, cw, ch in table.cell_bboxes
                    ]

                    # Combine confidence from YOLO and OpenCV
                    table.confidence = (region.confidence + table.confidence) / 2
                    table.detection_method = "hybrid_yolo_cv"
                    tables.append(table)
            else:
                # YOLO found table but OpenCV couldn't parse grid
                # Return as single-cell table for manual review
                table = DetectedTable(
                    bbox=region.bbox,
                    rows=1,
                    cols=1,
                    cell_bboxes=[region.bbox],
                    confidence=region.confidence * 0.8,  # Lower confidence
                    has_lines=False,
                    detection_method="yolo_only",
                )
                tables.append(table)

        return tables

    def _merge_results(
        self,
        cv_tables: List[DetectedTable],
        yolo_tables: List[DetectedTable],
    ) -> List[DetectedTable]:
        """
        Merge results from OpenCV and YOLO detection.

        - Tables detected by both get confidence boost
        - Non-overlapping tables from either method are kept
        """
        if not cv_tables:
            return yolo_tables
        if not yolo_tables:
            return cv_tables

        merged = []
        used_yolo = set()

        for cv_table in cv_tables:
            best_overlap = None
            best_iou = 0

            for i, yolo_table in enumerate(yolo_tables):
                if i in used_yolo:
                    continue

                if self.cv_detector._tables_overlap(cv_table, yolo_table, threshold=0.3):
                    # Calculate IOU for better matching
                    iou = self._calculate_iou(cv_table.bbox, yolo_table.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_overlap = i

            if best_overlap is not None:
                # Confirmed by both methods - boost confidence
                cv_table.confidence = min(cv_table.confidence + 0.15, 1.0)
                cv_table.detection_method = "hybrid_confirmed"
                used_yolo.add(best_overlap)

            merged.append(cv_table)

        # Add YOLO-only tables
        for i, yolo_table in enumerate(yolo_tables):
            if i not in used_yolo:
                merged.append(yolo_table)

        return merged

    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        """Calculate Intersection over Union for two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def detect_lineless(
        self,
        image: np.ndarray,
        text_blocks: List[dict],
        use_improved: bool = True,
    ) -> List[DetectedTable]:
        """
        Detect tables without visible lines.

        Args:
            image: Input image
            text_blocks: List of OCR text blocks with bbox info
            use_improved: Use improved graph-based detection

        Returns:
            List of detected lineless tables.
        """
        if use_improved:
            return self.cv_detector.detect_lineless_improved(image, text_blocks)
        else:
            return self.cv_detector.detect_lineless(image, text_blocks)
