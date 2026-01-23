import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image

from ..models import PreprocessingProfile


class ImagePreprocessor:
    """Handles image preprocessing for OCR and table detection."""

    def __init__(self, profile: Optional[PreprocessingProfile] = None):
        self.profile = profile or PreprocessingProfile()

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline based on profile."""
        result = image.copy()
        is_photo = self.profile.name == "photo"

        # For photos: first apply glare removal on color image
        if is_photo and len(result.shape) == 3:
            result = self._remove_glare(result)
            # Result is now grayscale
            gray = result
        else:
            # Convert to grayscale if needed
            if len(result.shape) == 3:
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            else:
                gray = result

        # Apply preprocessing steps based on profile
        if self.profile.shadow_removal:
            gray = self._remove_shadows(gray)

        # For photos: enhance lighting before other processing
        if is_photo:
            gray = self._enhance_lighting(gray)

        if self.profile.denoise:
            gray = self._denoise(gray)

        if self.profile.deskew:
            gray = self._deskew(gray)

        if self.profile.perspective_correction:
            gray = self._correct_perspective(gray)

        if self.profile.contrast_enhance:
            gray = self._enhance_contrast(gray)

        # For photos: sharpen text after other processing
        if is_photo:
            gray = self._sharpen_text(gray)

        if self.profile.line_enhancement:
            gray = self._enhance_lines(gray)

        return gray

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image."""
        return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew."""
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=100, maxLineGap=10
        )

        if lines is None:
            return image

        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # Only consider nearly horizontal lines
            if abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return image

        # Get median angle
        median_angle = np.median(angles)

        # Rotate image
        if abs(median_angle) > 0.5:  # Only correct if skew > 0.5 degrees
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image, rotation_matrix, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated

        return image

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def _enhance_lighting(self, image: np.ndarray) -> np.ndarray:
        """Correct uneven lighting using LAB color space.

        Particularly useful for photos with shadows or uneven illumination.
        """
        # Need BGR image for LAB conversion
        if len(image.shape) == 2:
            # Convert grayscale to BGR
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image

        # Convert to LAB color space
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        # Split channels
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel (lightness)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a, b])

        # Convert back to BGR then grayscale
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    def _remove_glare(self, image: np.ndarray) -> np.ndarray:
        """Remove glare/reflections from photos using HSV analysis.

        Detects very bright spots and uses inpainting to restore them.
        """
        # Need BGR image for HSV conversion
        if len(image.shape) == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image

        # Convert to HSV
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Detect very bright areas (glare)
        # V channel > 250 and S channel < 50 (white/bright spots)
        _, _, v = cv2.split(hsv)
        _, s, _ = cv2.split(hsv)

        # Create mask for glare regions
        bright_mask = (v > 245).astype(np.uint8) * 255
        low_sat_mask = (s < 30).astype(np.uint8) * 255
        glare_mask = cv2.bitwise_and(bright_mask, low_sat_mask)

        # Dilate mask slightly to cover edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)

        # Check if there's significant glare to remove
        if np.sum(glare_mask) / 255 < 100:  # Less than 100 pixels
            if len(image.shape) == 2:
                return image
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Use inpainting to fill glare regions
        result = cv2.inpaint(bgr, glare_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    def _sharpen_text(self, image: np.ndarray) -> np.ndarray:
        """Sharpen text using unsharp mask technique.

        Improves text clarity for better OCR recognition.
        """
        # Apply Gaussian blur
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)

        # Unsharp mask: original + (original - blurred) * amount
        # Using addWeighted: amount=1.5, blur_weight=-0.5
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

        return sharpened

    def _enhance_lines(self, image: np.ndarray) -> np.ndarray:
        """Enhance table lines."""
        # Apply morphological operations to enhance lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        # Detect horizontal lines
        horizontal = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_h)

        # Detect vertical lines
        vertical = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_v)

        # Combine with original
        lines_mask = cv2.add(horizontal, vertical)
        enhanced = cv2.addWeighted(image, 0.8, lines_mask, 0.2, 0)

        return enhanced

    def _remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """Remove shadows from image (for photos)."""
        # Dilate to get background
        dilated = cv2.dilate(image, np.ones((7, 7), np.uint8))
        blurred = cv2.medianBlur(dilated, 21)

        # Difference between original and background
        diff = 255 - cv2.absdiff(image, blurred)

        # Normalize
        normalized = cv2.normalize(diff, None, alpha=0, beta=255,
                                   norm_type=cv2.NORM_MINMAX)

        return normalized

    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """Correct perspective distortion (for photos).

        Uses multiple detection methods with validation.
        """
        # Try contour-based detection first
        result = self._detect_document_contour(image)
        if result is not None:
            return result

        # Fallback to Hough lines if contour method fails
        result = self._detect_document_hough(image)
        if result is not None:
            return result

        return image

    def _detect_document_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect document using contour detection."""
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 200)

        # Dilate to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        h, w = image.shape[:2]
        min_area = h * w * 0.1  # At least 10% of image area

        # Find quadrilateral candidates
        candidates = []
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                rect = self._order_points(pts)

                # Validate angles (should be 60-120 degrees)
                if self._validate_quadrilateral_angles(rect):
                    candidates.append((area, rect))

        if not candidates:
            return None

        # Use largest valid candidate
        _, rect = max(candidates, key=lambda x: x[0])
        return self._apply_perspective_transform(image, rect)

    def _detect_document_hough(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect document using Hough line detection (fallback)."""
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None or len(lines) < 4:
            return None

        # Separate horizontal and vertical lines
        h_lines = []
        v_lines = []

        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi

            if 80 < angle < 100:  # Horizontal-ish
                h_lines.append((rho, theta))
            elif angle < 10 or angle > 170:  # Vertical-ish
                v_lines.append((rho, theta))

        if len(h_lines) < 2 or len(v_lines) < 2:
            return None

        # Sort to find extreme lines
        h_lines = sorted(h_lines, key=lambda x: x[0])
        v_lines = sorted(v_lines, key=lambda x: x[0])

        # Get intersection points of extreme lines
        top_line = h_lines[0]
        bottom_line = h_lines[-1]
        left_line = v_lines[0]
        right_line = v_lines[-1]

        corners = []
        for h in [top_line, bottom_line]:
            for v in [left_line, right_line]:
                pt = self._line_intersection(h, v)
                if pt is not None:
                    corners.append(pt)

        if len(corners) != 4:
            return None

        rect = self._order_points(np.array(corners))

        # Validate
        if not self._validate_quadrilateral_angles(rect):
            return None

        return self._apply_perspective_transform(image, rect)

    def _line_intersection(self, line1: Tuple, line2: Tuple) -> Optional[Tuple[float, float]]:
        """Find intersection point of two Hough lines."""
        rho1, theta1 = line1
        rho2, theta2 = line2

        cos1, sin1 = np.cos(theta1), np.sin(theta1)
        cos2, sin2 = np.cos(theta2), np.sin(theta2)

        det = cos1 * sin2 - cos2 * sin1
        if abs(det) < 1e-10:
            return None

        x = (sin2 * rho1 - sin1 * rho2) / det
        y = (cos1 * rho2 - cos2 * rho1) / det

        return (x, y)

    def _validate_quadrilateral_angles(self, pts: np.ndarray) -> bool:
        """Validate that quadrilateral angles are reasonable (60-120 degrees)."""
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            p3 = pts[(i + 2) % 4]

            # Vectors from p2
            v1 = p1 - p2
            v2 = p3 - p2

            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

            if angle < 60 or angle > 120:
                return False

        return True

    def _apply_perspective_transform(self, image: np.ndarray, rect: np.ndarray) -> np.ndarray:
        """Apply perspective transform given ordered corner points."""
        (tl, tr, br, bl) = rect

        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        if max_width < 100 or max_height < 100:
            return image

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        return warped

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Convert image to binary (black and white)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Adaptive thresholding for better results
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        return binary

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """Load image from file path."""
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not load image: {path}")
        return image

    @staticmethod
    def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format."""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL format."""
        if len(cv2_image.shape) == 2:
            return Image.fromarray(cv2_image)
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
