"""
Shape-based symmetry detection using diamond's natural geometry

Key improvements:
1. Symmetry axis from fitted ellipse (major axis)
2. Brightness-based reflection detection (not edges)
3. Mirror comparison along natural axis
4. Focus on central table facet brightness
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy import ndimage


@dataclass
class ShapeSymmetryResult:
    """Result of shape-based symmetry analysis"""
    major_axis_angle: float  # Angle of major axis (degrees)
    minor_axis_angle: float  # Angle of minor axis (degrees)
    symmetry_score_major: float  # Symmetry along major axis [0, 1]
    symmetry_score_minor: float  # Symmetry along minor axis [0, 1]
    best_symmetry_score: float  # Max of major/minor symmetry
    table_brightness: float  # Central brightness (table facet indicator)
    brightness_uniformity: float  # How uniform brightness is (table-up = more uniform)
    directional_gradient: float  # Directional brightness bias (tilted = high)
    num_bright_spots: int  # Number of distinct bright regions
    axis_line_major: Tuple[Tuple[int, int], Tuple[int, int]]  # Major axis line
    axis_line_minor: Tuple[Tuple[int, int], Tuple[int, int]]  # Minor axis line
    aspect_ratio: float  # Minor/major axis ratio
    circularity: float  # Shape circularity
    is_circular: bool  # True if near-perfect circle (round brilliant cut)
    ring_brightness: float  # Brightness of outer ring (for circular diamonds)
    multi_axis_symmetry: float  # Average symmetry across multiple angles (for circular)
    num_symmetric_axes: int  # Number of axes with good symmetry (for circular)


class ShapeBasedSymmetryDetector:
    """
    Symmetry detection based on diamond's natural geometry

    Approach:
    1. Fit ellipse to smoothed outline → get natural axes
    2. Extract bright regions (table facet reflections)
    3. Mirror image along major/minor axes
    4. Compare brightness distributions
    5. Measure central brightness and uniformity
    """

    def __init__(
        self,
        brightness_percentile: float = 70,  # Top 30% brightest pixels
        min_bright_area: int = 10,
        max_bright_area: int = 1000
    ):
        """
        Args:
            brightness_percentile: Percentile threshold for bright regions
            min_bright_area: Minimum area for valid bright spot
            max_bright_area: Maximum area for valid bright spot
        """
        self.brightness_percentile = brightness_percentile
        self.min_bright_area = min_bright_area
        self.max_bright_area = max_bright_area

    def smooth_contour(self, mask: np.ndarray) -> np.ndarray:
        """
        Smooth contour using polygon approximation

        Args:
            mask: Binary mask

        Returns:
            smoothed_mask: Mask with smoothed contour
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return mask

        # Get largest contour
        contour = max(contours, key=cv2.contourArea)

        # Smooth using polygon approximation
        epsilon = 0.01 * cv2.arcLength(contour, True)
        smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)

        # Create smoothed mask
        smoothed_mask = np.zeros_like(mask)
        cv2.drawContours(smoothed_mask, [smoothed_contour], -1, 255, -1)

        return smoothed_mask

    def fit_ellipse_to_shape(
        self,
        mask: np.ndarray
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """
        Fit ellipse to smoothed mask to get natural axes

        Args:
            mask: Binary mask

        Returns:
            ellipse: ((cx, cy), (minor, major), angle) or None if fit fails
        """
        # Smooth contour first
        smoothed_mask = self.smooth_contour(mask)

        contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        contour = max(contours, key=cv2.contourArea)

        if len(contour) < 5:
            # Fallback: Try with original unsmoothed mask
            contours_orig, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours_orig) > 0:
                contour_orig = max(contours_orig, key=cv2.contourArea)
                if len(contour_orig) >= 5:
                    ellipse = cv2.fitEllipse(contour_orig)
                    return ellipse
            return None

        # Fit ellipse
        ellipse = cv2.fitEllipse(contour)
        return ellipse

    def get_axis_lines(
        self,
        ellipse: Tuple,
        image_shape: Tuple[int, int]
    ) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Get major and minor axis lines from fitted ellipse

        Args:
            ellipse: ((cx, cy), (width, height), angle)
            image_shape: (height, width)

        Returns:
            (major_axis_line, minor_axis_line)
        """
        (cx, cy), (width, height), angle = ellipse

        h, w = image_shape

        # Major axis is along the ellipse angle
        major_angle_rad = np.deg2rad(angle)
        minor_angle_rad = np.deg2rad(angle + 90)

        # Use image diagonal as line length
        length = int(np.sqrt(h**2 + w**2))

        # Major axis
        major_dx = int(length * np.cos(major_angle_rad))
        major_dy = int(length * np.sin(major_angle_rad))
        major_start = (int(cx - major_dx), int(cy - major_dy))
        major_end = (int(cx + major_dx), int(cy + major_dy))

        # Minor axis (perpendicular)
        minor_dx = int(length * np.cos(minor_angle_rad))
        minor_dy = int(length * np.sin(minor_angle_rad))
        minor_start = (int(cx - minor_dx), int(cy - minor_dy))
        minor_end = (int(cx + minor_dx), int(cy + minor_dy))

        return (major_start, major_end), (minor_start, minor_end)

    def extract_bright_regions(
        self,
        roi_image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        Extract bright regions (facet reflections) instead of edges

        Args:
            roi_image: Grayscale ROI
            mask: Binary mask

        Returns:
            bright_mask: Binary mask of bright regions
            num_spots: Number of distinct bright spots
        """
        # Apply mask to focus on diamond interior
        masked_image = roi_image.copy()
        masked_image[mask == 0] = 0

        # Get brightness threshold (top X% brightest pixels)
        masked_pixels = roi_image[mask > 0]
        if len(masked_pixels) == 0:
            return np.zeros_like(roi_image), 0

        threshold = np.percentile(masked_pixels, self.brightness_percentile)

        # Create bright region mask
        bright_mask = np.zeros_like(roi_image, dtype=np.uint8)
        bright_mask[(roi_image >= threshold) & (mask > 0)] = 255

        # Erode to remove outer edge brightness
        kernel = np.ones((5, 5), np.uint8)
        inner_mask = cv2.erode(mask, kernel, iterations=2)
        bright_mask = cv2.bitwise_and(bright_mask, bright_mask, mask=inner_mask)

        # Filter by size
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

        clean_bright_mask = np.zeros_like(bright_mask)
        num_spots = 0

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if self.min_bright_area <= area <= self.max_bright_area:
                clean_bright_mask[labels == label_id] = 255
                num_spots += 1

        return clean_bright_mask, num_spots

    def compute_mirror_symmetry(
        self,
        roi_image: np.ndarray,
        mask: np.ndarray,
        angle: float
    ) -> float:
        """
        Compute symmetry by mirroring image along axis and comparing brightness

        Args:
            roi_image: Grayscale ROI
            mask: Binary mask
            angle: Axis angle in degrees

        Returns:
            symmetry_score: Correlation between original and mirrored [0, 1]
        """
        h, w = roi_image.shape
        center = (w // 2, h // 2)

        # Rotate image so axis is vertical (easier to split/mirror)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle - 90, 1.0)
        rotated_image = cv2.warpAffine(roi_image, rotation_matrix, (w, h))
        rotated_mask = cv2.warpAffine(mask, rotation_matrix, (w, h))

        # Split in half vertically
        left = rotated_image[:, :w//2]
        right = rotated_image[:, w//2:w//2 + left.shape[1]]  # Match size

        left_mask = rotated_mask[:, :w//2]
        right_mask = rotated_mask[:, w//2:w//2 + left.shape[1]]

        # Mirror right half
        right_mirror = cv2.flip(right, 1)
        right_mask_mirror = cv2.flip(right_mask, 1)

        # Compare only where both sides have valid mask
        valid_mask = (left_mask > 127) & (right_mask_mirror > 127)

        if valid_mask.sum() < 10:  # Need at least 10 pixels
            return 0.0

        left_pixels = left[valid_mask].astype(np.float32)
        right_pixels = right_mirror[valid_mask].astype(np.float32)

        # Compute correlation
        if len(left_pixels) < 2:
            return 0.0

        # Normalize to [0, 1]
        left_norm = (left_pixels - left_pixels.mean()) / (left_pixels.std() + 1e-8)
        right_norm = (right_pixels - right_pixels.mean()) / (right_pixels.std() + 1e-8)

        correlation = np.corrcoef(left_norm, right_norm)[0, 1]

        # Map correlation [-1, 1] to symmetry score [0, 1]
        # High positive correlation = symmetric
        symmetry = (correlation + 1.0) / 2.0

        # Also check brightness difference (should be small for symmetric)
        brightness_diff = np.abs(left_pixels.mean() - right_pixels.mean()) / 255.0
        brightness_similarity = 1.0 - brightness_diff

        # Combine correlation and brightness similarity
        final_score = 0.7 * symmetry + 0.3 * brightness_similarity

        return np.clip(final_score, 0.0, 1.0)

    def compute_central_brightness(
        self,
        roi_image: np.ndarray,
        mask: np.ndarray,
        radius_fraction: float = 0.2
    ) -> float:
        """
        Measure brightness in central region (table facet indicator)

        Args:
            roi_image: Grayscale ROI
            mask: Binary mask
            radius_fraction: Fraction of image size for central region

        Returns:
            central_brightness: Normalized brightness [0, 1]
        """
        h, w = roi_image.shape
        center_y, center_x = h // 2, w // 2
        radius = int(min(h, w) * radius_fraction)

        y1 = max(0, center_y - radius)
        y2 = min(h, center_y + radius)
        x1 = max(0, center_x - radius)
        x2 = min(w, center_x + radius)

        central_region = roi_image[y1:y2, x1:x2]
        central_mask = mask[y1:y2, x1:x2]

        if central_mask.sum() == 0:
            return 0.0

        central_pixels = central_region[central_mask > 127]

        if len(central_pixels) == 0:
            return 0.0

        return central_pixels.mean() / 255.0

    def compute_brightness_uniformity(
        self,
        roi_image: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Measure how uniform the brightness is (table-up = more uniform)

        Args:
            roi_image: Grayscale ROI
            mask: Binary mask

        Returns:
            uniformity: 1.0 = perfectly uniform, 0.0 = very varied [0, 1]
        """
        pixels = roi_image[mask > 127]

        if len(pixels) == 0:
            return 0.0

        # Low standard deviation = more uniform
        std = pixels.std()

        # Normalize: std of 0 = perfectly uniform (1.0), std of 128 = very varied (0.0)
        uniformity = 1.0 - np.clip(std / 128.0, 0.0, 1.0)

        return uniformity

    def compute_directional_gradient(
        self,
        roi_image: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Measure directional brightness gradient (tilted = bright on one side)

        Args:
            roi_image: Grayscale ROI
            mask: Binary mask

        Returns:
            gradient: 0.0 = uniform, 1.0 = strong directional gradient [0, 1]
        """
        h, w = roi_image.shape

        # Compare left vs right halves
        left_mask = mask[:, :w//2]
        right_mask = mask[:, w//2:]

        left_pixels = roi_image[:, :w//2][left_mask > 127]
        right_pixels = roi_image[:, w//2:][right_mask > 127]

        if len(left_pixels) == 0 or len(right_pixels) == 0:
            return 0.0

        left_mean = left_pixels.mean()
        right_mean = right_pixels.mean()

        # Normalize difference
        horizontal_gradient = abs(left_mean - right_mean) / 255.0

        # Compare top vs bottom halves
        top_mask = mask[:h//2, :]
        bottom_mask = mask[h//2:, :]

        top_pixels = roi_image[:h//2, :][top_mask > 127]
        bottom_pixels = roi_image[h//2:, :][bottom_mask > 127]

        if len(top_pixels) == 0 or len(bottom_pixels) == 0:
            vertical_gradient = 0.0
        else:
            top_mean = top_pixels.mean()
            bottom_mean = bottom_pixels.mean()
            vertical_gradient = abs(top_mean - bottom_mean) / 255.0

        # Return maximum gradient (strongest directional bias)
        return max(horizontal_gradient, vertical_gradient)

    def compute_ring_brightness(
        self,
        roi_image: np.ndarray,
        mask: np.ndarray,
        inner_fraction: float = 0.25,
        outer_fraction: float = 0.50
    ) -> float:
        """
        Measure brightness of outer ring (for circular diamonds)

        Round brilliant cut diamonds when table-up have:
        - Dark center (table facet appears dark)
        - Lighter ring near edge (crown facets catch light)

        Args:
            roi_image: Grayscale ROI
            mask: Binary mask
            inner_fraction: Inner radius as fraction of image size
            outer_fraction: Outer radius as fraction of image size

        Returns:
            ring_brightness: Normalized brightness of ring [0, 1]
        """
        h, w = roi_image.shape
        center_y, center_x = h // 2, w // 2
        max_radius = min(h, w) / 2

        inner_radius = int(max_radius * inner_fraction)
        outer_radius = int(max_radius * outer_fraction)

        # Create ring mask
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)

        ring_mask = ((distances >= inner_radius) &
                     (distances <= outer_radius) &
                     (mask > 127))

        if ring_mask.sum() == 0:
            return 0.0

        ring_pixels = roi_image[ring_mask]

        if len(ring_pixels) == 0:
            return 0.0

        return ring_pixels.mean() / 255.0

    def is_circular_diamond(
        self,
        circularity: float,
        aspect_ratio: float,
        roi_image: np.ndarray,
        mask: np.ndarray,
        circularity_threshold: float = 0.85,
        aspect_ratio_threshold: float = 0.85,
        verify_dark_center: bool = True
    ) -> bool:
        """
        Detect if diamond is near-perfect circle (round brilliant cut)

        Args:
            circularity: Shape circularity (4π*area/perimeter²)
            aspect_ratio: Minor/major axis ratio
            roi_image: Grayscale ROI
            mask: Binary mask
            circularity_threshold: Minimum circularity for circle
            aspect_ratio_threshold: Minimum aspect ratio for circle
            verify_dark_center: Check for dark center + light ring pattern

        Returns:
            is_circular: True if near-perfect circle
        """
        # Basic shape checks
        if circularity < circularity_threshold:
            return False

        if aspect_ratio < aspect_ratio_threshold:
            return False

        # Optional: Verify dark center + light ring pattern
        if verify_dark_center:
            # Get central brightness
            center_brightness = self.compute_central_brightness(
                roi_image, mask, radius_fraction=0.20
            )

            # Get ring brightness
            ring_brightness = self.compute_ring_brightness(
                roi_image, mask, inner_fraction=0.30, outer_fraction=0.60
            )

            # Round brilliants typically have: dark center, lighter ring
            # If center is significantly darker than ring, this confirms circularity
            if ring_brightness > center_brightness + 0.05:  # Ring at least 5% brighter
                return True
            # Even if not dark center pattern, accept if shape is very circular
            elif circularity > 0.92 and aspect_ratio > 0.92:
                return True
            else:
                return False

        return True

    def compute_multi_axis_symmetry(
        self,
        roi_image: np.ndarray,
        mask: np.ndarray,
        num_angles: int = 8
    ) -> Tuple[float, int]:
        """
        Check symmetry at multiple rotation angles (for circular diamonds)

        Round diamonds on table have radial symmetry - similar appearance
        when rotated at multiple angles (e.g., every 45°).

        Non-round diamonds on side won't have this multi-axis symmetry.

        Args:
            roi_image: Grayscale ROI
            mask: Binary mask
            num_angles: Number of angles to check (default: 8 = every 45°)

        Returns:
            (avg_symmetry, num_symmetric_axes): Average symmetry and count of symmetric axes
        """
        angles = [i * 180.0 / num_angles for i in range(num_angles)]
        symmetry_scores = []

        threshold = 0.75  # Threshold for considering an axis "symmetric"
        num_symmetric = 0

        for angle in angles:
            symmetry = self.compute_mirror_symmetry(roi_image, mask, angle)
            symmetry_scores.append(symmetry)
            if symmetry > threshold:
                num_symmetric += 1

        avg_symmetry = np.mean(symmetry_scores)

        return avg_symmetry, num_symmetric

    def analyze_symmetry(
        self,
        roi_image: np.ndarray,
        mask: np.ndarray
    ) -> ShapeSymmetryResult:
        """
        Complete shape-based symmetry analysis

        Args:
            roi_image: Grayscale ROI
            mask: Binary mask

        Returns:
            result: ShapeSymmetryResult
        """
        # Apply CLAHE to normalize lighting (fixes inconsistent illumination)
        # This helps when one quadrant is lighter than the rest
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        roi_image_normalized = clahe.apply(roi_image)

        # Use normalized image for symmetry analysis
        roi_image = roi_image_normalized

        # Fit ellipse to get natural axes
        ellipse = self.fit_ellipse_to_shape(mask)

        if ellipse is None:
            # Fallback: Compute circularity from raw contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
                circularity = min(circularity, 1.0)
            else:
                circularity = 0.0

            # Fallback to horizontal/vertical axes
            major_angle = 0.0
            minor_angle = 90.0
            aspect_ratio = 1.0
            axis_line_major = ((0, 0), (1, 1))
            axis_line_minor = ((0, 0), (1, 1))
        else:
            (cx, cy), (width, height), angle = ellipse

            # Major axis is the longer one
            major_angle = angle
            minor_angle = angle + 90.0

            # Aspect ratio
            aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 1.0

            # Circularity from mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
                circularity = min(circularity, 1.0)
            else:
                circularity = 0.0

            # Get axis lines for visualization
            axis_line_major, axis_line_minor = self.get_axis_lines(ellipse, roi_image.shape)

        # Compute symmetry along both axes
        symmetry_major = self.compute_mirror_symmetry(roi_image, mask, major_angle)
        symmetry_minor = self.compute_mirror_symmetry(roi_image, mask, minor_angle)

        best_symmetry = max(symmetry_major, symmetry_minor)

        # Extract bright regions
        bright_mask, num_bright = self.extract_bright_regions(roi_image, mask)

        # Central brightness (table facet) - larger region for emerald cuts
        table_brightness = self.compute_central_brightness(roi_image, mask, radius_fraction=0.25)

        # Brightness uniformity
        uniformity = self.compute_brightness_uniformity(roi_image, mask)

        # Directional gradient (tilted diamonds are bright on one side)
        directional_grad = self.compute_directional_gradient(roi_image, mask)

        # Detect if diamond is circular (round brilliant cut)
        # Lower threshold for SAM-detected diamonds (masks aren't pixel-perfect)
        is_circular = self.is_circular_diamond(
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            roi_image=roi_image,
            mask=mask,
            circularity_threshold=0.75,  # Lowered from 0.85 for SAM masks
            aspect_ratio_threshold=0.85
        )

        # Compute ring brightness (for circular diamonds)
        ring_brightness = self.compute_ring_brightness(roi_image, mask)

        # For circular diamonds: check multi-axis symmetry (radial symmetry test)
        # Round diamonds on table have similar appearance from many angles
        if is_circular:
            multi_axis_sym, num_sym_axes = self.compute_multi_axis_symmetry(roi_image, mask, num_angles=8)
        else:
            # Non-circular: just use best of major/minor
            multi_axis_sym = best_symmetry
            num_sym_axes = 0

        result = ShapeSymmetryResult(
            major_axis_angle=major_angle,
            minor_axis_angle=minor_angle,
            symmetry_score_major=symmetry_major,
            symmetry_score_minor=symmetry_minor,
            best_symmetry_score=best_symmetry,
            table_brightness=table_brightness,
            brightness_uniformity=uniformity,
            directional_gradient=directional_grad,
            num_bright_spots=num_bright,
            axis_line_major=axis_line_major,
            axis_line_minor=axis_line_minor,
            aspect_ratio=aspect_ratio,
            circularity=circularity,
            is_circular=is_circular,
            ring_brightness=ring_brightness,
            multi_axis_symmetry=multi_axis_sym,
            num_symmetric_axes=num_sym_axes
        )

        return result
