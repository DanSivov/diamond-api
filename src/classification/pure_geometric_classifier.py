"""
Pure geometric classifier for diamond orientation

Based on three key principles:
1. Outline symmetry: Strong axis of symmetry in contour shape
2. Inner reflection symmetry: Light reflections follow the outline symmetry axis
3. Large spot analysis: Well-segmented light/dark regions (symmetric on table)

No CNN required - purely geometric analysis.
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class GeometricAnalysisResult:
    """Results of geometric analysis"""
    # Metric 1: Outline symmetry
    outline_symmetry_score: float  # [0, 1] - how symmetric is the contour shape
    outline_symmetry_axis: float   # Angle of best symmetry axis

    # Metric 2: Inner reflection symmetry
    reflection_symmetry_score: float  # [0, 1] - how symmetric are internal reflections
    num_reflection_spots: int         # Number of bright spots detected

    # Metric 3: Large spot analysis
    has_large_central_spot: bool      # True if there's a large central region
    spot_symmetry_score: float        # [0, 1] - how symmetric is the spot
    spot_is_light: bool              # True if spot is bright, False if dark

    # Metric 4: Aspect ratio
    aspect_ratio: float               # Minor/major axis ratio [0, 1]

    # Final decision
    orientation: str                  # 'table' or 'tilted'
    confidence: float                 # [0, 1] - confidence in decision


class PureGeometricClassifier:
    """
    Pure geometric classifier for emerald diamond orientation

    Uses only geometric features - no CNN required.
    """

    def __init__(self,
                 outline_sym_threshold: float = 0.70,
                 reflection_sym_threshold: float = 0.65,
                 spot_sym_threshold: float = 0.60,
                 aspect_ratio_threshold: float = 0.58):
        """
        Initialize classifier

        Args:
            outline_sym_threshold: Min outline symmetry for 'table' (default 0.70)
            reflection_sym_threshold: Min reflection symmetry for 'table' (default 0.65)
            spot_sym_threshold: Min spot symmetry for 'table' (default 0.60)
            aspect_ratio_threshold: Min aspect ratio for 'table' (default 0.58)
        """
        self.outline_sym_threshold = outline_sym_threshold
        self.reflection_sym_threshold = reflection_sym_threshold
        self.spot_sym_threshold = spot_sym_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold

        # Use existing symmetry detector for reliable measurements
        from classification.shape_based_symmetry import ShapeBasedSymmetryDetector
        self.symmetry_detector = ShapeBasedSymmetryDetector()

    def classify_orientation(self, contour: np.ndarray, mask: np.ndarray = None,
                            roi_image: np.ndarray = None) -> Tuple[str, float]:
        """
        Classify diamond orientation using pure geometric analysis

        Args:
            contour: Diamond contour
            mask: Binary mask (required)
            roi_image: ROI image in BGR format (required)

        Returns:
            (orientation, confidence):
            - orientation: 'table' or 'tilted'
            - confidence: confidence score (0-1)
        """
        if roi_image is None or mask is None or len(contour) < 5:
            return 'tilted', 0.0

        # Perform full analysis
        result = self.analyze(contour, mask, roi_image)

        return result.orientation, result.confidence

    def analyze(self, contour: np.ndarray, mask: np.ndarray,
                roi_image: np.ndarray) -> GeometricAnalysisResult:
        """
        Complete geometric analysis

        Args:
            contour: Diamond contour
            mask: Binary mask
            roi_image: ROI image (BGR)

        Returns:
            GeometricAnalysisResult with all metrics
        """
        # Convert to grayscale
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image

        # Metric 1: Outline symmetry (using overall brightness pattern)
        outline_sym_score, outline_sym_axis = self._analyze_outline_symmetry(contour, mask, gray)

        # Metric 2: Inner reflection symmetry (along the outline symmetry axis)
        reflection_sym_score, num_spots = self._analyze_reflection_symmetry(
            gray, mask, outline_sym_axis
        )

        # Metric 3: Large spot analysis
        has_spot, spot_sym, spot_is_light = self._analyze_large_spots(
            gray, mask, outline_sym_axis
        )

        # Metric 4: Aspect ratio
        aspect_ratio = self._calculate_aspect_ratio(contour)

        # Decision logic (now using 4 metrics)
        orientation, confidence = self._make_decision(
            outline_sym_score, reflection_sym_score, has_spot, spot_sym, aspect_ratio
        )

        return GeometricAnalysisResult(
            outline_symmetry_score=outline_sym_score,
            outline_symmetry_axis=outline_sym_axis,
            reflection_symmetry_score=reflection_sym_score,
            num_reflection_spots=num_spots,
            has_large_central_spot=has_spot,
            spot_symmetry_score=spot_sym,
            spot_is_light=spot_is_light,
            aspect_ratio=aspect_ratio,
            orientation=orientation,
            confidence=confidence
        )

    def _analyze_outline_symmetry(self, contour: np.ndarray, mask: np.ndarray,
                                   gray: np.ndarray) -> Tuple[float, float]:
        """
        Metric 1: Analyze outline symmetry

        Use the working ShapeBasedSymmetryDetector to find best symmetry axis.
        TABLE diamonds have strong symmetry, TILTED diamonds have weak symmetry.

        Returns:
            (symmetry_score, best_axis_angle)
        """
        if len(contour) < 5:
            return 0.0, 0.0

        # Use the proven symmetry detector
        result = self.symmetry_detector.analyze_symmetry(gray, mask)

        # Get best symmetry score and its axis
        best_sym = result.best_symmetry_score

        # Use the axis with better symmetry
        if result.symmetry_score_major >= result.symmetry_score_minor:
            best_axis = result.major_axis_angle
        else:
            best_axis = result.minor_axis_angle

        return best_sym, best_axis

    def _measure_contour_symmetry(self, contour: np.ndarray, mask: np.ndarray,
                                   axis_angle: float, center: Tuple[int, int]) -> float:
        """
        Measure how symmetric the contour is along an axis

        Strategy: Rotate so axis is vertical, compare left and right halves
        """
        h, w = mask.shape

        # Create contour mask
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

        # Rotate so axis is vertical
        rotation_matrix = cv2.getRotationMatrix2D(center, axis_angle - 90, 1.0)
        rotated_mask = cv2.warpAffine(contour_mask, rotation_matrix, (w, h))

        # Split in half vertically
        left = rotated_mask[:, :w//2]
        right = rotated_mask[:, w//2:w//2 + left.shape[1]]

        # Mirror right half
        right_mirror = cv2.flip(right, 1)

        # Calculate overlap
        # For binary masks: symmetric if pixels match
        total_pixels = left.size
        matching_pixels = np.sum(left == right_mirror)

        symmetry = matching_pixels / total_pixels if total_pixels > 0 else 0.0

        return symmetry

    def _analyze_reflection_symmetry(self, gray: np.ndarray, mask: np.ndarray,
                                      axis_angle: float) -> Tuple[float, int]:
        """
        Metric 2: Analyze inner reflection symmetry

        Extract bright regions (reflections) and check if they're symmetric
        along the outline symmetry axis.

        Returns:
            (reflection_symmetry_score, num_reflection_spots)
        """
        # Extract inner reflections (remove edge brightness)
        bright_mask, num_spots = self._extract_inner_reflections(gray, mask)

        if num_spots == 0:
            return 0.0, 0

        # Find center for rotation
        M = cv2.moments(mask)
        if M['m00'] == 0:
            return 0.0, num_spots

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Measure symmetry of reflections along axis
        h, w = bright_mask.shape

        # Rotate so axis is vertical
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), axis_angle - 90, 1.0)
        rotated_bright = cv2.warpAffine(bright_mask, rotation_matrix, (w, h))
        rotated_mask = cv2.warpAffine(mask, rotation_matrix, (w, h))

        # Split in half
        left = rotated_bright[:, :w//2]
        right = rotated_bright[:, w//2:w//2 + left.shape[1]]

        left_mask = rotated_mask[:, :w//2]
        right_mask = rotated_mask[:, w//2:w//2 + left.shape[1]]

        # Mirror right
        right_mirror = cv2.flip(right, 1)
        right_mask_mirror = cv2.flip(right_mask, 1)

        # Valid comparison region
        valid = (left_mask > 127) & (right_mask_mirror > 127)

        if valid.sum() < 10:
            return 0.0, num_spots

        # Compare brightness similarity (not exact matching)
        left_pixels = left[valid].astype(np.float32)
        right_pixels = right_mirror[valid].astype(np.float32)

        # Normalize and compute correlation
        if len(left_pixels) < 2:
            return 0.0, num_spots

        left_std = left_pixels.std()
        right_std = right_pixels.std()

        # If std is too small (uniform region), use simple difference
        if left_std < 1e-6 or right_std < 1e-6:
            # Calculate simple brightness similarity
            brightness_diff = np.abs(left_pixels.mean() - right_pixels.mean()) / 255.0
            symmetry = 1.0 - brightness_diff
            return np.clip(symmetry, 0.0, 1.0), num_spots

        left_norm = (left_pixels - left_pixels.mean()) / (left_std + 1e-8)
        right_norm = (right_pixels - right_pixels.mean()) / (right_std + 1e-8)

        try:
            correlation = np.corrcoef(left_norm, right_norm)[0, 1]
            # Handle NaN from correlation
            if np.isnan(correlation):
                # Fallback to brightness similarity
                brightness_diff = np.abs(left_pixels.mean() - right_pixels.mean()) / 255.0
                correlation = 1.0 - 2.0 * brightness_diff  # Map to [-1, 1]
        except:
            # Fallback in case of error
            brightness_diff = np.abs(left_pixels.mean() - right_pixels.mean()) / 255.0
            correlation = 1.0 - 2.0 * brightness_diff

        # Map correlation to [0, 1]
        symmetry = (correlation + 1.0) / 2.0

        return np.clip(symmetry, 0.0, 1.0), num_spots

    def _extract_inner_reflections(self, gray: np.ndarray,
                                    mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Extract bright regions inside the diamond (inner reflections)

        Remove edge highlights to focus only on internal facet reflections.

        Returns:
            (bright_mask, num_spots)
        """
        # Apply mask
        masked = gray.copy()
        masked[mask == 0] = 0

        # Get brightness threshold (top 20% brightest pixels)
        masked_pixels = gray[mask > 0]
        if len(masked_pixels) == 0:
            return np.zeros_like(gray, dtype=np.uint8), 0

        threshold = np.percentile(masked_pixels, 80)

        # Create bright region mask
        bright_mask = np.zeros_like(gray, dtype=np.uint8)
        bright_mask[(gray >= threshold) & (mask > 0)] = 255

        # Erode to remove edge brightness (CRITICAL for inner reflections only)
        kernel = np.ones((5, 5), np.uint8)
        inner_mask = cv2.erode(mask, kernel, iterations=2)
        bright_mask = cv2.bitwise_and(bright_mask, bright_mask, mask=inner_mask)

        # Count spots
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

        # Filter small noise
        clean_bright = np.zeros_like(bright_mask)
        num_spots = 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= 5:  # Min 5 pixels
                clean_bright[labels == i] = 255
                num_spots += 1

        return clean_bright, num_spots

    def _analyze_large_spots(self, gray: np.ndarray, mask: np.ndarray,
                             axis_angle: float) -> Tuple[bool, float, bool]:
        """
        Metric 3: Analyze large light/dark spots

        TABLE diamonds often have a large central region (either bright or dark)
        that is well-segmented and symmetric. TILTED diamonds have irregular patterns.

        Returns:
            (has_large_spot, spot_symmetry, is_light_spot)
        """
        # Apply CLAHE to normalize lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(gray)

        # Apply mask
        masked = normalized.copy()
        masked[mask == 0] = 0

        # Threshold to find light regions
        masked_pixels = normalized[mask > 0]
        if len(masked_pixels) == 0:
            return False, 0.0, False

        # Try bright spots first (top 30%)
        light_threshold = np.percentile(masked_pixels, 70)
        light_mask = np.zeros_like(gray, dtype=np.uint8)
        light_mask[(normalized >= light_threshold) & (mask > 0)] = 255

        # Find largest light component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            light_mask, connectivity=8
        )

        large_light_spot = False
        light_spot_mask = None
        light_spot_area = 0

        if num_labels > 1:
            # Find largest component
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) > 0:
                largest_idx = np.argmax(areas) + 1
                light_spot_area = areas[largest_idx - 1]

                # Check if it's "large" (>15% of total area)
                total_area = mask.sum() / 255
                if light_spot_area > 0.15 * total_area:
                    large_light_spot = True
                    light_spot_mask = (labels == largest_idx).astype(np.uint8) * 255

        # Try dark spots (bottom 30%)
        dark_threshold = np.percentile(masked_pixels, 30)
        dark_mask = np.zeros_like(gray, dtype=np.uint8)
        dark_mask[(normalized <= dark_threshold) & (mask > 0)] = 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dark_mask, connectivity=8
        )

        large_dark_spot = False
        dark_spot_mask = None
        dark_spot_area = 0

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) > 0:
                largest_idx = np.argmax(areas) + 1
                dark_spot_area = areas[largest_idx - 1]

                total_area = mask.sum() / 255
                if dark_spot_area > 0.15 * total_area:
                    large_dark_spot = True
                    dark_spot_mask = (labels == largest_idx).astype(np.uint8) * 255

        # Pick the larger spot
        has_large_spot = False
        spot_mask = None
        is_light = True

        if large_light_spot and light_spot_area >= dark_spot_area:
            has_large_spot = True
            spot_mask = light_spot_mask
            is_light = True
        elif large_dark_spot:
            has_large_spot = True
            spot_mask = dark_spot_mask
            is_light = False

        # If we found a large spot, measure its symmetry
        spot_symmetry = 0.0
        if has_large_spot and spot_mask is not None:
            spot_symmetry = self._measure_spot_symmetry(spot_mask, mask, axis_angle)

        return has_large_spot, spot_symmetry, is_light

    def _measure_spot_symmetry(self, spot_mask: np.ndarray, diamond_mask: np.ndarray,
                                axis_angle: float) -> float:
        """Measure how symmetric a spot is along an axis"""
        # Find center
        M = cv2.moments(diamond_mask)
        if M['m00'] == 0:
            return 0.0

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Rotate so axis is vertical
        h, w = spot_mask.shape
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), axis_angle - 90, 1.0)
        rotated_spot = cv2.warpAffine(spot_mask, rotation_matrix, (w, h))
        rotated_mask = cv2.warpAffine(diamond_mask, rotation_matrix, (w, h))

        # Split and compare
        left = rotated_spot[:, :w//2]
        right = rotated_spot[:, w//2:w//2 + left.shape[1]]

        left_mask = rotated_mask[:, :w//2]
        right_mask = rotated_mask[:, w//2:w//2 + left.shape[1]]

        right_mirror = cv2.flip(right, 1)
        right_mask_mirror = cv2.flip(right_mask, 1)

        valid = (left_mask > 127) & (right_mask_mirror > 127)

        if valid.sum() < 10:
            return 0.0

        # Binary overlap
        matching = (left[valid] == right_mirror[valid]).sum()
        total = valid.sum()

        return matching / total if total > 0 else 0.0

    def _calculate_aspect_ratio(self, contour: np.ndarray) -> float:
        """
        Metric 4: Calculate aspect ratio (minor/major axis)

        TABLE diamonds tend to be more square (higher aspect ratio)
        TILTED diamonds tend to be more oval (lower aspect ratio)

        Args:
            contour: Diamond contour

        Returns:
            Aspect ratio [0, 1]
        """
        if len(contour) < 5:
            return 0.0

        ellipse = cv2.fitEllipse(contour)
        (_, (major, minor), _) = ellipse
        aspect_ratio = min(major, minor) / max(major, minor) if max(major, minor) > 0 else 0.0

        return aspect_ratio

    def _make_decision(self, outline_sym: float, reflection_sym: float,
                       has_spot: bool, spot_sym: float, aspect_ratio: float) -> Tuple[str, float]:
        """
        Make final decision based on four metrics

        Decision logic (hierarchical):
        1. Primary: outline_sym + reflection_sym + aspect_ratio
        2. Secondary: spot analysis as tie-breaker

        Returns:
            (orientation, confidence)
        """
        # Strong TABLE indicators (all 3 primary metrics high)
        if (outline_sym >= self.outline_sym_threshold and
            reflection_sym >= self.reflection_sym_threshold and
            aspect_ratio >= self.aspect_ratio_threshold):
            # All three primary metrics strong → TABLE (high confidence)
            confidence = (outline_sym + reflection_sym + aspect_ratio) / 3.0
            return 'table', confidence

        # Strong TILTED indicators (2+ primary metrics weak)
        weak_count = 0
        if outline_sym < 0.60:
            weak_count += 1
        if reflection_sym < 0.55:
            weak_count += 1
        if aspect_ratio < 0.52:
            weak_count += 1

        if weak_count >= 2:
            # Multiple weak metrics → TILTED (high confidence)
            confidence = 1.0 - (outline_sym + reflection_sym + aspect_ratio) / 3.0
            return 'tilted', confidence

        # Mixed signals - use weighted combination
        # Aspect ratio has highest weight (strongest discriminator: 2.15 separation)
        primary_score = 0.25 * outline_sym + 0.35 * reflection_sym + 0.40 * aspect_ratio

        # Use spot as tie-breaker if close to threshold
        if abs(primary_score - 0.60) < 0.05:  # Close to decision boundary
            if has_spot and spot_sym >= self.spot_sym_threshold:
                # Spot pushes toward TABLE
                final_score = 0.7 * primary_score + 0.3 * spot_sym
            else:
                final_score = primary_score
        else:
            final_score = primary_score

        # Final decision
        if final_score >= 0.60:
            return 'table', final_score
        else:
            return 'tilted', 1.0 - final_score


def create_pure_geometric_classifier(outline_threshold: float = 0.70,
                                     reflection_threshold: float = 0.65,
                                     spot_threshold: float = 0.60,
                                     aspect_ratio_threshold: float = 0.58):
    """Factory function to create classifier"""
    return PureGeometricClassifier(
        outline_sym_threshold=outline_threshold,
        reflection_sym_threshold=reflection_threshold,
        spot_sym_threshold=spot_threshold,
        aspect_ratio_threshold=aspect_ratio_threshold
    )
