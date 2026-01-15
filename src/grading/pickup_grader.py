"""
Diamond pickup grading system based on orientation and neighbor proximity
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GradedDiamond:
    """Diamond with pickup grade"""
    roi: any  # DiamondROI
    grade: Optional[float]  # -1 (invalid), None (oval/not pickable), 0-10 (pickup priority)
    nearest_distance: Optional[float]  # Distance to nearest neighbor
    radius: float  # Estimated radius from area


class PickupGrader:
    """
    Grades diamonds for pickup based on:
    1. Orientation: Only circular (table) diamonds are gradable for round diamonds
    2. Proximity: Diamonds too close to neighbors are invalid (-1)
    3. Isolation: More isolated diamonds get higher grades (0-10)

    Supports all diamond types (round, emerald, baguette, heart)
    """

    def __init__(self, invalid_threshold: float = 1.5, check_orientation: bool = True, diamond_type: str = 'round',
                 plate_width_mm: float = 100.0, min_distance_mm: float = 2.0, image_width_px: Optional[int] = None):
        """
        Initialize pickup grader

        Args:
            invalid_threshold: DEPRECATED - kept for compatibility but not used with new edge-to-edge logic
            check_orientation: If True, check orientation for round diamonds.
                             If False, grade all diamonds regardless of orientation.
            diamond_type: Type of diamond ('round', 'baguette', 'emerald', 'heart')
            plate_width_mm: Width of the plate in millimeters (default: 100mm = 10cm)
            min_distance_mm: Minimum edge-to-edge distance in millimeters (default: 2mm)
            image_width_px: Image width in pixels (required for mm-to-pixel conversion)
        """
        self.invalid_threshold = invalid_threshold
        self.check_orientation = check_orientation
        self.diamond_type = diamond_type
        self.plate_width_mm = plate_width_mm
        self.min_distance_mm = min_distance_mm
        self.image_width_px = image_width_px

        # Calculate pixels per mm if image width provided
        if image_width_px is not None:
            self.px_per_mm = image_width_px / plate_width_mm
            self.min_distance_px = min_distance_mm * self.px_per_mm
        else:
            self.px_per_mm = None
            self.min_distance_px = None

    def grade_diamonds(self, diamond_rois: List) -> List[GradedDiamond]:
        """
        Grade all diamonds for pickup

        Args:
            diamond_rois: List of DiamondROI objects

        Returns:
            List of GradedDiamond objects with pickup grades
        """
        graded_diamonds = []

        for i, roi in enumerate(diamond_rois):
            # Calculate radius from area (assuming roughly circular)
            radius = np.sqrt(roi.area / np.pi)

            # Check orientation if enabled and orientation attribute exists
            if self.check_orientation and hasattr(roi, 'orientation'):
                if roi.orientation == 'tilted':
                    # Oval diamonds are not pickable
                    graded = GradedDiamond(
                        roi=roi,
                        grade=None,
                        nearest_distance=None,
                        radius=radius
                    )
                    graded_diamonds.append(graded)
                    continue

            # Calculate edge-to-edge distance to nearest neighbor
            edge_distance, safety_threshold = self._find_nearest_edge_distance(roi, diamond_rois)

            # NEW RULE: If edge-to-edge distance < safety threshold, mark invalid
            # Safety threshold = radius for circular, diameter/6 for non-circular
            if edge_distance < safety_threshold:
                # Too close to another diamond - invalid for pickup
                grade = -1.0
            else:
                # Grade 0-10 based on isolation
                # Normalize: closest valid distance (safety_threshold) = grade 0
                #            far away (e.g., 6*radius or more) = grade 10
                min_valid_distance = safety_threshold
                max_distance = 6 * radius  # Assume 6*radius is "very isolated"

                normalized = (edge_distance - min_valid_distance) / (max_distance - min_valid_distance)
                grade = min(10.0, max(0.0, normalized * 10))

            graded = GradedDiamond(
                roi=roi,
                grade=grade,
                nearest_distance=edge_distance,
                radius=radius
            )
            graded_diamonds.append(graded)

        return graded_diamonds

    def _get_safety_threshold(self, roi) -> float:
        """
        Get safety threshold for a diamond based on plate width and min distance

        If mm-to-pixel conversion is available (image_width_px provided):
            - Use fixed min_distance_mm threshold (default 2mm)
        Otherwise (fallback):
            - For circular diamonds: threshold = radius
            - For non-circular diamonds: threshold = radius/3
        """
        # If we have mm-to-pixel conversion, use fixed threshold
        if self.min_distance_px is not None:
            return self.min_distance_px

        # Fallback to radius-based thresholds
        radius = np.sqrt(roi.area / np.pi)

        # Check if this is a circular diamond based on detected type
        is_circular = False
        if hasattr(roi, 'detected_type') and hasattr(roi, 'orientation'):
            is_circular = (roi.detected_type == 'round' and roi.orientation == 'table')

        if is_circular:
            return radius  # Use full radius for circular diamonds
        else:
            return radius / 3  # Use diameter/6 = radius/3 for non-circular diamonds

    def _find_nearest_edge_distance(self, target_roi, all_rois: List) -> Tuple[float, float]:
        """
        Find edge-to-edge distance to nearest neighbor and required safety threshold

        Returns:
            (edge_distance, safety_threshold):
            - edge_distance: closest edge-to-edge distance to any neighbor
            - safety_threshold: minimum threshold required (min of target and neighbor thresholds)

        IMPORTANT: Checks distance to ALL diamonds (pickable AND non-pickable).
        """
        target_center = np.array(target_roi.center)
        target_radius = np.sqrt(target_roi.area / np.pi)
        target_threshold = self._get_safety_threshold(target_roi)

        min_edge_distance = float('inf')
        required_threshold = target_threshold

        for roi in all_rois:
            # Skip self-comparison
            if roi.id == target_roi.id:
                continue

            # Calculate edge-to-edge distance
            other_center = np.array(roi.center)
            other_radius = np.sqrt(roi.area / np.pi)
            other_threshold = self._get_safety_threshold(roi)

            center_distance = np.linalg.norm(target_center - other_center)
            edge_distance = center_distance - target_radius - other_radius

            if edge_distance < min_edge_distance:
                min_edge_distance = edge_distance
                # Use the smaller threshold between target and neighbor
                required_threshold = min(target_threshold, other_threshold)

        return (min_edge_distance if min_edge_distance != float('inf') else 0.0,
                required_threshold)

    def _find_nearest_neighbor_distance(self, target_roi, all_rois: List) -> float:
        """
        Find distance to nearest neighbor diamond

        IMPORTANT: Checks distance to ALL diamonds (pickable AND non-pickable).
        Even if the nearest neighbor is an oval/tilted diamond (not pickable),
        its proximity still affects the grading of the target diamond.
        """
        target_center = np.array(target_roi.center)
        min_distance = float('inf')

        for roi in all_rois:
            # Skip self-comparison
            if roi.id == target_roi.id:
                continue

            # Check distance to ALL other diamonds (pickable or not)
            other_center = np.array(roi.center)
            distance = np.linalg.norm(target_center - other_center)

            if distance < min_distance:
                min_distance = distance

        return min_distance if min_distance != float('inf') else 0.0

    def visualize_grades(self, image: np.ndarray, graded_diamonds: List[GradedDiamond],
                        save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize pickup grades on image

        Color coding:
        - Green (bright): High priority (grade 7-10)
        - Yellow: Medium priority (grade 4-7)
        - Orange: Low priority (grade 0-4)
        - Red: Invalid for pickup (grade -1)
        - No marking: Oval/tilted (not pickable)

        Args:
            image: Input image
            graded_diamonds: List of graded diamonds
            save_path: Optional path to save visualization

        Returns:
            Visualized image
        """
        vis_image = image.copy()

        for gd in graded_diamonds:
            # Skip oval diamonds (no marking)
            if gd.grade is None:
                continue

            # Determine color based on grade
            if gd.grade == -1:
                color = (0, 0, 255)  # Red - invalid
                label = "INVALID"
            elif gd.grade >= 7:
                color = (0, 255, 0)  # Green - high priority
                label = f"P{int(gd.grade)}"
            elif gd.grade >= 4:
                color = (0, 255, 255)  # Yellow - medium priority
                label = f"P{int(gd.grade)}"
            else:
                color = (0, 165, 255)  # Orange - low priority
                label = f"P{int(gd.grade)}"

            cx, cy = int(gd.roi.center[0]), int(gd.roi.center[1])

            # Check if this is a circular diamond based on detected type
            is_circular = (hasattr(gd.roi, 'detected_type') and
                          gd.roi.detected_type == 'round' and
                          hasattr(gd.roi, 'orientation') and
                          gd.roi.orientation == 'table')

            if is_circular:
                # Draw filled circle at center for circular diamonds
                cv2.circle(vis_image, (cx, cy), int(gd.radius * 0.3), color, -1)
                # Draw circular outline
                cv2.circle(vis_image, (cx, cy), int(gd.radius), color, 2)
            else:
                # Draw contour outline for non-circular diamonds (baguettes, etc.)
                cv2.drawContours(vis_image, [gd.roi.contour], -1, color, 2)
                # Draw small filled circle at center
                cv2.circle(vis_image, (cx, cy), 5, color, -1)

            # Draw label
            cv2.putText(vis_image, label, (cx - 20, cy - int(gd.radius) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis_image, label, (cx - 20, cy - int(gd.radius) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw ID
            cv2.putText(vis_image, str(gd.roi.id), (cx - 8, cy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

        # Add legend
        legend_y = 30
        cv2.putText(vis_image, "Pickup Priority:", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 30
        cv2.putText(vis_image, "High (7-10)", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        legend_y += 25
        cv2.putText(vis_image, "Medium (4-7)", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        legend_y += 25
        cv2.putText(vis_image, "Low (0-4)", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        legend_y += 25
        cv2.putText(vis_image, "Invalid", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if save_path:
            cv2.imwrite(save_path, vis_image)

        return vis_image

    def visualize_pickup_order(self, image: np.ndarray, graded_diamonds: List[GradedDiamond],
                              save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize pickup status (simplified view for operators)

        Simplified color coding (no priority numbers displayed, saved to JSON only):
        - Green: Pickable diamonds (on table, passes proximity check)
        - Orange: On table but too close to neighbors (fails proximity check)
        - Red: Tilted/not on table (not pickable)

        Args:
            image: Input image
            graded_diamonds: List of graded diamonds
            save_path: Optional path to save visualization

        Returns:
            Visualized image
        """
        vis_image = image.copy()

        # Draw TILTED diamonds (grade is None) - RED
        tilted_diamonds = [gd for gd in graded_diamonds if gd.grade is None]
        for gd in tilted_diamonds:
            color = (0, 0, 255)  # Red for tilted/not on table
            cx, cy = int(gd.roi.center[0]), int(gd.roi.center[1])
            radius = np.sqrt(gd.roi.area / np.pi)

            # Check if this is a round diamond
            is_round = (hasattr(gd.roi, 'detected_type') and gd.roi.detected_type == 'round')

            if is_round:
                cv2.circle(vis_image, (cx, cy), int(radius), color, 2)
            else:
                cv2.drawContours(vis_image, [gd.roi.contour], -1, color, 2)

        # Draw TABLE diamonds that are too close (grade == -1) - ORANGE
        proximity_failed = [gd for gd in graded_diamonds if gd.grade == -1]
        for gd in proximity_failed:
            color = (0, 165, 255)  # Orange for proximity failed
            cx, cy = int(gd.roi.center[0]), int(gd.roi.center[1])

            # Check if this is a circular diamond based on detected type
            is_circular = (hasattr(gd.roi, 'detected_type') and
                          gd.roi.detected_type == 'round' and
                          hasattr(gd.roi, 'orientation') and
                          gd.roi.orientation == 'table')

            if is_circular:
                cv2.circle(vis_image, (cx, cy), int(gd.radius), color, 2)
            else:
                cv2.drawContours(vis_image, [gd.roi.contour], -1, color, 2)

        # Draw PICKABLE diamonds (grade >= 0) - GREEN
        pickable = [gd for gd in graded_diamonds if gd.grade is not None and gd.grade >= 0]
        for gd in pickable:
            color = (0, 255, 0)  # Green for pickable
            cx, cy = int(gd.roi.center[0]), int(gd.roi.center[1])

            # Check if this is a circular diamond based on detected type
            is_circular = (hasattr(gd.roi, 'detected_type') and
                          gd.roi.detected_type == 'round' and
                          hasattr(gd.roi, 'orientation') and
                          gd.roi.orientation == 'table')

            if is_circular:
                cv2.circle(vis_image, (cx, cy), int(gd.radius), color, 2)
            else:
                cv2.drawContours(vis_image, [gd.roi.contour], -1, color, 2)

        # Add simplified legend
        legend_y = 30
        cv2.putText(vis_image, "Diamond Status:", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 30
        cv2.putText(vis_image, "Green: Pickable", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        legend_y += 25
        cv2.putText(vis_image, "Orange: Too Close", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        legend_y += 25
        cv2.putText(vis_image, "Red: Tilted", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if save_path:
            cv2.imwrite(save_path, vis_image)

        return vis_image


def create_pickup_grader(invalid_threshold: float = 0.5) -> PickupGrader:
    """Factory function to create pickup grader"""
    return PickupGrader(invalid_threshold)
