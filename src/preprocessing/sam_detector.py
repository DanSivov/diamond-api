"""
Diamond Detection using FastSAM
Production version - simplified for ML-based classification
"""
import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from ultralytics import FastSAM


@dataclass
class DiamondROI:
    """Diamond Region of Interest"""
    contour: np.ndarray
    bounding_box: Tuple[int, int, int, int]
    center: Tuple[float, float]
    area: float
    perimeter: float
    roi_image: np.ndarray
    mask: np.ndarray
    id: int
    ellipse: Tuple
    major_axis: float
    minor_axis: float
    aspect_ratio: float
    orientation: str = 'unknown'
    detected_type: str = 'other'


class SAMDiamondDetector:
    """Diamond detector using FastSAM"""

    def __init__(self,
                 min_area: int = 200,
                 max_area: int = 50000,
                 padding: int = 10,
                 merge_overlapping: bool = False,
                 overlap_threshold: float = 0.25):
        """
        Initialize FastSAM diamond detector

        Args:
            min_area: Minimum diamond area in pixels
            max_area: Maximum diamond area in pixels
            padding: Padding around ROI bounding box
            merge_overlapping: If True, merge overlapping/nested masks
            overlap_threshold: IoU threshold for merging masks
        """
        self.min_area = min_area
        self.max_area = max_area
        self.padding = padding
        self.merge_overlapping = merge_overlapping
        self.overlap_threshold = overlap_threshold
        self.model = None

    def load_model(self):
        """Load FastSAM model (lazy loading)"""
        if self.model is None:
            self.model = FastSAM('FastSAM-x.pt')

    def _calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate Intersection over Union between two masks"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0

    def _is_valid_diamond_shape(self, contour: np.ndarray, mask: np.ndarray) -> bool:
        """
        Validate that contour represents a real diamond

        Checks:
        1. Solidity (ratio of contour area to convex hull area) - should be >0.7
        2. Number of holes - should be 0
        3. Extent (ratio of contour area to bounding box area) - should be >0.5
        """
        # Solidity check
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)

        if hull_area > 0:
            solidity = contour_area / hull_area
            if solidity < 0.70:
                return False
        else:
            return False

        # Check for holes
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 2:
            return False

        # Extent check
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        if bbox_area > 0:
            extent = contour_area / bbox_area
            if extent < 0.50:
                return False

        # Circularity check
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = (4 * np.pi * contour_area) / (perimeter ** 2)
            if circularity < 0.30:
                return False

        return True

    def _merge_masks(self, masks_list: List[np.ndarray]) -> List[np.ndarray]:
        """Merge overlapping and nested masks"""
        if not self.merge_overlapping or len(masks_list) == 0:
            return masks_list

        binary_masks = [(mask > 0.5).astype(np.uint8) for mask in masks_list]
        merged = []
        used = [False] * len(binary_masks)

        for i in range(len(binary_masks)):
            if used[i]:
                continue

            current_merged = binary_masks[i].copy()
            used[i] = True
            current_area = current_merged.sum()

            changed = True
            while changed:
                changed = False
                for j in range(len(binary_masks)):
                    if used[j]:
                        continue

                    mask_j = binary_masks[j]
                    area_j = mask_j.sum()

                    iou = self._calculate_iou(current_merged, mask_j)
                    intersection = np.logical_and(current_merged, mask_j).sum()
                    containment_j = intersection / area_j if area_j > 0 else 0.0
                    containment_current = intersection / current_area if current_area > 0 else 0.0
                    max_containment = max(containment_j, containment_current)

                    size_ratio = max(current_area, area_j) / min(current_area, area_j) if min(current_area, area_j) > 0 else 1.0

                    should_merge = (max_containment > 0.5) or (iou > self.overlap_threshold and size_ratio > 1.5) or (iou > 0.4)

                    if should_merge:
                        current_merged = np.logical_or(current_merged, mask_j).astype(np.uint8)
                        current_area = current_merged.sum()
                        used[j] = True
                        changed = True

            merged.append(current_merged.astype(np.float32))

        return merged

    def detect(self, image: np.ndarray) -> List[DiamondROI]:
        """
        Detect diamonds in image using FastSAM

        Args:
            image: Input BGR image

        Returns:
            List of DiamondROI objects
        """
        self.load_model()

        results = self.model(
            image,
            device='cpu',
            retina_masks=True,
            imgsz=1536,
            conf=0.15,
            iou=0.9,
            verbose=False
        )

        diamond_rois = []

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            masks = self._merge_masks(list(masks))

            roi_id = 0

            for mask in masks:
                mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
                area = np.sum(mask_uint8 > 0)

                if area < self.min_area or area > self.max_area:
                    continue

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)

                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                if not contours:
                    continue

                contour = max(contours, key=cv2.contourArea)

                if len(contour) < 5:
                    continue

                if not self._is_valid_diamond_shape(contour, mask_uint8):
                    continue

                perimeter = cv2.arcLength(contour, True)
                moments = cv2.moments(contour)

                if moments['m00'] != 0:
                    cx = moments['m10'] / moments['m00']
                    cy = moments['m01'] / moments['m00']
                else:
                    continue

                ellipse = cv2.fitEllipse(contour)
                (_, axes, _) = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)
                aspect_ratio = minor_axis / major_axis if major_axis > 0 else 0

                # Detect diamond type based on aspect ratio
                if aspect_ratio > 0.85:
                    detected_type = 'round'
                elif aspect_ratio < 0.70:
                    detected_type = 'emerald'
                else:
                    detected_type = 'other'

                x, y, w, h = cv2.boundingRect(contour)
                x_pad = max(0, x - self.padding)
                y_pad = max(0, y - self.padding)
                x_end = min(image.shape[1], x + w + self.padding)
                y_end = min(image.shape[0], y + h + self.padding)

                roi_image = image[y_pad:y_end, x_pad:x_end].copy()
                roi_mask = mask_uint8[y_pad:y_end, x_pad:x_end].copy()

                diamond = DiamondROI(
                    contour=contour,
                    bounding_box=(x, y, w, h),
                    center=(cx, cy),
                    area=area,
                    perimeter=perimeter,
                    roi_image=roi_image,
                    mask=roi_mask,
                    id=roi_id,
                    ellipse=ellipse,
                    major_axis=major_axis,
                    minor_axis=minor_axis,
                    aspect_ratio=aspect_ratio,
                    orientation='unknown',
                    detected_type=detected_type
                )

                diamond_rois.append(diamond)
                roi_id += 1

        diamond_rois = self._remove_nested_rois(diamond_rois)
        return diamond_rois

    def _remove_nested_rois(self, diamond_rois: List) -> List:
        """Remove nested ROIs where one diamond is contained inside another"""
        if len(diamond_rois) <= 1:
            return diamond_rois

        to_remove = set()

        for i, roi_i in enumerate(diamond_rois):
            if i in to_remove:
                continue

            for j, roi_j in enumerate(diamond_rois):
                if i == j or j in to_remove:
                    continue

                if roi_i.area >= roi_j.area:
                    continue

                cx_i, cy_i = roi_i.center
                center_inside = cv2.pointPolygonTest(roi_j.contour, (float(cx_i), float(cy_i)), False) >= 0

                if center_inside:
                    x_i, y_i, w_i, h_i = roi_i.bounding_box
                    x_j, y_j, w_j, h_j = roi_j.bounding_box

                    bbox_contained = (
                        x_i >= x_j and y_i >= y_j and
                        (x_i + w_i) <= (x_j + w_j) and
                        (y_i + h_i) <= (y_j + h_j)
                    )

                    if bbox_contained:
                        to_remove.add(i)
                        break

        filtered_rois = [roi for idx, roi in enumerate(diamond_rois) if idx not in to_remove]

        for new_id, roi in enumerate(filtered_rois):
            roi.id = new_id

        return filtered_rois
