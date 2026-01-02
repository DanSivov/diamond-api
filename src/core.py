"""
Core Diamond Classification Engine
Handles detection, classification, and grading
"""
import cv2
import numpy as np
import joblib
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

from preprocessing import SAMDiamondDetector, DiamondROI
from classification import PureGeometricClassifier
from grading import PickupGrader


@dataclass
class ClassificationResult:
    """Single diamond classification result"""
    roi_id: int
    diamond_type: str  # Auto-detected: 'round', 'emerald', 'other'
    orientation: str  # 'table' or 'tilted'
    confidence: float
    features: Dict[str, float]
    bounding_box: Tuple[int, int, int, int]
    center: Tuple[float, float]
    area: float


@dataclass
class ImageResult:
    """Complete image classification result"""
    image_name: str
    total_diamonds: int
    table_count: int
    tilted_count: int
    pickable_count: int
    invalid_count: int
    average_grade: float
    classifications: List[ClassificationResult]
    model_name: str = 'RandomForest'
    model_accuracy: str = '95.6%'

    def to_dict(self):
        """Convert to dictionary for JSON export"""
        result = asdict(self)
        result['classifications'] = [asdict(c) for c in self.classifications]
        return result


class DiamondClassifier:
    """
    Core diamond classification engine

    Auto-detects diamond type and classifies orientation using ML
    """

    def __init__(self, model_path: str, feature_names_path: str):
        """
        Initialize classifier

        Args:
            model_path: Path to trained ML model (.pkl)
            feature_names_path: Path to feature names JSON
        """
        self.model = joblib.load(model_path)
        with open(feature_names_path, 'r') as f:
            self.feature_names = json.load(f)

        self.feature_extractor = PureGeometricClassifier()
        self.detector = None
        self.grader = None

    def _initialize_detector(self, image_shape: Tuple[int, int]):
        """Initialize detector with adaptive area thresholds"""
        h, w = image_shape
        image_pixels = h * w
        base_pixels = 1944 * 2592
        scale_factor = np.sqrt(image_pixels / base_pixels)
        base_area_scale = scale_factor ** 2
        min_area = max(30, int(200 * base_area_scale))
        max_area = max(1000, int(20000 * base_area_scale))

        self.detector = SAMDiamondDetector(
            min_area=min_area,
            max_area=max_area,
            padding=10,
            merge_overlapping=False
        )

    def _initialize_grader(self, image_width: int):
        """Initialize grader with image-specific parameters"""
        self.grader = PickupGrader(
            check_orientation=True,
            image_width_px=image_width
        )

    def classify_image(self, image: np.ndarray, image_name: str = "image") -> ImageResult:
        """
        Classify all diamonds in an image

        Args:
            image: Input BGR image
            image_name: Name of the image (for result tracking)

        Returns:
            ImageResult with all classifications
        """
        h, w = image.shape[:2]

        # Initialize detector and grader if needed
        if self.detector is None:
            self._initialize_detector((h, w))
        if self.grader is None:
            self._initialize_grader(w)

        # Detect diamonds
        diamond_rois = self.detector.detect(image)

        if len(diamond_rois) == 0:
            return ImageResult(
                image_name=image_name,
                total_diamonds=0,
                table_count=0,
                tilted_count=0,
                pickable_count=0,
                invalid_count=0,
                average_grade=0.0,
                classifications=[]
            )

        # Classify each diamond
        table_count = 0
        tilted_count = 0
        classifications = []

        for roi in diamond_rois:
            # Extract geometric features
            result = self.feature_extractor.analyze(roi.contour, roi.mask, roi.roi_image)

            # AUTO-DETECT diamond type (no user input required)
            diamond_type = roi.detected_type  # 'round', 'emerald', or 'other'

            # Prepare features for ML model
            features = [
                result.outline_symmetry_score,
                result.reflection_symmetry_score,
                result.aspect_ratio,
                result.spot_symmetry_score,
                1 if result.has_large_central_spot else 0,
                result.num_reflection_spots,
                1 if diamond_type == 'emerald' else 0,
                1 if diamond_type == 'other' else 0
            ]

            # Predict orientation
            X = np.array([features])
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]

            orientation = 'table' if prediction == 1 else 'tilted'
            confidence = probability[prediction]

            # Update ROI with classification
            roi.orientation = orientation
            roi.ml_confidence = confidence

            if orientation == 'table':
                table_count += 1
            else:
                tilted_count += 1

            # Store classification result
            classifications.append(ClassificationResult(
                roi_id=roi.id,
                diamond_type=diamond_type,
                orientation=orientation,
                confidence=float(confidence),
                features={
                    'outline_sym': float(result.outline_symmetry_score),
                    'reflection_sym': float(result.reflection_symmetry_score),
                    'aspect_ratio': float(result.aspect_ratio),
                    'spot_sym': float(result.spot_symmetry_score),
                    'has_spot': bool(result.has_large_central_spot),
                    'num_reflections': int(result.num_reflection_spots)
                },
                bounding_box=roi.bounding_box,
                center=roi.center,
                area=float(roi.area)
            ))

        # Grade diamonds for pickup
        graded_diamonds = self.grader.grade_diamonds(diamond_rois)
        pickable = [gd for gd in graded_diamonds if gd.grade is not None and gd.grade >= 0]
        invalid = [gd for gd in graded_diamonds if gd.grade == -1]
        avg_grade = sum(gd.grade for gd in pickable) / len(pickable) if len(pickable) > 0 else 0.0

        # Store graded diamonds for visualization
        self._last_graded_diamonds = graded_diamonds
        self._last_image = image

        return ImageResult(
            image_name=image_name,
            total_diamonds=len(diamond_rois),
            table_count=table_count,
            tilted_count=tilted_count,
            pickable_count=len(pickable),
            invalid_count=len(invalid),
            average_grade=float(avg_grade),
            classifications=classifications
        )

    def get_visualization(self) -> Optional[np.ndarray]:
        """
        Get visualization of last classified image

        Returns:
            Graded image with pickup order overlay
        """
        if not hasattr(self, '_last_graded_diamonds') or not hasattr(self, '_last_image'):
            return None

        return self.grader.visualize_pickup_order(
            self._last_image,
            self._last_graded_diamonds
        )

    def get_roi_image(self, roi_id: int) -> Optional[np.ndarray]:
        """
        Get ROI image for verification

        Args:
            roi_id: ROI index

        Returns:
            ROI image or None if not found
        """
        if not hasattr(self, '_last_graded_diamonds'):
            return None

        for gd in self._last_graded_diamonds:
            if gd.roi.id == roi_id:
                return gd.roi.roi_image

        return None
