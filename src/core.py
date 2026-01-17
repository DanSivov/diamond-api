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
    model_name: str = 'StackedRandomForest'
    model_accuracy: str = '92.5%'

    def to_dict(self):
        """Convert to dictionary for JSON export"""
        result = asdict(self)
        result['classifications'] = [asdict(c) for c in self.classifications]
        return result


class DiamondClassifier:
    """
    Core diamond classification engine

    Auto-detects diamond type and classifies orientation using ML.
    Uses a stacked model architecture for improved accuracy (92.5%):
    - Base model: RandomForest trained on geometric features
    - Stacked model: Uses base model predictions + extended features
    """

    def __init__(self, model_path: str, feature_names_path: str):
        """
        Initialize classifier

        Args:
            model_path: Path to trained ML model (.pkl) - used as base model
            feature_names_path: Path to feature names JSON
        """
        # Load base model (original RandomForest)
        self.base_model = joblib.load(model_path)
        with open(feature_names_path, 'r') as f:
            self.feature_names = json.load(f)

        # Load stacked model if available (improved accuracy)
        model_dir = Path(model_path).parent
        stacked_model_path = model_dir / 'orientation_classifier_improved.pkl'
        if stacked_model_path.exists():
            self.stacked_model = joblib.load(stacked_model_path)
            self.use_stacked_model = True
            print(f"Loaded stacked model (92.5% accuracy)")
        else:
            self.stacked_model = None
            self.use_stacked_model = False
            print(f"Stacked model not found, using base model only")

        # For backwards compatibility
        self.model = self.base_model

        self.feature_extractor = PureGeometricClassifier()
        self.detector = None
        self.grader = None

    def _extract_contour_features(self, mask: np.ndarray) -> dict:
        """
        Extract circularity and solidity from mask for stacked model

        Args:
            mask: Binary mask of the diamond

        Returns:
            Dict with circularity and solidity
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return {'circularity': 0.5, 'solidity': 0.5}

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 10:
            return {'circularity': 0.5, 'solidity': 0.5}

        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        circularity = min(circularity, 1.0)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        return {
            'circularity': circularity,
            'solidity': solidity
        }

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
        print(f"Detecting diamonds in image of size {h}x{w}")
        diamond_rois = self.detector.detect(image)
        print(f"Detection complete: found {len(diamond_rois)} diamonds")

        if len(diamond_rois) == 0:
            print("No diamonds detected, returning empty result")
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
            diamond_type = roi.detected_type  # 'round' or 'other'

            # For ROUND diamonds, use geometric features
            # Round brilliant cuts have distinct patterns
            if diamond_type == 'round':
                # Round diamonds on table: Large central bright spot + high symmetry
                # Round diamonds tilted: No central spot OR low symmetry

                # Use available geometric features
                has_bright_center = result.has_large_central_spot and result.spot_is_light
                high_symmetry = result.outline_symmetry_score > 0.65 or result.reflection_symmetry_score > 0.65

                # Simple classification for round diamonds
                # TABLE: Has bright center OR high symmetry
                # TILTED: No bright center AND low symmetry
                if has_bright_center or high_symmetry:
                    orientation = 'table'
                    # Use max of symmetry scores as confidence
                    confidence = max(result.outline_symmetry_score, result.reflection_symmetry_score)
                else:
                    orientation = 'tilted'
                    confidence = 1.0 - max(result.outline_symmetry_score, result.reflection_symmetry_score)

            else:
                # For NON-ROUND diamonds, use ML model trained on rectangular diamonds
                # Extract contour features for stacked model
                contour_features = self._extract_contour_features(roi.mask)
                circularity = contour_features['circularity']
                solidity = contour_features['solidity']

                # Base features for original model
                base_features = [
                    result.outline_symmetry_score,
                    result.reflection_symmetry_score,
                    result.aspect_ratio,
                    result.spot_symmetry_score,
                    1 if result.has_large_central_spot else 0,
                    result.num_reflection_spots,
                    0,  # type_emerald
                    1   # type_other
                ]

                if self.use_stacked_model and self.stacked_model is not None:
                    # Get base model prediction probability
                    X_base = np.array([base_features])
                    base_prob = self.base_model.predict_proba(X_base)[0][1]  # P(table)

                    # Calculate derived features
                    outline_sym = result.outline_symmetry_score
                    reflection_sym = result.reflection_symmetry_score
                    aspect_ratio = result.aspect_ratio
                    sym_product = outline_sym * reflection_sym
                    sym_diff = abs(outline_sym - reflection_sym)
                    aspect_circ_ratio = aspect_ratio / (circularity + 0.01)

                    # Extended features for stacked model (14 features + base_model_prob)
                    # Order: outline_sym, reflection_sym, aspect_ratio, spot_sym, has_spot,
                    #        num_reflections, circularity, image_is_round, type_emerald, type_other,
                    #        sym_product, sym_diff, aspect_circ_ratio, solidity, base_model_prob
                    extended_features = [
                        outline_sym,
                        reflection_sym,
                        aspect_ratio,
                        result.spot_symmetry_score,
                        1 if result.has_large_central_spot else 0,
                        result.num_reflection_spots,
                        circularity,
                        0,  # image_is_round (not applicable for single ROI)
                        0,  # type_emerald
                        1,  # type_other
                        sym_product,
                        sym_diff,
                        aspect_circ_ratio,
                        solidity,
                        base_prob  # Base model prediction
                    ]

                    X_stacked = np.array([extended_features])
                    prediction = self.stacked_model.predict(X_stacked)[0]
                    probability = self.stacked_model.predict_proba(X_stacked)[0]
                else:
                    # Fallback to base model only
                    X_base = np.array([base_features])
                    prediction = self.base_model.predict(X_base)[0]
                    probability = self.base_model.predict_proba(X_base)[0]

                orientation = 'table' if prediction == 1 else 'tilted'
                confidence = probability[prediction]

            # Update ROI with classification
            roi.orientation = orientation
            roi.ml_confidence = confidence

            if orientation == 'table':
                table_count += 1
            else:
                tilted_count += 1

            # Extract circularity for non-round diamonds (for features output)
            if diamond_type != 'round':
                cf = self._extract_contour_features(roi.mask)
                circ = cf['circularity']
            else:
                circ = 0.0

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
                    'num_reflections': int(result.num_reflection_spots),
                    'circularity': float(circ)
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
