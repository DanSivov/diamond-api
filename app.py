from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core import DiamondClassifier

app = Flask(__name__)
CORS(app)

# Lazy loading to reduce memory footprint
classifier = None

def get_classifier():
    """Lazy load classifier only when needed"""
    global classifier
    if classifier is None:
        model_path = Path(__file__).parent / 'models' / 'ml_classifier' / 'best_model_randomforest.pkl'
        features_path = Path(__file__).parent / 'models' / 'ml_classifier' / 'feature_names.json'

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")

        print(f"Loading classifier from {model_path}")
        classifier = DiamondClassifier(str(model_path), str(features_path))
        print("Classifier loaded successfully")

    return classifier

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Classify
    try:
        print(f"Processing image: {file.filename}, shape: {image.shape}")
        clf = get_classifier()
        result = clf.classify_image(image, file.filename)
        print(f"Result: {result.total_diamonds} diamonds found")

        # Convert to JSON-serializable format
        response = {
            'image_name': result.image_name,
            'total_diamonds': result.total_diamonds,
            'table_count': result.table_count,
            'tilted_count': result.tilted_count,
            'pickable_count': result.pickable_count,
            'invalid_count': result.invalid_count,
            'average_grade': result.average_grade,
            'classifications': [
                {
                    'roi_id': c.roi_id,
                    'diamond_type': c.diamond_type,
                    'orientation': c.orientation,
                    'confidence': c.confidence,
                    'features': c.features,
                    'bounding_box': list(c.bounding_box),
                    'center': list(c.center),
                    'area': c.area
                }
                for c in result.classifications
            ]
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classify-batch', methods=['POST'])
def classify_batch():
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    files = request.files.getlist('images')
    results = []
    clf = get_classifier()

    for file in files:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            continue

        try:
            result = clf.classify_image(image, file.filename)

            results.append({
                'image_name': result.image_name,
                'total_diamonds': result.total_diamonds,
                'table_count': result.table_count,
                'tilted_count': result.tilted_count,
                'pickable_count': result.pickable_count,
                'invalid_count': result.invalid_count,
                'average_grade': result.average_grade,
                'classifications': [
                    {
                        'roi_id': c.roi_id,
                        'diamond_type': c.diamond_type,
                        'orientation': c.orientation,
                        'confidence': c.confidence,
                        'features': c.features,
                        'bounding_box': list(c.bounding_box),
                        'center': list(c.center),
                        'area': c.area
                    }
                    for c in result.classifications
                ]
            })
        except Exception as e:
            results.append({
                'image_name': file.filename,
                'error': str(e)
            })

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
