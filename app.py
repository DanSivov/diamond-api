from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tempfile
import os
import base64
from pathlib import Path
import sys
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

app = Flask(__name__)
CORS(app)

# Lazy loading to reduce memory footprint
classifier = None

def get_classifier():
    """Lazy load classifier only when needed"""
    global classifier
    if classifier is None:
        # Import only when needed to avoid loading all dependencies at startup
        from core import DiamondClassifier

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

        # Get ROI images for visualization
        roi_images = []
        if hasattr(clf, '_last_graded_diamonds'):
            for gd in clf._last_graded_diamonds:
                roi_img = gd.roi.roi_image
                _, buffer = cv2.imencode('.png', roi_img)
                roi_b64 = base64.b64encode(buffer).decode('utf-8')
                roi_images.append(roi_b64)

        # Encode full image for context view
        _, full_buffer = cv2.imencode('.png', image)
        full_image_b64 = base64.b64encode(full_buffer).decode('utf-8')

        # Get graded visualization image
        graded_image_b64 = None
        visualization = clf.get_visualization()
        if visualization is not None:
            _, graded_buffer = cv2.imencode('.png', visualization)
            graded_image_b64 = base64.b64encode(graded_buffer).decode('utf-8')

        # Convert to JSON-serializable format
        response = {
            'image_name': result.image_name,
            'total_diamonds': result.total_diamonds,
            'table_count': result.table_count,
            'tilted_count': result.tilted_count,
            'pickable_count': result.pickable_count,
            'invalid_count': result.invalid_count,
            'average_grade': result.average_grade,
            'full_image_base64': full_image_b64,
            'graded_image_base64': graded_image_b64,
            'classifications': [
                {
                    'roi_id': c.roi_id,
                    'diamond_type': c.diamond_type,
                    'orientation': c.orientation,
                    'confidence': c.confidence,
                    'features': c.features,
                    'bounding_box': list(c.bounding_box),
                    'center': list(c.center),
                    'area': c.area,
                    'roi_image_base64': roi_images[i] if i < len(roi_images) else None
                }
                for i, c in enumerate(result.classifications)
            ]
        }

        # Clear memory
        del image, file_bytes, full_buffer, roi_images
        if visualization is not None:
            del visualization, graded_buffer
        gc.collect()

        return jsonify(response)

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        gc.collect()
        return jsonify({'error': str(e)}), 500

@app.route('/classify-batch', methods=['POST'])
def classify_batch():
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    files = request.files.getlist('images')
    results = []
    clf = get_classifier()

    for idx, file in enumerate(files):
        print(f"Processing image {idx + 1}/{len(files)}: {file.filename}")

        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                print(f"Failed to decode image: {file.filename}")
                results.append({
                    'image_name': file.filename,
                    'error': 'Failed to decode image'
                })
                continue

            result = clf.classify_image(image, file.filename)

            # Get ROI images for visualization
            roi_images = []
            if hasattr(clf, '_last_graded_diamonds'):
                for gd in clf._last_graded_diamonds:
                    roi_img = gd.roi.roi_image
                    _, buffer = cv2.imencode('.png', roi_img)
                    roi_b64 = base64.b64encode(buffer).decode('utf-8')
                    roi_images.append(roi_b64)

            # Encode full image for context view
            _, full_buffer = cv2.imencode('.png', image)
            full_image_b64 = base64.b64encode(full_buffer).decode('utf-8')

            # Get graded visualization image
            graded_image_b64 = None
            visualization = clf.get_visualization()
            if visualization is not None:
                _, graded_buffer = cv2.imencode('.png', visualization)
                graded_image_b64 = base64.b64encode(graded_buffer).decode('utf-8')

            results.append({
                'image_name': result.image_name,
                'total_diamonds': result.total_diamonds,
                'table_count': result.table_count,
                'tilted_count': result.tilted_count,
                'pickable_count': result.pickable_count,
                'invalid_count': result.invalid_count,
                'average_grade': result.average_grade,
                'full_image_base64': full_image_b64,
                'graded_image_base64': graded_image_b64,
                'classifications': [
                    {
                        'roi_id': c.roi_id,
                        'diamond_type': c.diamond_type,
                        'orientation': c.orientation,
                        'confidence': c.confidence,
                        'features': c.features,
                        'bounding_box': list(c.bounding_box),
                        'center': list(c.center),
                        'area': c.area,
                        'roi_image_base64': roi_images[i] if i < len(roi_images) else None
                    }
                    for i, c in enumerate(result.classifications)
                ]
            })

            # Clear memory after each image
            del image, file_bytes, full_buffer
            if visualization is not None:
                del visualization, graded_buffer
            del roi_images
            gc.collect()

            print(f"Successfully processed {file.filename}: {result.total_diamonds} diamonds found")

        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                'image_name': file.filename,
                'error': str(e)
            })
            # Clear memory on error too
            gc.collect()

    print(f"Batch processing complete: {len(results)} results")
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
