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

            # For batch processing, skip heavy image encoding to reduce response size
            # Only include classification data, not images
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

            # Clear memory after each image
            del image, file_bytes
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

# ============================================================================
# Async Job Management Endpoints
# ============================================================================

from models import get_session, init_db, Job, Image, ROI, Verification
from tasks import process_batch_job
from datetime import datetime

# Initialize database on startup
try:
    init_db()
    print("Database initialized")
except Exception as e:
    print(f"Database initialization warning: {e}")

@app.route('/jobs/create', methods=['POST'])
def create_job():
    """Create a new batch processing job"""
    try:
        data = request.get_json()

        if not data or 'files' not in data:
            return jsonify({'error': 'No files provided'}), 400

        files_data = data['files']
        if not files_data or len(files_data) == 0:
            return jsonify({'error': 'Empty file list'}), 400

        session = get_session()

        # Create job record
        job = Job(
            total_images=len(files_data),
            processed_images=0,
            status='pending'
        )
        session.add(job)
        session.commit()

        job_id = job.id
        session.close()

        # Queue async task
        process_batch_job.delay(job_id, files_data)

        print(f"Created job {job_id} with {len(files_data)} images")

        return jsonify({
            'job_id': job_id,
            'status': 'pending',
            'total_images': len(files_data)
        })

    except Exception as e:
        print(f"Error creating job: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/jobs/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    """Get job status and progress"""
    try:
        session = get_session()
        job = session.query(Job).filter_by(id=job_id).first()

        if not job:
            session.close()
            return jsonify({'error': 'Job not found'}), 404

        response = job.to_dict()
        session.close()

        return jsonify(response)

    except Exception as e:
        print(f"Error getting job status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/jobs/<job_id>/images', methods=['GET'])
def get_job_images(job_id):
    """Get all images for a job"""
    try:
        session = get_session()
        job = session.query(Job).filter_by(id=job_id).first()

        if not job:
            session.close()
            return jsonify({'error': 'Job not found'}), 404

        images = session.query(Image).filter_by(job_id=job_id).all()
        response = {
            'job': job.to_dict(),
            'images': [img.to_dict() for img in images]
        }
        session.close()

        return jsonify(response)

    except Exception as e:
        print(f"Error getting job images: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/images/<image_id>/rois', methods=['GET'])
def get_image_rois(image_id):
    """Get all ROIs for an image"""
    try:
        session = get_session()
        image = session.query(Image).filter_by(id=image_id).first()

        if not image:
            session.close()
            return jsonify({'error': 'Image not found'}), 404

        rois = session.query(ROI).filter_by(image_id=image_id).order_by(ROI.roi_index).all()
        response = {
            'image': image.to_dict(),
            'rois': [roi.to_dict(include_verifications=True) for roi in rois]
        }
        session.close()

        return jsonify(response)

    except Exception as e:
        print(f"Error getting image ROIs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rois/<roi_id>/verify', methods=['POST'])
def verify_roi(roi_id):
    """Submit verification for an ROI"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        required_fields = ['user_email', 'is_correct']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        session = get_session()
        roi = session.query(ROI).filter_by(id=roi_id).first()

        if not roi:
            session.close()
            return jsonify({'error': 'ROI not found'}), 404

        # Create verification record
        verification = Verification(
            roi_id=roi_id,
            user_email=data['user_email'],
            is_correct=data['is_correct'],
            corrected_type=data.get('corrected_type'),
            corrected_orientation=data.get('corrected_orientation'),
            notes=data.get('notes')
        )
        session.add(verification)
        session.commit()

        response = verification.to_dict()
        session.close()

        print(f"Verification recorded for ROI {roi_id} by {data['user_email']}")

        return jsonify(response)

    except Exception as e:
        print(f"Error verifying ROI: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/jobs/<job_id>/export', methods=['GET'])
def export_job_labels(job_id):
    """Export verified labels for training"""
    try:
        session = get_session()
        job = session.query(Job).filter_by(id=job_id).first()

        if not job:
            session.close()
            return jsonify({'error': 'Job not found'}), 404

        # Get all images and ROIs with verifications
        images = session.query(Image).filter_by(job_id=job_id).all()

        export_data = []
        for image in images:
            rois = session.query(ROI).filter_by(image_id=image.id).all()

            for roi in rois:
                verifications = session.query(Verification).filter_by(roi_id=roi.id).all()

                # Use latest verification if available
                final_label = {
                    'image_filename': image.filename,
                    'roi_index': roi.roi_index,
                    'roi_image_url': roi.roi_image_url,
                    'predicted_type': roi.predicted_type,
                    'predicted_orientation': roi.predicted_orientation,
                    'confidence': roi.confidence,
                    'bounding_box': roi.bounding_box,
                    'center': roi.center,
                    'area': roi.area
                }

                if verifications:
                    latest_verification = max(verifications, key=lambda v: v.verified_at)
                    final_label['verified'] = True
                    final_label['is_correct'] = latest_verification.is_correct
                    final_label['corrected_type'] = latest_verification.corrected_type
                    final_label['corrected_orientation'] = latest_verification.corrected_orientation
                    final_label['verified_by'] = latest_verification.user_email
                    final_label['verified_at'] = latest_verification.verified_at.isoformat()

                    # Use corrected labels if marked wrong
                    if not latest_verification.is_correct:
                        final_label['final_type'] = latest_verification.corrected_type or roi.predicted_type
                        final_label['final_orientation'] = latest_verification.corrected_orientation or roi.predicted_orientation
                    else:
                        final_label['final_type'] = roi.predicted_type
                        final_label['final_orientation'] = roi.predicted_orientation
                else:
                    final_label['verified'] = False
                    final_label['final_type'] = roi.predicted_type
                    final_label['final_orientation'] = roi.predicted_orientation

                export_data.append(final_label)

        session.close()

        return jsonify({
            'job_id': job_id,
            'total_labels': len(export_data),
            'labels': export_data
        })

    except Exception as e:
        print(f"Error exporting labels: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    try:
        session = get_session()
        jobs = session.query(Job).order_by(Job.created_at.desc()).limit(50).all()
        response = [job.to_dict() for job in jobs]
        session.close()

        return jsonify({'jobs': response})

    except Exception as e:
        print(f"Error listing jobs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a job and all associated data"""
    try:
        session = get_session()
        job = session.query(Job).filter_by(id=job_id).first()

        if not job:
            session.close()
            return jsonify({'error': 'Job not found'}), 404

        # Get all images for this job
        images = session.query(Image).filter_by(job_id=job_id).all()

        # Delete all ROIs and verifications for each image
        for image in images:
            rois = session.query(ROI).filter_by(image_id=image.id).all()
            for roi in rois:
                # Delete verifications for this ROI
                session.query(Verification).filter_by(roi_id=roi.id).delete()
                # Delete the ROI
                session.delete(roi)
            # Delete the image
            session.delete(image)

        # Delete the job
        session.delete(job)
        session.commit()
        session.close()

        print(f"Deleted job {job_id}")

        return jsonify({'success': True, 'message': 'Job deleted successfully'})

    except Exception as e:
        print(f"Error deleting job: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
