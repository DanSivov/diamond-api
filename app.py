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
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

app = Flask(__name__)

# Configure CORS to allow all origins with proper headers
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
})

# Add CORS headers to all error responses
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# Admin configuration
ADMIN_EMAIL = 'sivovolenkodaniil@gmail.com'

def is_admin(email):
    """Check if the given email is an admin"""
    return email and email.lower() == ADMIN_EMAIL.lower()

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
    session = None
    try:
        data = request.get_json()

        if not data or 'files' not in data:
            return jsonify({'error': 'No files provided'}), 400

        files_data = data['files']
        if not files_data or len(files_data) == 0:
            return jsonify({'error': 'Empty file list'}), 400

        user_email = data.get('user_email')  # Get user email from request

        session = get_session()

        # Create job record
        job = Job(
            user_email=user_email,
            total_images=len(files_data),
            processed_images=0,
            status='pending'
        )
        session.add(job)
        session.commit()

        job_id = job.id
        session.close()
        session = None

        # Queue async task with retry handling for Redis connection issues
        try:
            process_batch_job.delay(job_id, files_data)
            print(f"Created job {job_id} with {len(files_data)} images")
        except Exception as redis_error:
            print(f"Warning: Failed to queue task (Redis may be unavailable): {redis_error}")
            # Update job status to indicate queue failure
            session = get_session()
            job = session.query(Job).filter_by(id=job_id).first()
            if job:
                job.status = 'queue_failed'
                job.error_message = f"Task queue unavailable: {str(redis_error)}"
                session.commit()
            session.close()
            return jsonify({
                'job_id': job_id,
                'status': 'queue_failed',
                'error': 'Task queue temporarily unavailable. Please try again in a few moments.',
                'total_images': len(files_data)
            }), 503

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

        # Check if this ROI already has a verification from this user
        existing = session.query(Verification).filter_by(
            roi_id=roi_id,
            user_email=data['user_email']
        ).first()

        if existing:
            # Update existing verification
            existing.is_correct = data['is_correct']
            existing.corrected_type = data.get('corrected_type')
            existing.corrected_orientation = data.get('corrected_orientation')
            existing.notes = data.get('notes')
            existing.verified_at = datetime.utcnow()
            verification = existing
        else:
            # Create new verification record
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

        # Update job progress
        # Get the job for this ROI
        image = session.query(Image).filter_by(id=roi.image_id).first()
        if image:
            job = session.query(Job).filter_by(id=image.job_id).first()
            if job:
                # Count verified ROIs for this job
                from sqlalchemy import func, distinct
                verified_count = session.query(func.count(distinct(Verification.roi_id)))\
                    .join(ROI, Verification.roi_id == ROI.id)\
                    .join(Image, ROI.image_id == Image.id)\
                    .filter(Image.job_id == job.id)\
                    .scalar()

                job.verified_rois = verified_count or 0

                # Update job status
                if job.status == 'ready' and verified_count > 0:
                    job.status = 'in_progress'
                elif verified_count >= job.total_rois:
                    job.status = 'complete'
                    job.completed_at = datetime.utcnow()

                session.commit()
                print(f"Job {job.id} progress: {job.verified_rois}/{job.total_rois} ROIs verified, status: {job.status}")

        response = verification.to_dict()
        session.close()

        print(f"Verification recorded for ROI {roi_id} by {data['user_email']}")

        return jsonify(response)

    except Exception as e:
        print(f"Error verifying ROI: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/images/<image_id>/regenerate-graded', methods=['POST'])
def regenerate_graded_image(image_id):
    """
    Regenerate graded image with human corrections applied.

    This endpoint is called after all ROIs in an image have been verified.
    It regenerates the visualization using the corrected orientations.

    Returns:
        Base64 encoded PNG of the corrected graded image
    """
    try:
        from storage import get_storage
        from grading import PickupGrader
        from dataclasses import dataclass

        session = get_session()

        # Get image record
        image = session.query(Image).filter_by(id=image_id).first()
        if not image:
            session.close()
            return jsonify({'error': 'Image not found'}), 404

        # Get all ROIs with their verifications
        rois = session.query(ROI).filter_by(image_id=image_id).order_by(ROI.roi_index).all()

        if not rois:
            session.close()
            return jsonify({'error': 'No ROIs found for image'}), 404

        # Determine final orientation for each ROI (applying human corrections)
        roi_data = []
        for roi in rois:
            verifications = session.query(Verification).filter_by(roi_id=roi.id).all()

            # Default to predicted orientation
            final_orientation = roi.predicted_orientation

            if verifications:
                latest = max(verifications, key=lambda v: v.verified_at)
                if not latest.is_correct and latest.corrected_orientation:
                    final_orientation = latest.corrected_orientation
                elif not latest.is_correct:
                    # If marked wrong but no correction specified, flip it
                    final_orientation = 'tilted' if roi.predicted_orientation == 'table' else 'table'

            roi_data.append({
                'roi_index': roi.roi_index,
                'bounding_box': roi.bounding_box,
                'center': roi.center,
                'area': roi.area,
                'final_orientation': final_orientation,
                'predicted_type': roi.predicted_type
            })

        session.close()

        # Download original image from R2
        storage = get_storage()

        # Extract R2 key from URL
        original_url = image.original_url
        if not original_url:
            return jsonify({'error': 'Original image URL not found'}), 404

        # Download image via HTTP (simpler than extracting R2 key)
        print(f"Downloading original image from: {original_url}")
        response = requests.get(original_url, timeout=30)
        if response.status_code != 200:
            return jsonify({'error': f'Failed to download original image: {response.status_code}'}), 500

        # Decode image
        image_array = np.frombuffer(response.content, np.uint8)
        original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if original_image is None:
            return jsonify({'error': 'Failed to decode original image'}), 500

        h, w = original_image.shape[:2]
        print(f"Original image size: {w}x{h}")

        # Create mock GradedDiamond objects for visualization
        @dataclass
        class MockROI:
            contour: np.ndarray
            bounding_box: tuple
            center: tuple
            area: float
            detected_type: str
            orientation: str
            id: int

        @dataclass
        class MockGradedDiamond:
            roi: MockROI
            grade: float  # None=tilted, -1=too close, 0-10=pickable
            nearest_distance: float
            radius: float

        graded_diamonds = []
        for idx, rd in enumerate(roi_data):
            # Create contour from bounding box (rectangle)
            x, y, bw, bh = rd['bounding_box']
            contour = np.array([
                [[x, y]], [[x + bw, y]], [[x + bw, y + bh]], [[x, y + bh]]
            ], dtype=np.int32)

            # Calculate radius from area
            radius = np.sqrt(rd['area'] / np.pi)

            mock_roi = MockROI(
                contour=contour,
                bounding_box=tuple(rd['bounding_box']),
                center=tuple(rd['center']),
                area=rd['area'],
                detected_type=rd['predicted_type'],
                orientation=rd['final_orientation'],
                id=idx
            )

            # Determine grade based on final orientation
            # For simplicity, tilted diamonds get None, table diamonds get grade 5
            # (The actual proximity check would need more data)
            if rd['final_orientation'] == 'tilted':
                grade = None  # Will show as red
            else:
                grade = 5.0  # Will show as green (simplified - real logic would check proximity)

            graded_diamonds.append(MockGradedDiamond(
                roi=mock_roi,
                grade=grade,
                nearest_distance=100.0,  # Dummy value
                radius=radius
            ))

        # Create grader and generate visualization
        grader = PickupGrader(check_orientation=True, image_width_px=w)
        corrected_vis = grader.visualize_pickup_order(original_image, graded_diamonds)

        # Encode to base64
        _, buffer = cv2.imencode('.png', corrected_vis)
        corrected_b64 = base64.b64encode(buffer).decode('utf-8')

        print(f"Generated corrected visualization for image {image_id}")

        # Count how many corrections were applied
        corrections_count = 0
        for i, rd in enumerate(roi_data):
            if i < len(rois) and rd['final_orientation'] != rois[i].predicted_orientation:
                corrections_count += 1

        return jsonify({
            'image_id': image_id,
            'corrected_graded_base64': corrected_b64,
            'roi_count': len(roi_data),
            'corrections_applied': corrections_count
        })

    except Exception as e:
        print(f"Error regenerating graded image: {e}")
        import traceback
        traceback.print_exc()
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
    """List jobs for a specific user"""
    try:
        user_email = request.args.get('user_email')

        session = get_session()

        # Build query - filter must come BEFORE limit
        query = session.query(Job)

        # Filter by user_email if provided
        if user_email:
            query = query.filter(Job.user_email == user_email)

        # Apply ordering and limit
        query = query.order_by(Job.created_at.desc()).limit(50)

        jobs = query.all()
        response = [job.to_dict() for job in jobs]
        session.close()

        return jsonify({'jobs': response})

    except Exception as e:
        print(f"Error listing jobs: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace to logs
        return jsonify({'error': str(e)}), 500

# ============================================================================
# Admin Endpoints
# ============================================================================

@app.route('/admin/users', methods=['GET'])
def list_admin_users():
    """List all unique users who have created jobs (admin only)"""
    try:
        requester_email = request.args.get('requester_email')

        if not is_admin(requester_email):
            return jsonify({'error': 'Unauthorized - admin access required'}), 403

        session = get_session()

        from sqlalchemy import func

        # Get unique user emails with job counts and aggregate stats
        users_query = session.query(
            Job.user_email,
            func.count(Job.id).label('job_count'),
            func.sum(Job.total_rois).label('total_rois'),
            func.sum(Job.verified_rois).label('verified_rois'),
            func.max(Job.created_at).label('last_activity')
        ).filter(
            Job.user_email.isnot(None)
        ).group_by(Job.user_email).all()

        users = []
        for user in users_query:
            users.append({
                'email': user.user_email,
                'job_count': user.job_count,
                'total_rois': user.total_rois or 0,
                'verified_rois': user.verified_rois or 0,
                'last_activity': user.last_activity.isoformat() if user.last_activity else None
            })

        session.close()

        return jsonify({'users': users})

    except Exception as e:
        print(f"Error listing admin users: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/admin/jobs', methods=['GET'])
def list_admin_jobs():
    """List all jobs for admin panel with last verification timestamp"""
    try:
        requester_email = request.args.get('requester_email')
        user_filter = request.args.get('user_email')  # Optional: filter by specific user

        if not is_admin(requester_email):
            return jsonify({'error': 'Unauthorized - admin access required'}), 403

        session = get_session()

        from sqlalchemy import func

        query = session.query(Job)

        if user_filter:
            query = query.filter(Job.user_email == user_filter)

        jobs = query.order_by(Job.created_at.desc()).limit(100).all()

        result = []
        for job in jobs:
            job_dict = job.to_dict()

            # Get last verification timestamp for this job
            last_verification = session.query(func.max(Verification.verified_at))\
                .join(ROI, Verification.roi_id == ROI.id)\
                .join(Image, ROI.image_id == Image.id)\
                .filter(Image.job_id == job.id)\
                .scalar()

            job_dict['last_verification_at'] = last_verification.isoformat() if last_verification else None
            result.append(job_dict)

        session.close()

        return jsonify({'jobs': result})

    except Exception as e:
        print(f"Error listing admin jobs: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/admin/activity', methods=['GET'])
def get_admin_activity():
    """Get recent activity history for admin panel"""
    try:
        requester_email = request.args.get('requester_email')
        limit = request.args.get('limit', 50, type=int)

        if not is_admin(requester_email):
            return jsonify({'error': 'Unauthorized - admin access required'}), 403

        session = get_session()

        # Get recent verifications with job context
        verifications = session.query(
            Verification,
            ROI,
            Image,
            Job
        ).join(
            ROI, Verification.roi_id == ROI.id
        ).join(
            Image, ROI.image_id == Image.id
        ).join(
            Job, Image.job_id == Job.id
        ).order_by(
            Verification.verified_at.desc()
        ).limit(limit).all()

        activity = []
        for v, roi, image, job in verifications:
            activity.append({
                'type': 'verification',
                'timestamp': v.verified_at.isoformat(),
                'user_email': v.user_email,
                'job_id': str(job.id),
                'job_owner': job.user_email,
                'image_filename': image.filename,
                'roi_index': roi.roi_index,
                'is_correct': v.is_correct,
                'corrected_orientation': v.corrected_orientation
            })

        # Also get recent job creations
        recent_jobs = session.query(Job).order_by(Job.created_at.desc()).limit(limit).all()
        for job in recent_jobs:
            activity.append({
                'type': 'job_created',
                'timestamp': job.created_at.isoformat(),
                'user_email': job.user_email,
                'job_id': str(job.id),
                'total_images': job.total_images
            })

        # Sort combined activity by timestamp
        activity.sort(key=lambda x: x['timestamp'], reverse=True)

        session.close()

        return jsonify({'activity': activity[:limit]})

    except Exception as e:
        print(f"Error getting admin activity: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a job and all associated data including R2 files"""
    try:
        requester_email = request.args.get('requester_email')

        session = get_session()
        job = session.query(Job).filter_by(id=job_id).first()

        if not job:
            session.close()
            return jsonify({'error': 'Job not found'}), 404

        # Authorization check: must be job owner OR admin
        if job.user_email != requester_email and not is_admin(requester_email):
            session.close()
            return jsonify({'error': 'Unauthorized - not job owner or admin'}), 403

        # Delete R2 files first
        r2_deleted_count = 0
        try:
            from storage import get_storage
            storage = get_storage()
            r2_prefix = f"jobs/{job_id}/"
            files_to_delete = storage.list_files(prefix=r2_prefix)

            for file_key in files_to_delete:
                if storage.delete_image(file_key):
                    r2_deleted_count += 1

            print(f"Deleted {r2_deleted_count}/{len(files_to_delete)} R2 files for job {job_id}")
        except Exception as r2_error:
            print(f"Warning: Failed to delete R2 files for job {job_id}: {r2_error}")
            # Continue with DB deletion even if R2 fails

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

        print(f"Deleted job {job_id} (requested by {requester_email})")

        return jsonify({
            'success': True,
            'message': 'Job deleted successfully',
            'r2_files_deleted': r2_deleted_count
        })

    except Exception as e:
        print(f"Error deleting job: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
