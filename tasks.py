"""
Celery tasks for async image processing
"""
from celery_app import celery_app
from models import get_session, Job, Image, ROI
from storage import get_storage
import cv2
import numpy as np
from datetime import datetime
import gc
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

@celery_app.task(bind=True)
def process_batch_job(self, job_id, image_files_data):
    """
    Process a batch of images asynchronously

    Args:
        job_id: UUID of the job
        image_files_data: List of dicts with 'filename' and 'data' (base64 or bytes)
    """
    from core import DiamondClassifier

    session = get_session()
    storage = get_storage()

    try:
        # Get job from database
        job = session.query(Job).filter_by(id=job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Update job status
        job.status = 'processing'
        session.commit()

        # Load classifier
        model_path = Path(__file__).parent / 'models' / 'ml_classifier' / 'best_model_randomforest.pkl'
        features_path = Path(__file__).parent / 'models' / 'ml_classifier' / 'feature_names.json'

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        classifier = DiamondClassifier(str(model_path), str(features_path))

        # Process each image
        for idx, file_data in enumerate(image_files_data):
            try:
                filename = file_data['filename']
                image_data = file_data['data']

                print(f"Processing image {idx + 1}/{len(image_files_data)}: {filename}")

                # Decode base64 to bytes
                import base64
                image_bytes = base64.b64decode(image_data)

                # Decode image
                file_array = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

                if image is None:
                    print(f"Failed to decode image: {filename}")
                    continue

                # Classify image
                result = classifier.classify_image(image, filename)

                # Upload original image to R2
                original_filename = f"jobs/{job_id}/originals/{filename}"
                original_url = storage.upload_numpy_image(image, original_filename)

                # Upload graded visualization to R2
                graded_url = None
                visualization = classifier.get_visualization()
                if visualization is not None:
                    graded_filename = f"jobs/{job_id}/graded/{filename}_graded.png"
                    graded_url = storage.upload_numpy_image(visualization, graded_filename)

                # Create image record
                image_record = Image(
                    job_id=job_id,
                    filename=filename,
                    original_url=original_url,
                    graded_url=graded_url,
                    total_diamonds=result.total_diamonds,
                    table_count=result.table_count,
                    tilted_count=result.tilted_count,
                    pickable_count=result.pickable_count,
                    invalid_count=result.invalid_count,
                    average_grade=result.average_grade
                )
                session.add(image_record)
                session.flush()  # Get image_record.id

                # Create ROI records and upload ROI images
                for roi_idx, classification in enumerate(result.classifications):
                    # Upload ROI image to R2
                    roi_url = None
                    if hasattr(classifier, '_last_graded_diamonds') and roi_idx < len(classifier._last_graded_diamonds):
                        gd = classifier._last_graded_diamonds[roi_idx]
                        roi_img = gd.roi.roi_image.copy()  # Copy to avoid modifying original

                        # Draw diamond outline on ROI image for visual reference
                        if hasattr(gd.roi, 'contour') and gd.roi.contour is not None:
                            # Adjust contour coordinates to ROI local space
                            x, y, w, h = gd.roi.bounding_box
                            padding = 10  # Same padding used in detector
                            contour_local = gd.roi.contour - np.array([x - padding, y - padding])

                            # Color based on orientation: green=table, red=tilted
                            orientation = classification.orientation
                            contour_color = (0, 255, 0) if orientation == 'table' else (0, 0, 255)
                            cv2.drawContours(roi_img, [contour_local], -1, contour_color, 2)

                        roi_filename = f"jobs/{job_id}/rois/{image_record.id}_{roi_idx}.png"
                        roi_url = storage.upload_numpy_image(roi_img, roi_filename)

                    # Create ROI record
                    roi_record = ROI(
                        image_id=image_record.id,
                        roi_index=roi_idx,
                        roi_image_url=roi_url,
                        predicted_type=classification.diamond_type,
                        predicted_orientation=classification.orientation,
                        confidence=classification.confidence,
                        bounding_box=list(classification.bounding_box),
                        center=list(classification.center),
                        area=classification.area,
                        features=classification.features
                    )
                    session.add(roi_record)

                # Update job progress
                job.processed_images += 1
                session.commit()

                # Update task progress
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': idx + 1,
                        'total': len(image_files_data),
                        'filename': filename
                    }
                )

                # Clear memory
                del image, file_array
                if visualization is not None:
                    del visualization
                gc.collect()

                print(f"Successfully processed {filename}: {result.total_diamonds} diamonds")

            except Exception as e:
                print(f"Error processing {file_data.get('filename', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next image

        # Count total ROIs for this job
        print(f"Counting total ROIs for job {job_id}...")
        import time
        start_time = time.time()

        from sqlalchemy import func
        total_rois = session.query(func.count(ROI.id)).join(Image).filter(Image.job_id == job_id).scalar()

        count_time = time.time() - start_time
        print(f"ROI count query took {count_time:.2f} seconds, found {total_rois} ROIs")

        job.total_rois = total_rois or 0
        job.verified_rois = 0

        # Mark job as ready for verification (not complete yet)
        print(f"Setting job status to 'ready' and committing to database...")
        start_commit = time.time()
        job.status = 'ready'
        session.commit()
        commit_time = time.time() - start_commit
        print(f"Database commit took {commit_time:.2f} seconds")

        print(f"Job {job_id} processing complete: {job.processed_images}/{job.total_images} images, {total_rois} ROIs ready for verification")

        return {
            'job_id': job_id,
            'status': 'complete',
            'processed_images': job.processed_images,
            'total_images': job.total_images
        }

    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()

        # Mark job as failed
        if 'job' in locals():
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            session.commit()

        raise

    finally:
        session.close()
        gc.collect()
