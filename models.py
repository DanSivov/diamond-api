"""
Database models for diamond classification verification system
"""
from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import uuid
import os

Base = declarative_base()

def get_db_engine():
    """Get database engine from environment"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    # Railway PostgreSQL URLs start with postgres://, but SQLAlchemy 2.0 requires postgresql://
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)

    return create_engine(database_url, pool_pre_ping=True)

def get_session():
    """Get database session"""
    engine = get_db_engine()
    Session = sessionmaker(bind=engine)
    return Session()

class Job(Base):
    """Batch processing job"""
    __tablename__ = 'jobs'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_email = Column(String(255), nullable=True, index=True)  # Nullable for backwards compatibility
    status = Column(String(20), nullable=False, default='pending')  # pending, processing, in_progress, ready, complete, failed
    total_images = Column(Integer, nullable=False)
    processed_images = Column(Integer, nullable=False, default=0)
    total_rois = Column(Integer, nullable=True, default=0)  # Nullable for backwards compatibility
    verified_rois = Column(Integer, nullable=True, default=0)  # Nullable for backwards compatibility
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    images = relationship('Image', back_populates='job', cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'user_email': getattr(self, 'user_email', None),  # Default to None if column doesn't exist
            'status': self.status,
            'total_images': self.total_images,
            'processed_images': self.processed_images,
            'total_rois': getattr(self, 'total_rois', 0),  # Default to 0 if column doesn't exist
            'verified_rois': getattr(self, 'verified_rois', 0),  # Default to 0 if column doesn't exist
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message
        }

class Image(Base):
    """Processed image in a job"""
    __tablename__ = 'images'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String(36), ForeignKey('jobs.id'), nullable=False)
    filename = Column(String(500), nullable=False)
    original_url = Column(Text, nullable=True)  # R2 URL
    graded_url = Column(Text, nullable=True)    # R2 URL
    total_diamonds = Column(Integer, nullable=False, default=0)
    table_count = Column(Integer, nullable=False, default=0)
    tilted_count = Column(Integer, nullable=False, default=0)
    pickable_count = Column(Integer, nullable=False, default=0)
    invalid_count = Column(Integer, nullable=False, default=0)
    average_grade = Column(Float, nullable=True)
    processed_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    job = relationship('Job', back_populates='images')
    rois = relationship('ROI', back_populates='image', cascade='all, delete-orphan')

    def to_dict(self, include_rois=False):
        data = {
            'id': self.id,
            'job_id': self.job_id,
            'filename': self.filename,
            'original_url': self.original_url,
            'graded_url': self.graded_url,
            'total_diamonds': self.total_diamonds,
            'table_count': self.table_count,
            'tilted_count': self.tilted_count,
            'pickable_count': self.pickable_count,
            'invalid_count': self.invalid_count,
            'average_grade': self.average_grade,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None
        }
        if include_rois:
            data['rois'] = [roi.to_dict() for roi in self.rois]
        return data

class ROI(Base):
    """Individual diamond ROI"""
    __tablename__ = 'rois'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String(36), ForeignKey('images.id'), nullable=False)
    roi_index = Column(Integer, nullable=False)
    roi_image_url = Column(Text, nullable=True)  # R2 URL
    predicted_type = Column(String(20), nullable=False)
    predicted_orientation = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    bounding_box = Column(JSON, nullable=False)  # [x, y, w, h]
    center = Column(JSON, nullable=False)        # [x, y]
    area = Column(Integer, nullable=False)
    features = Column(JSON, nullable=True)       # Additional features

    # Relationships
    image = relationship('Image', back_populates='rois')
    verifications = relationship('Verification', back_populates='roi', cascade='all, delete-orphan')

    def to_dict(self, include_verifications=False):
        data = {
            'id': self.id,
            'image_id': self.image_id,
            'roi_index': self.roi_index,
            'roi_image_url': self.roi_image_url,
            'predicted_type': self.predicted_type,
            'predicted_orientation': self.predicted_orientation,
            'confidence': self.confidence,
            'bounding_box': self.bounding_box,
            'center': self.center,
            'area': self.area,
            'features': self.features
        }
        if include_verifications:
            data['verifications'] = [v.to_dict() for v in self.verifications]
        return data

class Verification(Base):
    """User verification of ROI classification"""
    __tablename__ = 'verifications'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    roi_id = Column(String(36), ForeignKey('rois.id'), nullable=False)
    user_email = Column(String(255), nullable=False)  # Email used as username (no auth)
    is_correct = Column(Boolean, nullable=False)
    corrected_type = Column(String(20), nullable=True)
    corrected_orientation = Column(String(20), nullable=True)
    is_sam_failure = Column(Boolean, nullable=False, default=False)  # Contour/SAM segmentation failure
    notes = Column(Text, nullable=True)
    verified_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    roi = relationship('ROI', back_populates='verifications')

    def to_dict(self):
        return {
            'id': self.id,
            'roi_id': self.roi_id,
            'user_email': self.user_email,
            'is_correct': self.is_correct,
            'corrected_type': self.corrected_type,
            'corrected_orientation': self.corrected_orientation,
            'is_sam_failure': self.is_sam_failure,
            'notes': self.notes,
            'verified_at': self.verified_at.isoformat() if self.verified_at else None
        }

def init_db():
    """Initialize database tables and run migrations"""
    from sqlalchemy import text

    engine = get_db_engine()
    Base.metadata.create_all(engine)
    print("Database tables created successfully")

    # Run migration: Add user_email column for user isolation
    try:
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                # Check if user_email column exists
                result = conn.execute(text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name='jobs' AND column_name='user_email'
                """))

                if result.fetchone() is None:
                    print("Running migration: Adding user_email column to jobs table...")

                    # Add user_email column
                    conn.execute(text("""
                        ALTER TABLE jobs
                        ADD COLUMN user_email VARCHAR(255)
                    """))

                    # Add index for performance
                    conn.execute(text("""
                        CREATE INDEX idx_jobs_user_email
                        ON jobs (user_email)
                    """))

                    print("✓ Migration completed: user_email column added")
                else:
                    print("✓ user_email column already exists")

                trans.commit()
            except Exception as e:
                trans.rollback()
                print(f"Migration warning (may be safe to ignore if column exists): {e}")
    except Exception as e:
        print(f"Migration check failed: {e}")

    # Run migration: Add is_sam_failure column to verifications table
    try:
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                # Check if is_sam_failure column exists
                result = conn.execute(text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name='verifications' AND column_name='is_sam_failure'
                """))

                if result.fetchone() is None:
                    print("Running migration: Adding is_sam_failure column to verifications table...")

                    # Add is_sam_failure column with default false
                    conn.execute(text("""
                        ALTER TABLE verifications
                        ADD COLUMN is_sam_failure BOOLEAN NOT NULL DEFAULT FALSE
                    """))

                    print("✓ Migration completed: is_sam_failure column added")
                else:
                    print("✓ is_sam_failure column already exists")

                trans.commit()
            except Exception as e:
                trans.rollback()
                print(f"Migration warning (may be safe to ignore if column exists): {e}")
    except Exception as e:
        print(f"Migration check failed: {e}")

if __name__ == '__main__':
    init_db()
