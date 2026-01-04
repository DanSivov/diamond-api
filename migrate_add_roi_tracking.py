"""
Database migration to add ROI tracking to jobs table
"""
import os
from sqlalchemy import create_engine, text

def get_db_url():
    """Get database URL from environment"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    # Railway PostgreSQL URLs start with postgres://, but SQLAlchemy 2.0 requires postgresql://
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)

    return database_url

def migrate():
    """Add total_rois and verified_rois columns to jobs table"""
    engine = create_engine(get_db_url())

    with engine.connect() as conn:
        # Check if columns already exist
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'jobs' AND column_name IN ('total_rois', 'verified_rois')
        """))
        existing_columns = [row[0] for row in result]

        # Add total_rois column if it doesn't exist
        if 'total_rois' not in existing_columns:
            print("Adding total_rois column...")
            conn.execute(text("""
                ALTER TABLE jobs
                ADD COLUMN total_rois INTEGER DEFAULT 0
            """))
            conn.commit()
            print("✓ Added total_rois column")
        else:
            print("✓ total_rois column already exists")

        # Add verified_rois column if it doesn't exist
        if 'verified_rois' not in existing_columns:
            print("Adding verified_rois column...")
            conn.execute(text("""
                ALTER TABLE jobs
                ADD COLUMN verified_rois INTEGER DEFAULT 0
            """))
            conn.commit()
            print("✓ Added verified_rois column")
        else:
            print("✓ verified_rois column already exists")

        # Update existing jobs with ROI counts
        print("Updating existing jobs with ROI counts...")
        conn.execute(text("""
            UPDATE jobs j
            SET total_rois = (
                SELECT COUNT(r.id)
                FROM rois r
                JOIN images i ON r.image_id = i.id
                WHERE i.job_id = j.id
            ),
            verified_rois = (
                SELECT COUNT(DISTINCT v.roi_id)
                FROM verifications v
                JOIN rois r ON v.roi_id = r.id
                JOIN images i ON r.image_id = i.id
                WHERE i.job_id = j.id
            )
        """))
        conn.commit()
        print("✓ Updated existing jobs with ROI counts")

        # Update job statuses
        print("Updating job statuses...")
        conn.execute(text("""
            UPDATE jobs
            SET status = CASE
                WHEN status = 'complete' AND verified_rois = 0 THEN 'ready'
                WHEN status = 'complete' AND verified_rois > 0 AND verified_rois < total_rois THEN 'in_progress'
                ELSE status
            END
        """))
        conn.commit()
        print("✓ Updated job statuses")

        # Show updated job counts
        result = conn.execute(text("""
            SELECT
                status,
                COUNT(*) as count,
                SUM(total_rois) as total_rois,
                SUM(verified_rois) as verified_rois
            FROM jobs
            GROUP BY status
        """))

        print("\nJob summary:")
        print("Status        Count  Total ROIs  Verified ROIs")
        print("-" * 50)
        for row in result:
            print(f"{row[0]:<12}  {row[1]:>5}  {row[2] or 0:>10}  {row[3] or 0:>13}")

    print("\n✓ Migration complete!")

if __name__ == '__main__':
    migrate()
