"""
Migration script to add user_email column to jobs table for user isolation
Run this once to update the database schema
"""
from models import get_db_engine
from sqlalchemy import text

def migrate():
    """Add user_email column to jobs table"""
    engine = get_db_engine()

    with engine.connect() as conn:
        # Start transaction
        trans = conn.begin()

        try:
            print("Adding user_email column to jobs table...")

            # Add user_email column (nullable for backwards compatibility)
            conn.execute(text("""
                ALTER TABLE jobs
                ADD COLUMN IF NOT EXISTS user_email VARCHAR(255)
            """))

            # Add index for faster queries
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_jobs_user_email
                ON jobs (user_email)
            """))

            print("✓ Migration completed successfully")
            print("  - Added user_email column to jobs table")
            print("  - Added index on user_email for performance")
            print()
            print("Note: Existing jobs will have NULL user_email and will not be visible")
            print("      to any specific user (backwards compatible behavior)")

            trans.commit()

        except Exception as e:
            print(f"✗ Migration failed: {e}")
            trans.rollback()
            raise

if __name__ == '__main__':
    print("=" * 60)
    print("Database Migration: Add User Isolation to Jobs")
    print("=" * 60)
    print()

    migrate()
