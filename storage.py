"""
Cloudflare R2 storage service for images
"""
import boto3
from botocore.client import Config
import os
import io
from typing import Optional

class R2Storage:
    """Cloudflare R2 storage client"""

    def __init__(self):
        self.account_id = os.environ.get('R2_ACCOUNT_ID')
        self.access_key_id = os.environ.get('R2_ACCESS_KEY_ID')
        self.secret_access_key = os.environ.get('R2_SECRET_ACCESS_KEY')
        self.bucket_name = os.environ.get('R2_BUCKET_NAME')
        self.public_url = os.environ.get('R2_PUBLIC_URL')  # Public R2.dev URL

        if not all([self.account_id, self.access_key_id, self.secret_access_key, self.bucket_name]):
            raise ValueError("Missing R2 configuration. Check environment variables.")

        # R2 endpoint for API access
        self.endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"

        # Initialize S3 client for R2
        self.client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(signature_version='s3v4'),
            region_name='auto'
        )

    def upload_image(self, image_bytes: bytes, filename: str, content_type: str = 'image/png') -> str:
        """
        Upload image to R2 and return public URL

        Args:
            image_bytes: Image data as bytes
            filename: Filename/key in R2
            content_type: MIME type

        Returns:
            Public URL to access the image
        """
        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=filename,
                Body=image_bytes,
                ContentType=content_type
            )

            # Return public URL (use R2.dev public URL if available, otherwise fall back)
            if self.public_url:
                public_url = f"{self.public_url}/{filename}"
            else:
                # Fallback to direct URL (won't work in browser without CORS)
                public_url = f"{self.endpoint_url}/{self.bucket_name}/{filename}"

            return public_url

        except Exception as e:
            print(f"Error uploading to R2: {e}")
            raise

    def upload_numpy_image(self, image_array, filename: str) -> str:
        """
        Upload numpy array image to R2

        Args:
            image_array: OpenCV/numpy image array
            filename: Filename/key in R2

        Returns:
            Public URL to access the image
        """
        import cv2

        # Encode image to PNG
        success, buffer = cv2.imencode('.png', image_array)
        if not success:
            raise ValueError("Failed to encode image")

        # Upload
        return self.upload_image(buffer.tobytes(), filename, 'image/png')

    def download_image(self, filename: str) -> bytes:
        """
        Download image from R2

        Args:
            filename: Filename/key in R2

        Returns:
            Image bytes
        """
        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=filename
            )
            return response['Body'].read()

        except Exception as e:
            print(f"Error downloading from R2: {e}")
            raise

    def delete_image(self, filename: str) -> bool:
        """
        Delete image from R2

        Args:
            filename: Filename/key in R2

        Returns:
            True if successful
        """
        try:
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=filename
            )
            return True

        except Exception as e:
            print(f"Error deleting from R2: {e}")
            return False

    def list_files(self, prefix: str = '') -> list:
        """
        List files in R2 bucket

        Args:
            prefix: Filter by prefix

        Returns:
            List of file keys
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if 'Contents' not in response:
                return []

            return [obj['Key'] for obj in response['Contents']]

        except Exception as e:
            print(f"Error listing R2 files: {e}")
            return []

# Global instance
_storage = None

def get_storage() -> R2Storage:
    """Get or create R2 storage instance"""
    global _storage
    if _storage is None:
        _storage = R2Storage()
    return _storage
