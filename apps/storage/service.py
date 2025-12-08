import asyncio
from typing import List
from urllib.parse import urlparse

import boto3
from fastapi import HTTPException, status

from settings.config import get_settings


class StorageService:
    """
    Storage-related helpers (S3 for now).
    """

    @staticmethod
    def _get_s3_client():
        """
        Construct a boto3 S3 client using application settings.
        Prefers explicit credentials from settings when provided.
        """
        settings = get_settings()
        kwargs: dict = {}

        if getattr(settings, "AWS_REGION", None):
            kwargs["region_name"] = settings.AWS_REGION
        if getattr(settings, "AWS_ACCESS_KEY_ID", None) and getattr(settings, "AWS_SECRET_ACCESS_KEY", None):
            kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
            kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY

        return boto3.client("s3", **kwargs)

    @staticmethod
    async def generate_presigned_get_urls(urls: List[str], expires_in: int = 3600) -> List[dict]:
        """
        Given a list of S3 object URLs that were generated for the configured bucket,
        return a list of dicts with original_url + pre-signed URL for GET.

        Assumes URLs follow the pattern used elsewhere in the app, e.g.:
          https://{bucket}.s3.{region}.amazonaws.com/path/to/object
        In that case we only need the path portion as S3 object key and the bucket
        from settings.
        """
        settings = get_settings()
        bucket = settings.AWS_S3_BUCKET
        if not bucket:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="AWS_S3_BUCKET is not configured in environment.",
            )

        client = StorageService._get_s3_client()

        async def _presign_single(url: str) -> dict:
            parsed = urlparse(url)
            key = (parsed.path or "").lstrip("/")
            if not key:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not extract S3 object key from URL: {url}",
                )

            def _generate() -> str:
                return client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": bucket, "Key": key},
                    ExpiresIn=expires_in,
                )

            presigned_url = await asyncio.to_thread(_generate)
            return {"original_url": url, "presigned_url": presigned_url}

        results: List[dict] = []
        for u in urls or []:
            # Skip empty/whitespace-only values defensively
            if not (u and u.strip()):
                continue
            results.append(await _presign_single(u.strip()))

        return results



