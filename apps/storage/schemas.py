from typing import List

from pydantic import BaseModel, constr


class PresignRequest(BaseModel):
    """
    Request schema for generating S3 pre-signed URLs.
    Accepts an array of S3 object URLs (as strings).
    """

    urls: List[constr(strip_whitespace=True, min_length=1)]


class PresignedUrlItem(BaseModel):
    """
    Represents a single original URL and its corresponding pre-signed URL.
    """

    original_url: str
    presigned_url: str


class PresignResponse(BaseModel):
    """
    Response schema wrapping all generated pre-signed URLs.
    """

    items: List[PresignedUrlItem]



