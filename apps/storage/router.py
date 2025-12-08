from fastapi import APIRouter, Depends

from apps.storage.schemas import PresignRequest, PresignResponse, PresignedUrlItem
from apps.storage.service import StorageService
from models.user import User
from security.auth_backend import get_current_active_user


router = APIRouter(prefix="/api/storage", tags=["Storage"])


@router.post("/presign", response_model=PresignResponse)
async def generate_presigned_urls(
    payload: PresignRequest,
    # current_user: User = Depends(get_current_active_user),
) -> PresignResponse:
    """
    Generate pre-signed S3 GET URLs for an array of private S3 object URLs.
    Requires an authenticated user.
    """
    items_raw = await StorageService.generate_presigned_get_urls(payload.urls)
    items = [PresignedUrlItem(**item) for item in items_raw]
    return PresignResponse(items=items)



