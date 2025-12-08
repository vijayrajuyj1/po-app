from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Any
from pydantic import BaseModel, Field as PydanticField, conlist, constr


class ProcessRequest(BaseModel):
    po_id: str = PydanticField(alias="poId")
    vendor_id: str = PydanticField(alias="vendorId")
    selected_fields: List[str] = PydanticField(default=[], alias="selectedFields")
    previous_file_urls: Optional[List[str]] = PydanticField(default=None, alias="previousFileUrls")
    metadata: Optional[dict[str, Any]] = PydanticField(default=None, alias="metadata")

    class Config:
        allow_population_by_field_name = True


class SessionResponse(BaseModel):
    id: str = PydanticField(alias="id")
    po_id: str = PydanticField(alias="poId")
    vendor_id: str = PydanticField(alias="vendorId")
    po_number: Optional[str] = PydanticField(default=None, alias="poNumber")
    current_version: int = PydanticField(alias="currentVersion")
    session_key: Optional[str] = PydanticField(default=None, alias="sessionKey")
    created_at: datetime = PydanticField(alias="createdAt")
    updated_at: datetime = PydanticField(alias="updatedAt")

    class Config:
        from_attributes = True
        allow_population_by_field_name = True


class RunFileResponse(BaseModel):
    id: str = PydanticField(alias="id")
    filename: str = PydanticField(alias="filename")
    s3_url: str = PydanticField(alias="s3Url")
    mime_type: str = PydanticField(alias="mimeType")
    size_bytes: int = PydanticField(alias="sizeBytes")
    created_at: datetime = PydanticField(alias="createdAt")
    metadata: Optional[dict] = PydanticField(default=None, alias="metadata")

    class Config:
        from_attributes = True
        allow_population_by_field_name = True


class RunResponse(BaseModel):
    id: str = PydanticField(alias="id")
    session_id: str = PydanticField(alias="sessionId")
    version: int = PydanticField(alias="version")
    status: str = PydanticField(alias="status")
    # Enriched relations (optional; populated in specific endpoints)
    po_id: Optional[str] = PydanticField(default=None, alias="poId")
    po_number: Optional[str] = PydanticField(default=None, alias="poNumber")
    vendor_id: Optional[str] = PydanticField(default=None, alias="vendorId")
    vendor_name: Optional[str] = PydanticField(default=None, alias="vendorName")
    # Supports grouped selected fields by category:
    # [{ "categoryId": "uuid", "fieldIds": ["uuid", ...] }, ...]
    selected_fields: Optional[List[dict]] = PydanticField(default=None, alias="selectedFields")
    uploaded_files_count: int = PydanticField(alias="uploadedFilesCount")
    reused_files_count: int = PydanticField(alias="reusedFilesCount")
    created_at: datetime = PydanticField(alias="createdAt")
    files: List[RunFileResponse] = PydanticField(default_factory=list, alias="files")

    class Config:
        from_attributes = True
        allow_population_by_field_name = True


class ProcessResponse(BaseModel):
    session: SessionResponse
    run: RunResponse


class VerifyRejectResponse(BaseModel):
    id: str = PydanticField(alias="id")
    status: str = PydanticField(alias="status")


class ActivityItem(BaseModel):
    id: str = PydanticField(alias="id")
    user_id: Optional[str] = PydanticField(default=None, alias="userId")
    session_id: Optional[str] = PydanticField(default=None, alias="sessionId")
    extraction_run_id: Optional[str] = PydanticField(default=None, alias="runId")
    action: str = PydanticField(alias="action")
    description: Optional[str] = PydanticField(default=None, alias="description")
    status_from: Optional[str] = PydanticField(default=None, alias="statusFrom")
    status_to: Optional[str] = PydanticField(default=None, alias="statusTo")
    metadata: Optional[dict] = PydanticField(default=None, alias="metadata")
    created_at: datetime = PydanticField(alias="createdAt")

    class Config:
        from_attributes = True
        allow_population_by_field_name = True


