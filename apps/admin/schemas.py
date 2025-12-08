from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field as PydanticField

from apps.extraction.schemas import ProcessResponse


class RunVersionItem(BaseModel):
    run_id: str = PydanticField(alias="runId")
    version: int = PydanticField(alias="version")
    status: str = PydanticField(alias="status")
    created_at: datetime = PydanticField(alias="createdAt")

    class Config:
        from_attributes = True
        allow_population_by_field_name = True


class AdminFlagItem(BaseModel):
    id: str = PydanticField(alias="id")
    status: str = PydanticField(alias="status")
    reason: str = PydanticField(alias="reason")
    created_at: datetime = PydanticField(alias="createdAt")
    flagged_date: datetime = PydanticField(alias="flaggedDate")
    flagged_by: Optional[str] = PydanticField(default=None, alias="flaggedBy")
    flagged_by_email: Optional[str] = PydanticField(default=None, alias="flaggedByEmail")
    priority: Optional[str] = PydanticField(default=None, alias="priority")
    admin_note: Optional[str] = PydanticField(default=None, alias="adminNote")
    session_id: Optional[str] = PydanticField(default=None, alias="sessionId")
    run_id: Optional[str] = PydanticField(default=None, alias="runId")
    run_version: Optional[int] = PydanticField(default=None, alias="runVersion")
    current_version: Optional[int] = PydanticField(default=None, alias="currentVersion")
    versions: Optional[List[RunVersionItem]] = PydanticField(default=None, alias="versions")

    vendor_id: Optional[str] = PydanticField(default=None, alias="vendorId")
    vendor_name: Optional[str] = PydanticField(default=None, alias="vendorName")
    po_id: Optional[str] = PydanticField(default=None, alias="poId")
    po_number: Optional[str] = PydanticField(default=None, alias="poNumber")

    class Config:
        from_attributes = True
        allow_population_by_field_name = True


class RecentProcessItem(BaseModel):
    session_id: str = PydanticField(alias="sessionId")
    vendor_id: str = PydanticField(alias="vendorId")
    vendor_name: Optional[str] = PydanticField(default=None, alias="vendorName")
    po_id: str = PydanticField(alias="poId")
    po_number: Optional[str] = PydanticField(default=None, alias="poNumber")
    current_version: int = PydanticField(alias="currentVersion")
    status: Optional[str] = PydanticField(default=None, alias="status")
    latest_run_id: Optional[str] = PydanticField(default=None, alias="latestRunId")
    latest_run_status: Optional[str] = PydanticField(default=None, alias="latestRunStatus")
    extracted_fields: int = PydanticField(alias="extractedFields")
    documents_count: int = PydanticField(alias="documentsCount")
    last_updated_at: datetime = PydanticField(alias="lastUpdatedAt")
    versions: Optional[List[RunVersionItem]] = PydanticField(default=None, alias="versions")

    class Config:
        from_attributes = True
        allow_population_by_field_name = True


class RecentProcessList(BaseModel):
    items: List[RecentProcessItem] = PydanticField(alias="items")

    class Config:
        from_attributes = True
        allow_population_by_field_name = True


