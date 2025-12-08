from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field as PydanticField, constr


class POCreate(BaseModel):
    id: constr(strip_whitespace=True, min_length=1, max_length=50)
    number: constr(strip_whitespace=True, min_length=1, max_length=100)
    vendor_id: constr(strip_whitespace=True, min_length=1, max_length=50)


class POUpdate(BaseModel):
    pass


class POResponse(BaseModel):
    id: str = PydanticField(alias="id")
    number: str = PydanticField(alias="number")
    status: Optional[str] = PydanticField(default=None, alias="status")
    created_at: datetime = PydanticField(alias="createdAt")
    vendor_id: Optional[str] = PydanticField(default=None, alias="vendorId")
    vendor_name: Optional[str] = PydanticField(default=None, alias="vendorName")
    session_id: Optional[str] = PydanticField(default=None, alias="sessionId")
    latest_run_id: Optional[str] = PydanticField(default=None, alias="latestRunId")

    class Config:
        from_attributes = True
        allow_population_by_field_name = True


