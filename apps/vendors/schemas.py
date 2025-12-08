from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field as PydanticField, constr, EmailStr
from apps.pos.schemas import POResponse


# -------------------------------
# Vendor schemas
# -------------------------------

class VendorMinimal(BaseModel):
    id: str = PydanticField(alias="id")
    name: str = PydanticField(alias="name")

    class Config:
        from_attributes = True
        allow_population_by_field_name = True


class VendorCreate(BaseModel):
    id: constr(strip_whitespace=True, min_length=1, max_length=50)
    name: constr(strip_whitespace=True, min_length=1, max_length=255)
    address: Optional[constr(strip_whitespace=True, min_length=1, max_length=500)] = None
    contact_email: Optional[EmailStr] = None
    phone: Optional[constr(strip_whitespace=True, min_length=1, max_length=50)] = None


class VendorUpdate(BaseModel):
    name: Optional[constr(strip_whitespace=True, min_length=1, max_length=255)] = None
    address: Optional[constr(strip_whitespace=True, min_length=1, max_length=500)] = None
    contact_email: Optional[EmailStr] = None
    phone: Optional[constr(strip_whitespace=True, min_length=1, max_length=50)] = None


class VendorResponse(BaseModel):
    id: str = PydanticField(alias="id")
    name: str = PydanticField(alias="name")
    address: Optional[str] = PydanticField(default=None, alias="address")
    contact_email: Optional[str] = PydanticField(default=None, alias="contactEmail")
    phone: Optional[str] = PydanticField(default=None, alias="phone")
    created_at: datetime = PydanticField(alias="createdAt")
    updated_at: datetime = PydanticField(alias="updatedAt")

    class Config:
        from_attributes = True
        allow_population_by_field_name = True


class VendorCheckResponse(BaseModel):
    exists: bool
    vendor: Optional[VendorMinimal] = None


class VendorByPOResponse(BaseModel):
    vendor: VendorMinimal
    po: POResponse


class VendorPOListResponse(BaseModel):
    vendor_id: str = PydanticField(alias="vendorId")
    vendor_name: str = PydanticField(alias="vendorName")
    pos: List[POResponse]
    pagination: dict

    class Config:
        from_attributes = True
        allow_population_by_field_name = True


class VendorListItem(BaseModel):
    id: str = PydanticField(alias="id")
    name: str = PydanticField(alias="name")
    contact_email: Optional[str] = PydanticField(default=None, alias="contactEmail")
    created_at: datetime = PydanticField(alias="createdAt")
    updated_at: datetime = PydanticField(alias="updatedAt")

    class Config:
        from_attributes = True
        allow_population_by_field_name = True


class VendorListResponse(BaseModel):
    items: List[VendorListItem]
    pagination: dict


