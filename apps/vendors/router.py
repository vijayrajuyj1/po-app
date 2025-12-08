from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from apps.vendors.schemas import (
    VendorCreate,
    VendorUpdate,
    VendorResponse,
    VendorCheckResponse,
    VendorMinimal,
    VendorByPOResponse,
    VendorPOListResponse,
    VendorListItem,
    VendorListResponse,
)
from apps.pos.schemas import POResponse
from apps.pos.service import POService
from apps.vendors.service import VendorService
from models.base import get_db
from security.auth_backend import require_roles, get_current_active_user
from constants.roles import ADMIN


router = APIRouter(prefix="/api/vendors", tags=["Vendors"])


# 3.1 Check Vendor Exists
@router.get("/check/{vendor_id}", response_model=VendorCheckResponse, dependencies=[Depends(get_current_active_user)])
async def check_vendor_exists(vendor_id: str, db: AsyncSession = Depends(get_db)):
    vendor = await VendorService.check_vendor_exists(db, vendor_id)
    if not vendor:
        return VendorCheckResponse(exists=False, vendor=None)
    return VendorCheckResponse(exists=True, vendor=VendorMinimal(id=vendor.id, name=vendor.name))


# 3.2 Get Vendor by ID
@router.get("/{vendor_id}", response_model=VendorResponse, dependencies=[Depends(get_current_active_user)])
async def get_vendor(vendor_id: str, db: AsyncSession = Depends(get_db)):
    vendor = await VendorService.get_vendor(db, vendor_id)
    return VendorResponse(
        id=vendor.id,
        name=vendor.name,
        address=vendor.address,
        contactEmail=vendor.contact_email,  # type: ignore[arg-type]
        phone=vendor.phone,
        createdAt=vendor.created_at,  # type: ignore[arg-type]
        updatedAt=vendor.updated_at,  # type: ignore[arg-type]
    )


# 3.3 Get Vendor by PO Number
@router.get("/by-po/{po_number}", response_model=VendorByPOResponse, dependencies=[Depends(get_current_active_user)])
async def get_vendor_by_po_number(po_number: str, db: AsyncSession = Depends(get_db)):
    vendor, po = await POService.get_vendor_by_po_number(db, po_number)
    return VendorByPOResponse(
        vendor=VendorMinimal(id=vendor.id, name=vendor.name),
        po=POResponse(id=po.id, number=po.number, status=po.status, createdAt=po.created_at),  # type: ignore[arg-type]
    )


# 3.4 Get Vendor's PO List
@router.get("/{vendor_id}/pos", response_model=VendorPOListResponse, dependencies=[Depends(get_current_active_user)])
async def list_vendor_pos(
    vendor_id: str,
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    status: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    items, pagination = await POService.list_vendor_pos(db, vendor_id, page, size, status)
    pos = [
        POResponse(id=i.id, number=i.number, status=i.status, createdAt=i.created_at)  # type: ignore[arg-type]
        for i in items
    ]
    vendor = await VendorService.get_vendor(db, vendor_id)
    return VendorPOListResponse(vendorId=vendor.id, vendorName=vendor.name, pos=pos, pagination=pagination)  # type: ignore[arg-type]


# 3.5 Create New Vendor (Admin Only)
@router.post("", response_model=VendorMinimal, status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_roles(ADMIN))])
async def create_vendor(payload: VendorCreate, db: AsyncSession = Depends(get_db)):
    vendor = await VendorService.create_vendor(db, payload)
    return VendorMinimal(id=vendor.id, name=vendor.name)


# Update Vendor (Admin Only)
@router.patch("/{vendor_id}", response_model=VendorResponse, dependencies=[Depends(require_roles(ADMIN))])
async def update_vendor(vendor_id: str, payload: VendorUpdate, db: AsyncSession = Depends(get_db)):
    vendor = await VendorService.update_vendor(db, vendor_id, payload)
    return VendorResponse(
        id=vendor.id,
        name=vendor.name,
        address=vendor.address,
        contactEmail=vendor.contact_email,  # type: ignore[arg-type]
        phone=vendor.phone,
        createdAt=vendor.created_at,  # type: ignore[arg-type]
        updatedAt=vendor.updated_at,  # type: ignore[arg-type]
    )


# Soft Delete Vendor (Admin Only)
@router.delete("/{vendor_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_roles(ADMIN))])
async def delete_vendor(vendor_id: str, db: AsyncSession = Depends(get_db)):
    await VendorService.soft_delete_vendor(db, vendor_id)
    return None


# Restore Vendor (Admin Only)
@router.patch("/{vendor_id}/restore", response_model=VendorMinimal, dependencies=[Depends(require_roles(ADMIN))])
async def restore_vendor(vendor_id: str, db: AsyncSession = Depends(get_db)):
    vendor = await VendorService.restore_vendor(db, vendor_id)
    return VendorMinimal(id=vendor.id, name=vendor.name)


# Get All Vendors (List + Search + Paginate)
@router.get("", response_model=VendorListResponse, dependencies=[Depends(get_current_active_user)])
async def list_vendors(
    search: Optional[str] = Query(default=None, description="Search by name or email"),
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    sort_by: Optional[str] = Query(default=None, description="Field to sort by (e.g., name, created_at)"),
    sort_order: str = Query(default="asc", pattern="^(?i)(asc|desc)$"),
    include_deleted: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
):
    items, pagination = await VendorService.list_vendors(db, page, size, search, sort_by, sort_order, include_deleted)
    mapped = [
        VendorListItem(
            id=v.id,
            name=v.name,
            contactEmail=v.contact_email,  # type: ignore[arg-type]
            createdAt=v.created_at,  # type: ignore[arg-type]
            updatedAt=v.updated_at,  # type: ignore[arg-type]
        )
        for v in items
    ]
    return VendorListResponse(items=mapped, pagination=pagination)





