from typing import List, Optional, Tuple
from fastapi import HTTPException, status
from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from apps.vendors.pagination import paginate_select
from apps.vendors.schemas import VendorCreate, VendorUpdate
from models.vendor import Vendor
from models.purchase_order import PurchaseOrder


class VendorService:
    @staticmethod
    async def check_vendor_exists(db: AsyncSession, vendor_id: str) -> Optional[Vendor]:
        stmt = select(Vendor).where(and_(Vendor.id == vendor_id, Vendor.is_deleted.is_(False)))
        res = await db.execute(stmt)
        return res.scalar_one_or_none()

    @staticmethod
    async def get_vendor(db: AsyncSession, vendor_id: str) -> Vendor:
        vendor = await VendorService.check_vendor_exists(db, vendor_id)
        if not vendor:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vendor not found.")
        return vendor

    @staticmethod
    async def get_vendor_by_po_number(db: AsyncSession, po_number: str) -> Tuple[Vendor, PurchaseOrder]:
        # Moved to POService in apps.pos.service; kept temporarily for backward compatibility if imported elsewhere.
        po_stmt = select(PurchaseOrder).where(
            and_(PurchaseOrder.number == po_number, PurchaseOrder.is_deleted.is_(False))
        )
        po_res = await db.execute(po_stmt)
        po = po_res.scalar_one_or_none()
        if not po:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PO not found.")
        vendor = await VendorService.get_vendor(db, po.vendor_id)
        return vendor, po

    @staticmethod
    async def list_vendor_pos(
        db: AsyncSession,
        vendor_id: str,
        page: int,
        size: int,
        status_filter: Optional[str] = None,
    ) -> Tuple[List[PurchaseOrder], dict]:
        # Moved to POService in apps.pos.service; kept temporarily for backward compatibility if imported elsewhere.
        where_clause = [PurchaseOrder.vendor_id == vendor_id, PurchaseOrder.is_deleted.is_(False)]
        if status_filter:
            where_clause.append(PurchaseOrder.status == status_filter)
        base_stmt = select(PurchaseOrder).where(and_(*where_clause)).order_by(PurchaseOrder.created_at.desc())
        count_stmt = select(func.count()).select_from(
            select(PurchaseOrder.id).where(and_(*where_clause)).subquery()
        )
        items, pagination = await paginate_select(db, base_stmt, count_stmt, page, size)
        return items, pagination

    @staticmethod
    async def create_vendor(db: AsyncSession, payload: VendorCreate) -> Vendor:
        # ID must be unique and immutable; check both active and deleted to avoid collisions
        exists_stmt = select(Vendor).where(Vendor.id == payload.id)
        exists_res = await db.execute(exists_stmt)
        if exists_res.scalar_one_or_none():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Vendor ID already exists.")

        vendor = Vendor(
            id=payload.id.strip(),
            name=payload.name.strip(),
            address=payload.address.strip() if payload.address else None,
            contact_email=payload.contact_email,
            phone=payload.phone.strip() if payload.phone else None,
        )
        db.add(vendor)
        await db.commit()
        await db.refresh(vendor)
        return vendor

    @staticmethod
    async def update_vendor(db: AsyncSession, vendor_id: str, payload: VendorUpdate) -> Vendor:
        vendor = await VendorService.get_vendor(db, vendor_id)
        if payload.name is not None:
            vendor.name = payload.name.strip()
        if payload.address is not None:
            vendor.address = payload.address.strip() if payload.address else None
        if payload.contact_email is not None:
            vendor.contact_email = payload.contact_email
        if payload.phone is not None:
            vendor.phone = payload.phone.strip() if payload.phone else None
        await db.commit()
        await db.refresh(vendor)
        return vendor

    @staticmethod
    async def soft_delete_vendor(db: AsyncSession, vendor_id: str) -> None:
        vendor = await VendorService.get_vendor(db, vendor_id)
        if vendor.is_deleted:
            return
        vendor.is_deleted = True
        vendor.deleted_at = func.now()
        await db.commit()

    @staticmethod
    async def restore_vendor(db: AsyncSession, vendor_id: str) -> Vendor:
        # Allow restoring a soft-deleted vendor
        stmt = select(Vendor).where(and_(Vendor.id == vendor_id, Vendor.is_deleted.is_(True)))
        res = await db.execute(stmt)
        vendor = res.scalar_one_or_none()
        if not vendor:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vendor not found or not deleted.")
        vendor.is_deleted = False
        vendor.deleted_at = None
        await db.commit()
        await db.refresh(vendor)
        return vendor

    @staticmethod
    async def list_vendors(
        db: AsyncSession,
        page: int,
        size: int,
        search: Optional[str],
        sort_by: Optional[str],
        sort_order: str,
        include_deleted: bool = False,
    ) -> Tuple[List[Vendor], dict]:
        where_clause = []
        if not include_deleted:
            where_clause.append(Vendor.is_deleted.is_(False))
        if search:
            s = f"%{search.lower()}%"
            where_clause.append(or_(func.lower(Vendor.name).like(s), func.lower(Vendor.contact_email).like(s)))

        stmt = select(Vendor)
        if where_clause:
            stmt = stmt.where(and_(*where_clause))

        # Sorting
        if sort_by and hasattr(Vendor, sort_by):
            col = getattr(Vendor, sort_by)
            if (sort_order or "asc").lower() == "desc":
                stmt = stmt.order_by(col.desc())
            else:
                stmt = stmt.order_by(col.asc())
        else:
            stmt = stmt.order_by(Vendor.created_at.desc())

        count_stmt = select(func.count()).select_from(stmt.with_only_columns(Vendor.id).subquery())
        items, pagination = await paginate_select(db, stmt, count_stmt, page, size)
        return items, pagination





