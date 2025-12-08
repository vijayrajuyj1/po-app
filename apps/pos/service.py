from typing import List, Optional, Tuple
from fastapi import HTTPException, status
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.vendors.pagination import paginate_select
from apps.pos.schemas import POCreate, POUpdate
from models.vendor import Vendor
from models.purchase_order import PurchaseOrder


class POService:
    @staticmethod
    async def create_po(db: AsyncSession, payload: POCreate) -> PurchaseOrder:
        vendor_stmt = select(Vendor).where(and_(Vendor.id == payload.vendor_id, Vendor.is_deleted.is_(False)))
        vendor_res = await db.execute(vendor_stmt)
        if not vendor_res.scalar_one_or_none():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vendor not found.")

        number_stmt = select(PurchaseOrder).where(PurchaseOrder.number == payload.number)
        number_res = await db.execute(number_stmt)
        if number_res.scalar_one_or_none():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="PO number must be unique.")

        po = PurchaseOrder(
            id=payload.id.strip(),
            number=payload.number.strip(),
            vendor_id=payload.vendor_id.strip(),
        )
        db.add(po)
        await db.commit()
        await db.refresh(po)
        return po

    @staticmethod
    async def get_po_by_id(db: AsyncSession, po_id: str) -> PurchaseOrder:
        stmt = select(PurchaseOrder).where(and_(PurchaseOrder.id == po_id, PurchaseOrder.is_deleted.is_(False)))
        res = await db.execute(stmt)
        po = res.scalar_one_or_none()
        if not po:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PO not found.")
        return po

    @staticmethod
    async def update_po(db: AsyncSession, po_id: str, payload: POUpdate) -> PurchaseOrder:
        po = await POService.get_po_by_id(db, po_id)
        # Currently, nothing updatable here since status has been removed.
        await db.commit()
        await db.refresh(po)
        return po

    @staticmethod
    async def soft_delete_po(db: AsyncSession, po_id: str) -> None:
        po = await POService.get_po_by_id(db, po_id)
        if po.is_deleted:
            return
        po.is_deleted = True
        po.deleted_at = func.now()
        await db.commit()

    @staticmethod
    async def get_vendor_by_po_number(db: AsyncSession, po_number: str) -> Tuple[Vendor, PurchaseOrder]:
        po_stmt = select(PurchaseOrder).where(
            and_(PurchaseOrder.number == po_number, PurchaseOrder.is_deleted.is_(False))
        )
        po_res = await db.execute(po_stmt)
        po = po_res.scalar_one_or_none()
        if not po:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PO not found.")

        vendor_stmt = select(Vendor).where(and_(Vendor.id == po.vendor_id, Vendor.is_deleted.is_(False)))
        vendor_res = await db.execute(vendor_stmt)
        vendor = vendor_res.scalar_one_or_none()
        if not vendor:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vendor not found.")
        return vendor, po

    @staticmethod
    async def list_vendor_pos(
        db: AsyncSession,
        vendor_id: str,
        page: int,
        size: int,
        status_filter: Optional[str] = None,
    ) -> Tuple[List[PurchaseOrder], dict]:
        # Ensure vendor exists and is not deleted
        vendor_stmt = select(Vendor).where(and_(Vendor.id == vendor_id, Vendor.is_deleted.is_(False)))
        vendor_res = await db.execute(vendor_stmt)
        if not vendor_res.scalar_one_or_none():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vendor not found.")

        where_clause = [PurchaseOrder.vendor_id == vendor_id, PurchaseOrder.is_deleted.is_(False)]
        # status_filter ignored since PurchaseOrder no longer stores status.

        base_stmt = select(PurchaseOrder).where(and_(*where_clause)).order_by(PurchaseOrder.created_at.desc())
        count_stmt = select(func.count()).select_from(
            select(PurchaseOrder.id).where(and_(*where_clause)).subquery()
        )
        items, pagination = await paginate_select(db, base_stmt, count_stmt, page, size)
        return items, pagination


