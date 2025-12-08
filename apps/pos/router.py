from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from apps.pos.schemas import POCreate, POUpdate, POResponse
from apps.pos.service import POService
from models.base import get_db
from security.auth_backend import require_roles, get_current_active_user
from constants.roles import ADMIN
from models.extraction import Session as ExtractionSession, ExtractionRun
from models.vendor import Vendor


router = APIRouter(prefix="/api/pos", tags=["Purchase Orders"])


@router.get("/{po_id}", response_model=POResponse, dependencies=[Depends(get_current_active_user)])
async def get_po(po_id: str, db: AsyncSession = Depends(get_db)):
    po = await POService.get_po_by_id(db, po_id)

    vendor = await db.get(Vendor, po.vendor_id) if po.vendor_id else None

    # Try to fetch the latest session for this PO
    session_stmt = select(ExtractionSession).where(
        ExtractionSession.po_id == po.id,
        ExtractionSession.is_deleted.is_(False),
    ).order_by(ExtractionSession.updated_at.desc()).limit(1)
    session_res = await db.execute(session_stmt)
    session = session_res.scalar_one_or_none()

    latest_run_id = None
    latest_run_status = None
    if session:
        run_stmt = (
            select(ExtractionRun)
            .where(ExtractionRun.session_id == session.id)
            .order_by(ExtractionRun.version.desc())
            .limit(1)
        )
        run_res = await db.execute(run_stmt)
        run = run_res.scalar_one_or_none()
        latest_run_id = str(run.id) if run else None
        latest_run_status = run.status if run else None

    return POResponse(  # type: ignore[arg-type]
        id=po.id,
        number=po.number,
        status=latest_run_status,
        createdAt=po.created_at,
        vendorId=po.vendor_id,
        vendorName=vendor.name if vendor else None,
        sessionId=str(session.id) if session else None,
        latestRunId=latest_run_id,
    )


@router.post("", response_model=POResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_roles(ADMIN))])
async def create_po(payload: POCreate, db: AsyncSession = Depends(get_db)):
    po = await POService.create_po(db, payload)
    vendor = await db.get(Vendor, po.vendor_id) if po.vendor_id else None
    return POResponse(  # type: ignore[arg-type]
        id=po.id,
        number=po.number,
        status=None,
        createdAt=po.created_at,
        vendorId=po.vendor_id,
        vendorName=vendor.name if vendor else None,
        sessionId=None,
        latestRunId=None,
    )


@router.patch("/{po_id}", response_model=POResponse, dependencies=[Depends(require_roles(ADMIN))])
async def update_po(po_id: str, payload: POUpdate, db: AsyncSession = Depends(get_db)):
    po = await POService.update_po(db, po_id, payload)
    # Reuse the GET handler to ensure consistent derived status
    return await get_po(po.id, db)


@router.delete("/{po_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_roles(ADMIN))])
async def delete_po(po_id: str, db: AsyncSession = Depends(get_db)):
    await POService.soft_delete_po(db, po_id)
    return None


