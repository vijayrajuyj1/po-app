from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, and_, desc, or_
from sqlalchemy.ext.asyncio import AsyncSession

from apps.admin.schemas import RecentProcessItem, RecentProcessList, AdminFlagItem
from models.base import get_db
from models.extraction import Session as ExtractionSession, ExtractionRun, DocumentFile
from models.field_response import FieldResponse
from models.vendor import Vendor
from models.purchase_order import PurchaseOrder
from models.user import User
from models.po_update_flag import POUpdateFlag
from security.auth_backend import get_current_active_user, require_roles
from constants.roles import VALIDATOR, ADMIN

router = APIRouter(prefix="/api/validator", tags=["Validator Dashboard"])


@router.get(
    "/dashboard/recent",
    response_model=RecentProcessList,
    dependencies=[Depends(require_roles(VALIDATOR, ADMIN))],
)
async def recent_activity(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    search: Optional[str] = Query(default=None, description="Search by vendorId, vendorName, or poNumber"),
    statuses: List[str] = Query(
        default=["Processed", "To be verified"],
        alias="status",
        description="Repeatable: status=Processed&status=To%20be%20verified",
    ),
    include_versions: bool = Query(default=False, alias="includeVersions"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    fr_sub = (
        select(
            FieldResponse.session_id.label("session_id"),
            func.max(FieldResponse.updated_at).label("last_updated_at"),
            func.count(FieldResponse.id).label("fields_count"),
        )
        .group_by(FieldResponse.session_id)
        .subquery()
    )

    base_stmt = (
        select(
            ExtractionSession.id.label("session_id"),
            ExtractionSession.vendor_id.label("vendor_id"),
            Vendor.name.label("vendor_name"),
            ExtractionSession.po_id.label("po_id"),
            PurchaseOrder.number.label("po_number"),
            ExtractionSession.current_version.label("current_version"),
            fr_sub.c.last_updated_at.label("last_updated_at"),
            fr_sub.c.fields_count.label("fields_count"),
        )
        .join(fr_sub, fr_sub.c.session_id == ExtractionSession.id)
        .join(Vendor, Vendor.id == ExtractionSession.vendor_id)
        .join(PurchaseOrder, PurchaseOrder.id == ExtractionSession.po_id)
        .order_by(desc(fr_sub.c.last_updated_at))
        .offset(offset)
        .limit(limit)
    )

    if search:
        term = f"%{search.strip()}%"
        base_stmt = base_stmt.where(
            or_(ExtractionSession.vendor_id.ilike(term), Vendor.name.ilike(term), PurchaseOrder.number.ilike(term))
        )

    res = await db.execute(base_stmt)
    rows = list(res.all())

    items: List[RecentProcessItem] = []
    for row in rows:
        session_id = row.session_id
        latest_run_stmt = (
            select(ExtractionRun).where(ExtractionRun.session_id == session_id).order_by(ExtractionRun.version.desc()).limit(1)
        )
        latest_run_res = await db.execute(latest_run_stmt)
        latest_run = latest_run_res.scalar_one_or_none()

        if statuses and (latest_run is None or latest_run.status not in statuses):
            continue

        docs_count_stmt = select(func.count(DocumentFile.id)).where(
            and_(
                DocumentFile.session_id == session_id,
                DocumentFile.is_deleted.is_(False),
                DocumentFile.extraction_run_id == (latest_run.id if latest_run else None),
            )
        )
        docs_count_res = await db.execute(docs_count_stmt)
        docs_count = int(docs_count_res.scalar_one() or 0)

        versions = None
        if include_versions:
            runs_stmt = (
                select(ExtractionRun).where(ExtractionRun.session_id == session_id).order_by(ExtractionRun.version.asc())
            )
            runs_res = await db.execute(runs_stmt)
            runs = list(runs_res.scalars().all())
            versions = [
                {"runId": str(r.id), "version": r.version, "status": r.status, "createdAt": r.created_at}  # type: ignore[arg-type]
                for r in runs
            ]

        items.append(
            RecentProcessItem(
                sessionId=str(row.session_id),
                vendorId=str(row.vendor_id),
                vendorName=row.vendor_name,
                poId=str(row.po_id),
                poNumber=row.po_number,
                currentVersion=int(row.current_version),
                status=latest_run.status if latest_run else None,
                latestRunId=str(latest_run.id) if latest_run else None,
                latestRunStatus=latest_run.status if latest_run else None,
                extractedFields=int(row.fields_count or 0),
                documentsCount=docs_count,
                lastUpdatedAt=row.last_updated_at,  # type: ignore[arg-type]
                versions=versions,
            )
        )

    return RecentProcessList(items=items)


@router.get(
    "/dashboard/flags",
    response_model=List[AdminFlagItem],
    dependencies=[Depends(require_roles(VALIDATOR, ADMIN))],
)
async def flags_queue(
    status: Optional[str] = Query(default="open", description="Flag status filter (e.g., open)"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    include_versions: bool = Query(default=True, alias="includeVersions"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Validator view of flagged POs, mirroring Admin /dashboard/flags but
    restricted to flags whose related latest run status is in
    {'Processed', 'To be verified'}.
    """
    allowed_run_statuses = {"Processed", "To be verified"}

    where = []
    if status:
        where.append(POUpdateFlag.status == status)
    stmt = select(POUpdateFlag).where(and_(*where)) if where else select(POUpdateFlag)
    stmt = stmt.order_by(POUpdateFlag.created_at.desc()).offset(offset).limit(limit)
    res = await db.execute(stmt)
    flags = list(res.scalars().all())

    items: List[AdminFlagItem] = []
    for f in flags:
        run = await db.get(ExtractionRun, f.extraction_run_id) if f.extraction_run_id else None
        session = await db.get(ExtractionSession, run.session_id) if run else (await db.get(ExtractionSession, f.session_id) if f.session_id else None)
        if not run and session:
            latest_run_res = await db.execute(
                select(ExtractionRun).where(ExtractionRun.session_id == session.id).order_by(ExtractionRun.version.desc()).limit(1)
            )
            run = latest_run_res.scalar_one_or_none()

        # Filter by run status
        if not run or run.status not in allowed_run_statuses:
            continue

        vendor = await db.get(Vendor, session.vendor_id) if session else None
        po = await db.get(PurchaseOrder, session.po_id) if session else None

        flagged_by_name = None
        flagged_by_email = None
        if f.flagged_by:
            u = await db.get(User, f.flagged_by)
            flagged_by_name = u.name if u else None
            flagged_by_email = u.email if u else None

        versions = None
        if include_versions and session:
            runs_res = await db.execute(
                select(ExtractionRun).where(ExtractionRun.session_id == session.id).order_by(ExtractionRun.version.asc())
            )
            runs = list(runs_res.scalars().all())
            versions = [
                {"runId": str(r.id), "version": r.version, "status": r.status, "createdAt": r.created_at}  # type: ignore[arg-type]
                for r in runs
            ]

        items.append(
            AdminFlagItem(
                id=str(f.id),
                status=f.status,
                reason=f.reason,
                createdAt=f.created_at,  # type: ignore[arg-type]
                flaggedDate=f.created_at,  # type: ignore[arg-type]
                flaggedBy=flagged_by_name,
                flaggedByEmail=flagged_by_email,
                priority=None,
                adminNote=f.admin_note,
                sessionId=str(session.id) if session else None,
                runId=str(run.id) if run else None,
                runVersion=run.version if run else None,
                currentVersion=session.current_version if session else None,  # type: ignore[arg-type]
                versions=versions,
                vendorId=str(vendor.id) if vendor else None,
                vendorName=vendor.name if vendor else None,
                poId=str(po.id) if po else None,
                poNumber=po.number if po else None,
            )
        )

    return items


