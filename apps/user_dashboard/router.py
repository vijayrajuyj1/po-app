from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, and_, desc, or_
from sqlalchemy.ext.asyncio import AsyncSession

from apps.admin.schemas import RecentProcessItem, RecentProcessList
from models.base import get_db
from models.extraction import Session as ExtractionSession, ExtractionRun, DocumentFile
from models.field_response import FieldResponse
from models.vendor import Vendor
from models.purchase_order import PurchaseOrder
from models.user import User
from apps.responses.schemas import FieldResponseOut
from security.auth_backend import get_current_active_user, require_roles
from constants.roles import USER, ADMIN

router = APIRouter(prefix="/api/user", tags=["User Dashboard"])


@router.get(
    "/dashboard/recent",
    response_model=RecentProcessList,
    dependencies=[Depends(require_roles(USER, ADMIN))],
)
async def recent_verified(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    search: Optional[str] = Query(default=None, description="Search by vendorId, vendorName, or poNumber"),
    include_versions: bool = Query(default=False, alias="includeVersions"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    statuses = {"Verified"}

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

        if latest_run is None or latest_run.status not in statuses:
            continue

        docs_count_stmt = select(func.count(DocumentFile.id)).where(
            and_(
                DocumentFile.session_id == session_id,
                DocumentFile.is_deleted.is_(False),
                DocumentFile.extraction_run_id == latest_run.id,
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
                status=latest_run.status,
                latestRunId=str(latest_run.id),
                latestRunStatus=latest_run.status,
                extractedFields=int(row.fields_count or 0),
                documentsCount=docs_count,
                lastUpdatedAt=row.last_updated_at,  # type: ignore[arg-type]
                versions=versions,
            )
        )

    return RecentProcessList(items=items)


@router.get(
    "/dashboard/po/{po_number}/responses",
    response_model=List[FieldResponseOut],
    dependencies=[Depends(require_roles(USER, ADMIN))],
)
async def list_verified_field_responses_for_po(
    po_number: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Given a PO number, return field responses from the latest Verified run.
    Returns an empty list if the PO or a verified run is not found.
    """
    po_stmt = select(PurchaseOrder).where(PurchaseOrder.number == po_number, PurchaseOrder.is_deleted.is_(False))
    po_res = await db.execute(po_stmt)
    po = po_res.scalar_one_or_none()
    if not po:
        return []

    session_stmt = (
        select(ExtractionSession)
        .where(ExtractionSession.po_id == po.id, ExtractionSession.is_deleted.is_(False))
        .order_by(ExtractionSession.updated_at.desc())
        .limit(1)
    )
    session_res = await db.execute(session_stmt)
    session = session_res.scalar_one_or_none()
    
    if not session:
        return []

    run_stmt = (
        select(ExtractionRun)
        .where(ExtractionRun.session_id == session.id, ExtractionRun.status == "Verified")
        .order_by(ExtractionRun.version.desc())
        .limit(1)
    )
    run_res = await db.execute(run_stmt)
    run = run_res.scalar_one_or_none()
    if not run:
        return []

    fr_stmt = select(FieldResponse).where(FieldResponse.extraction_run_id == run.id)
    fr_res = await db.execute(fr_stmt)
    responses = list(fr_res.scalars().all())

    def _serialize(fr: FieldResponse) -> FieldResponseOut:
        return FieldResponseOut(
            id=str(fr.id),
            extractionRunId=str(fr.extraction_run_id),
            sessionId=str(fr.session_id),
            categoryId=str(fr.category_id),
            categoryName=fr.category.name if getattr(fr, "category", None) is not None else None,
            fieldId=str(fr.field_id),
            question=fr.question,
            answer=fr.answer,
            shortAnswer=fr.short_answer,
            confidenceScore=fr.confidence_score,
            citations=fr.citations,
            status=fr.status,
            isModified=fr.is_modified,
            modifiedBy=str(fr.modified_by) if fr.modified_by else None,
            modifiedAt=fr.modified_at,
            verifiedBy=str(fr.verified_by) if fr.verified_by else None,
            verifiedAt=fr.verified_at,
            createdAt=fr.created_at,
            updatedAt=fr.updated_at,
        )

    return [_serialize(fr) for fr in responses]

