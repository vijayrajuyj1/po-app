import uuid
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func, and_, desc, or_
from sqlalchemy.ext.asyncio import AsyncSession

from apps.admin.schemas import AdminFlagItem, RecentProcessItem, RecentProcessList
from apps.extraction.schemas import ProcessResponse, SessionResponse, RunResponse, RunFileResponse
from apps.extraction.service import ActivityLogService, SessionService, ExtractionRunService, FileService
from constants.roles import ADMIN
from models.base import get_db
from models.po_update_flag import POUpdateFlag
from models.extraction import Session as ExtractionSession, ExtractionRun, DocumentFile
from models.field_response import FieldResponse
from models.vendor import Vendor
from models.purchase_order import PurchaseOrder
from security.auth_backend import get_current_active_user, require_roles
from models.user import User


router = APIRouter(prefix="/api/admin", tags=["Admin Dashboard"])


def _serialize_flag_item(
    flag: POUpdateFlag,
    run: Optional[ExtractionRun],
    session: Optional[ExtractionSession],
    vendor: Optional[Vendor],
    po: Optional[PurchaseOrder],
    flagged_by_name: Optional[str],
    flagged_by_email: Optional[str],
    versions: Optional[List[dict]],
) -> AdminFlagItem:
    return AdminFlagItem(
        id=str(flag.id),
        status=flag.status,
        reason=flag.reason,
        createdAt=flag.created_at,  # type: ignore[arg-type]
        flaggedDate=flag.created_at,  # type: ignore[arg-type]
        flaggedBy=flagged_by_name,
        flaggedByEmail=flagged_by_email,
        priority=None,
        adminNote=flag.admin_note,
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


@router.get(
    "/dashboard/flags",
    response_model=List[AdminFlagItem],
    dependencies=[Depends(require_roles(ADMIN))],
)
async def list_dashboard_flags(
    status_filter: Optional[str] = Query(default="open", alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    where = []
    if status_filter:
        where.append(POUpdateFlag.status == status_filter)

    stmt = select(POUpdateFlag).where(and_(*where)) if where else select(POUpdateFlag)
    stmt = stmt.order_by(POUpdateFlag.created_at.desc()).offset(offset).limit(limit)
    res = await db.execute(stmt)
    flags: List[POUpdateFlag] = list(res.scalars().all())
    print(stmt, flags)
    out: List[AdminFlagItem] = []
    for f in flags:
        run: Optional[ExtractionRun] = f.run if hasattr(f, "run") else None
        session: Optional[ExtractionSession] = f.session if hasattr(f, "session") else None

        if not session and run:
            # load session lazily if not present
            session = await db.get(ExtractionSession, run.session_id)

        vendor: Optional[Vendor] = None
        po: Optional[PurchaseOrder] = None
        flagged_by_name: Optional[str] = None
        flagged_by_email: Optional[str] = None
        versions: Optional[List[dict]] = None
        if session:
            vendor = await db.get(Vendor, session.vendor_id)
            po = await db.get(PurchaseOrder, session.po_id)
            # Build versions list
            runs_stmt = select(ExtractionRun).where(ExtractionRun.session_id == session.id).order_by(ExtractionRun.version.asc())
            runs_res = await db.execute(runs_stmt)
            runs = list(runs_res.scalars().all())
            if runs:
                versions = [
                    {"runId": str(r.id), "version": r.version, "status": r.status, "createdAt": r.created_at}  # type: ignore[arg-type]
                    for r in runs
                ]
        if f.flagged_by:
            u = await db.get(User, f.flagged_by)
            flagged_by_name = u.name if u else None
            flagged_by_email = u.email if u else None

        out.append(_serialize_flag_item(f, run, session, vendor, po, flagged_by_name, flagged_by_email, versions))

    return out


@router.post(
    "/dashboard/flags/{flag_id}/reprocess",
    response_model=ProcessResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_roles(ADMIN))],
    include_in_schema=False,
)
async def reprocess_flagged_po(
    flag_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    flag = await db.get(POUpdateFlag, uuid.UUID(flag_id))
    if not flag:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Flag not found")

    # Resolve session for reprocessing
    session: Optional[ExtractionSession] = flag.session
    if not session and flag.extraction_run_id:
        run = await db.get(ExtractionRun, flag.extraction_run_id)
        if not run:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found for flag")
        session = await db.get(ExtractionSession, run.session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Session not found for flag")

    # Find the latest run to clone configuration/files
    latest_run_stmt = (
        select(ExtractionRun)
        .where(ExtractionRun.session_id == session.id)
        .order_by(ExtractionRun.version.desc())
        .limit(1)
    )
    latest_run_res = await db.execute(latest_run_stmt)
    latest_run: Optional[ExtractionRun] = latest_run_res.scalar_one_or_none()

    selected_fields = latest_run.selected_fields if latest_run else None

    # Increment session version and create new run
    session = await SessionService.increment_version(db, session, str(current_user.id))
    session = await SessionService.update_session_key(db, session)
    run = await ExtractionRunService.create_run(
        db, session, selected_fields or [], str(current_user.id), {"reprocessFromFlagId": str(flag.id)}
    )

    # Reuse previous files (from latest run if exists)
    previous_urls: List[str] = []
    if latest_run:
        files_stmt = select(DocumentFile).where(
            and_(DocumentFile.extraction_run_id == latest_run.id, DocumentFile.is_deleted.is_(False))
        )
        files_res = await db.execute(files_stmt)
        files = list(files_res.scalars().all())
        previous_urls = [f.s3_url for f in files]

    uploaded = []  # no new files in admin reprocess
    reused = await FileService.reuse_previous_files(db, previous_urls, session, run, str(current_user.id))

    # Update file counts
    run = await ExtractionRunService.set_file_counts(db, run, len(uploaded), len(reused))

    # Mark flag as processing
    old_status = flag.status
    flag.status = "processing"
    await db.commit()
    await db.refresh(flag)

    # Log actions
    await ActivityLogService.log_action(
        db,
        str(current_user.id),
        "ADMIN_REPROCESS_START",
        str(session.id),
        str(run.id),
        None,
        "Processing",
        {"flagId": str(flag.id)},
    )
    await ActivityLogService.log_action(
        db,
        str(current_user.id),
        "PO_UPDATE_FLAG_STATUS",
        str(session.id),
        str(run.id),
        old_status,
        "processing",
        {"flagId": str(flag.id)},
    )

    # Build response
    files_out = [
        RunFileResponse(
            id=str(f.id),
            filename=f.filename,
            s3Url=f.s3_url,  # type: ignore[arg-type]
            mimeType=f.mime_type,  # type: ignore[arg-type]
            sizeBytes=f.size_bytes,  # type: ignore[arg-type]
            createdAt=f.created_at,  # type: ignore[arg-type]
            metadata=f.metadata_json,
        )
        for f in reused
    ]

    session_out = SessionResponse(
        id=str(session.id),
        poId=session.po_id,  # type: ignore[arg-type]
        vendorId=session.vendor_id,  # type: ignore[arg-type]
        poNumber=session.po_number,  # type: ignore[arg-type]
        currentVersion=session.current_version,  # type: ignore[arg-type]
        sessionKey=session.session_key,  # type: ignore[arg-type]
        createdAt=session.created_at,  # type: ignore[arg-type]
        updatedAt=session.updated_at,  # type: ignore[arg-type]
    )

    run_out = RunResponse(
        id=str(run.id),
        sessionId=str(session.id),
        version=run.version,
        status=run.status,
        selectedFields=run.selected_fields,  # type: ignore[arg-type]
        uploadedFilesCount=run.uploaded_files_count,  # type: ignore[arg-type]
        reusedFilesCount=run.reused_files_count,  # type: ignore[arg-type]
        createdAt=run.created_at,  # type: ignore[arg-type]
        files=files_out,
    )

    return ProcessResponse(session=session_out, run=run_out)


@router.get(
    "/dashboard/recent",
    response_model=RecentProcessList,
    dependencies=[Depends(require_roles(ADMIN))],
)
async def recent_activity(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    search: Optional[str] = Query(default=None, description="Search by vendorId, vendorName, or poNumber"),
    statuses: List[str] = Query(default=[], alias="status", description="Repeatable: status=Verified&status=Processing"),
    include_versions: bool = Query(default=False, alias="includeVersions"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    # Subquery: latest field_response update per session
    fr_sub = (
        select(
            FieldResponse.session_id.label("session_id"),
            func.max(FieldResponse.updated_at).label("last_updated_at"),
            func.count(FieldResponse.id).label("fields_count"),
        )
        .group_by(FieldResponse.session_id)
        .subquery()
    )

    # Join with sessions and vendors/POs
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

    # Optional search filter
    if search:
        term = f"%{search.strip()}%"
        base_stmt = base_stmt.where(
            or_(
                ExtractionSession.vendor_id.ilike(term),
                Vendor.name.ilike(term),
                PurchaseOrder.number.ilike(term),
            )
        )

    res = await db.execute(base_stmt)
    rows = list(res.all())

    items: List[RecentProcessItem] = []
    for row in rows:
        session_id = row.session_id
        # latest run for the session
        latest_run_stmt = (
            select(ExtractionRun)
            .where(ExtractionRun.session_id == session_id)
            .order_by(ExtractionRun.version.desc())
            .limit(1)
        )
        latest_run_res = await db.execute(latest_run_stmt)
        latest_run: Optional[ExtractionRun] = latest_run_res.scalar_one_or_none()

        # Status filter against latest run status if provided
        if statuses and (latest_run is None or latest_run.status not in statuses):
            continue

        # docs count for latest run
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
                select(ExtractionRun)
                .where(ExtractionRun.session_id == session_id)
                .order_by(ExtractionRun.version.asc())
            )
            runs_res = await db.execute(runs_stmt)
            runs = list(runs_res.scalars().all())
            versions = [
                {
                    "runId": str(r.id),
                    "version": r.version,
                    "status": r.status,
                    "createdAt": r.created_at,  # type: ignore[arg-type]
                }
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


