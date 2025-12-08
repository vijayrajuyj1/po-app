import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from apps.flags.schemas import CreatePOFlagRequest, POFlagOut, UpdatePOFlagStatusRequest
from apps.extraction.service import ActivityLogService
from models.base import get_db
from models.po_update_flag import POUpdateFlag
from models.user import User
from security.auth_backend import get_current_active_user, require_roles
from constants.roles import ADMIN, VALIDATOR


router = APIRouter(prefix="/api/flags", tags=["PO Update Flags"])


def _serialize(flag: POUpdateFlag) -> POFlagOut:
    return POFlagOut(
        id=str(flag.id),
        runId=str(flag.extraction_run_id),
        sessionId=str(flag.session_id) if flag.session_id else None,
        reason=flag.reason,
        status=flag.status,
        flaggedBy=str(flag.flagged_by) if flag.flagged_by else None,
        createdAt=flag.created_at,  # type: ignore[arg-type]
        adminNote=flag.admin_note,
        resolvedBy=str(flag.resolved_by) if flag.resolved_by else None,
        resolvedAt=flag.resolved_at,  # type: ignore[arg-type]
    )


@router.post("/po-update", response_model=POFlagOut, status_code=status.HTTP_201_CREATED)
async def create_po_update_flag(
    payload: CreatePOFlagRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        run_id = uuid.UUID(payload.runId)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid runId") from exc

    session_uuid: Optional[uuid.UUID] = None
    if payload.sessionId:
        try:
            session_uuid = uuid.UUID(payload.sessionId)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid sessionId") from exc

    flag = POUpdateFlag(
        extraction_run_id=run_id,
        session_id=session_uuid,
        reason=payload.reason,
        status="open",
        flagged_by=current_user.id,
    )
    db.add(flag)
    await db.commit()
    await db.refresh(flag)

    await ActivityLogService.log_action(
        db,
        str(current_user.id),
        "PO_UPDATE_FLAG_CREATE",
        str(flag.session_id) if flag.session_id else None,
        str(flag.extraction_run_id),
        None,
        "open",
        {"flagId": str(flag.id)},
    )

    return _serialize(flag)


@router.get("/po-update", response_model=List[POFlagOut])
async def list_po_update_flags(
    sessionId: Optional[str] = Query(default=None),
    runId: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    where = []
    if sessionId:
        where.append(POUpdateFlag.session_id == uuid.UUID(sessionId))
    if runId:
        where.append(POUpdateFlag.extraction_run_id == uuid.UUID(runId))
    if status:
        where.append(POUpdateFlag.status == status)

    stmt = select(POUpdateFlag).where(and_(*where)) if where else select(POUpdateFlag)
    stmt = stmt.order_by(POUpdateFlag.created_at.desc()).offset(offset).limit(limit)
    res = await db.execute(stmt)
    items = res.scalars().all()
    return [_serialize(f) for f in items]


@router.patch(
    "/po-update/{flag_id}/status",
    response_model=POFlagOut,
    dependencies=[Depends(require_roles(VALIDATOR, ADMIN))],
)
async def update_po_flag_status(
    flag_id: str,
    payload: UpdatePOFlagStatusRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    flag = await db.get(POUpdateFlag, uuid.UUID(flag_id))
    if not flag:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Flag not found")

    old = flag.status
    flag.status = payload.status
    flag.admin_note = payload.adminNote or flag.admin_note
    if payload.status in {"resolved", "dismissed"}:
        from datetime import datetime, timezone
        flag.resolved_at = datetime.now(timezone.utc)
        flag.resolved_by = current_user.id

    await db.commit()
    await db.refresh(flag)

    await ActivityLogService.log_action(
        db,
        str(current_user.id),
        "PO_UPDATE_FLAG_STATUS",
        str(flag.session_id) if flag.session_id else None,
        str(flag.extraction_run_id),
        old,
        flag.status,
        {"flagId": str(flag.id)},
    )

    return _serialize(flag)


