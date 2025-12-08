import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from apps.extraction.schemas import (
    ProcessResponse,
    SessionResponse,
    RunResponse,
    RunFileResponse,
    VerifyRejectResponse,
    ActivityItem,
)
from apps.extraction.service import (
    SessionService,
    ExtractionRunService,
    FileService,
    ActivityLogService,
)
from apps.ai_engine.service import process_extraction_run_async
from models.base import get_db
from models.extraction import Session as ExtractionSession, ExtractionRun, DocumentFile, ActivityLog
from models.vendor import Vendor
from models.purchase_order import PurchaseOrder
from security.auth_backend import get_current_active_user, require_roles
from constants.roles import ADMIN, VALIDATOR, USER
from models.user import User

router = APIRouter(prefix="/api", tags=["Extraction Sessions"])


# Dedicated thread pool to offload heavy AI processing away from the event loop
EXECUTOR = ThreadPoolExecutor(max_workers=2)


def _run_extraction_sync(extraction_run_id: str) -> None:
    # Run the async workflow inside the worker thread's event loop
    asyncio.run(process_extraction_run_async(extraction_run_id))


def _parse_json_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    try:
        data = json.loads(value)
        if isinstance(data, list):
            return [str(x) for x in data]
        return []
    except Exception:
        return []


def _parse_json_obj(value: Optional[str]) -> Optional[dict]:
    if not value:
        return None
    try:
        data = json.loads(value)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _parse_selected_fields(value: Optional[str]) -> List[dict]:
    """
    Accepts JSON string representing either:
    - ["fieldId1", "fieldId2"]  (backward compatible)
    - [{"categoryId": "uuid", "fieldIds": ["uuid1", "uuid2"]}, ...]
    Returns a normalized list of {"categoryId": str | None, "fieldIds": [str, ...]}.
    """
    if not value:
        return []
    try:
        data = json.loads(value)
        if isinstance(data, list):
            if not data:
                return []
            if all(isinstance(item, str) for item in data):
                return [{"categoryId": None, "fieldIds": [str(x) for x in data]}]
            normalized: List[dict] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                cat = item.get("categoryId")
                fields = item.get("fieldIds") or []
                if isinstance(fields, list):
                    normalized.append({"categoryId": str(cat) if cat is not None else None, "fieldIds": [str(f) for f in fields]})
            return normalized
        return []
    except Exception:
        return []


@router.post("/sessions/process", response_model=ProcessResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_current_active_user)])
async def process_session(
    background_tasks: BackgroundTasks,
    po_id: str = Form(...),
    vendor_id: str = Form(...),
    selected_fields: Optional[str] = Form(default=None),
    previous_file_urls: Optional[str] = Form(default=None),
    metadata: Optional[str] = Form(default=None),
    files: Optional[List[UploadFile]] = File(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    # Suppress unused parameter warning; we offload via executor instead
    _ = background_tasks
    # Parse JSON string parameters
    # Normalize selected fields into list of {categoryId, fieldIds}
    selected_fields_list = _parse_selected_fields(selected_fields)
    previous_urls = _parse_json_list(previous_file_urls)
    metadata_obj = _parse_json_obj(metadata)
    print(previous_file_urls)
    # Create or retrieve session
    session, created = await SessionService.get_or_create_session(db, po_id, vendor_id, str(current_user.id))

    # Increment version and compute session_key
    session = await SessionService.increment_version(db, session, str(current_user.id))
    session = await SessionService.update_session_key(db, session)

    # Create new run for this version
    # Snapshot of incoming request for auditing/debug purposes
    request_snapshot = {
        "selectedFields": selected_fields_list,
        "previousFileUrls": previous_urls,
        "metadata": metadata_obj,
    }
    run = await ExtractionRunService.create_run(db, session, selected_fields_list, str(current_user.id), request_snapshot)

    # File handling: upload new files and/or reuse previous URLs
    uploaded = await FileService.upload_files_to_s3(db, files or [], session, run, str(current_user.id))
    reused = await FileService.reuse_previous_files(db, previous_urls or [], session, run, str(current_user.id))

    # Update run with file counts
    run = await ExtractionRunService.set_file_counts(db, run, len(uploaded), len(reused))

    # Log actions
    await ActivityLogService.log_action(
        db=db,
        user_id=str(current_user.id),
        action="PROCESS_RUN",
        session_id=str(session.id),
        run_id=str(run.id),
        status_from=None,
        status_to="Processing",
        metadata={"filesUploaded": len(uploaded), "filesReused": len(reused), "selectedFieldsCount": len(selected_fields_list)},
    )

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
        for f in (uploaded + reused)
    ]
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
    # Trigger AI processing in the background with status updates
    # background_tasks.add_task(process_extraction_run_async, str(run.id))

    # Trigger AI processing off the event loop using a thread pool executor
    loop = asyncio.get_running_loop()
    loop.run_in_executor(EXECUTOR, _run_extraction_sync, str(run.id))

    return ProcessResponse(session=session_out, run=run_out)


@router.get("/sessions/{session_id}", response_model=SessionResponse, dependencies=[Depends(get_current_active_user)])
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    session = await db.get(ExtractionSession, session_id)
    if not session or session.is_deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return SessionResponse(
        id=str(session.id),
        poId=session.po_id,  # type: ignore[arg-type]
        vendorId=session.vendor_id,  # type: ignore[arg-type]
        poNumber=session.po_number,  # type: ignore[arg-type]
        currentVersion=session.current_version,  # type: ignore[arg-type]
        sessionKey=session.session_key,  # type: ignore[arg-type]
        createdAt=session.created_at,  # type: ignore[arg-type]
        updatedAt=session.updated_at,  # type: ignore[arg-type]
    )


@router.get("/sessions/{session_id}/runs", response_model=List[RunResponse], dependencies=[Depends(get_current_active_user)])
async def list_runs(session_id: str, db: AsyncSession = Depends(get_db)):
    session = await db.get(ExtractionSession, session_id)
    if not session or session.is_deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    stmt = select(ExtractionRun).where(ExtractionRun.session_id == session.id).order_by(ExtractionRun.version.asc())
    res = await db.execute(stmt)
    runs = list(res.scalars().all())
    # Fetch files per run
    out: List[RunResponse] = []
    for r in runs:
        files_stmt = select(DocumentFile).where(DocumentFile.extraction_run_id == r.id, DocumentFile.is_deleted.is_(False))
        files_res = await db.execute(files_stmt)
        files = list(files_res.scalars().all())
        out.append(
            RunResponse(
                id=str(r.id),
                sessionId=str(session.id),
                version=r.version,
                status=r.status,
                selectedFields=r.selected_fields,  # type: ignore[arg-type]
                uploadedFilesCount=r.uploaded_files_count,  # type: ignore[arg-type]
                reusedFilesCount=r.reused_files_count,  # type: ignore[arg-type]
                createdAt=r.created_at,  # type: ignore[arg-type]
                files=[
                    RunFileResponse(
                        id=str(f.id),
                        filename=f.filename,
                        s3Url=f.s3_url,  # type: ignore[arg-type]
                        mimeType=f.mime_type,  # type: ignore[arg-type]
                        sizeBytes=f.size_bytes,  # type: ignore[arg-type]
                        createdAt=f.created_at,  # type: ignore[arg-type]
                        metadata=f.metadata_json,
                    )
                    for f in files
                ],
            )
        )
    return out


@router.get("/runs/{run_id}", response_model=RunResponse, dependencies=[Depends(get_current_active_user)])
async def get_run(run_id: str, db: AsyncSession = Depends(get_db)):
    run = await db.get(ExtractionRun, run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
    session = await db.get(ExtractionSession, run.session_id)
    vendor = await db.get(Vendor, session.vendor_id) if session else None
    po = await db.get(PurchaseOrder, session.po_id) if session else None
    files_stmt = select(DocumentFile).where(DocumentFile.extraction_run_id == run.id, DocumentFile.is_deleted.is_(False))
    files_res = await db.execute(files_stmt)
    files = list(files_res.scalars().all())
    return RunResponse(
        id=str(run.id),
        sessionId=str(run.session_id),
        version=run.version,
        status=run.status,
        poId=str(session.po_id) if session else None,  # type: ignore[arg-type]
        poNumber=session.po_number if session else None,  # type: ignore[arg-type]
        vendorId=str(session.vendor_id) if session else None,  # type: ignore[arg-type]
        vendorName=vendor.name if vendor else None,
        selectedFields=run.selected_fields,  # type: ignore[arg-type]
        uploadedFilesCount=run.uploaded_files_count,  # type: ignore[arg-type]
        reusedFilesCount=run.reused_files_count,  # type: ignore[arg-type]
        createdAt=run.created_at,  # type: ignore[arg-type]
        files=[
            RunFileResponse(
                id=str(f.id),
                filename=f.filename,
                s3Url=f.s3_url,  # type: ignore[arg-type]
                mimeType=f.mime_type,  # type: ignore[arg-type]
                sizeBytes=f.size_bytes,  # type: ignore[arg-type]
                createdAt=f.created_at,  # type: ignore[arg-type]
                metadata=f.metadata_json,
            )
            for f in files
        ],
    )


@router.post("/runs/{run_id}/verify", response_model=VerifyRejectResponse, dependencies=[Depends(require_roles(VALIDATOR))], tags=["Extraction Sessions", "Validator Dashboard"])
async def verify_run(run_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    run = await db.get(ExtractionRun, run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
    old = run.status
    run = await ExtractionRunService.verify_run(db, run, str(current_user.id))
    await ActivityLogService.log_action(db, str(current_user.id), "VERIFY_RUN", str(run.session_id), str(run.id), old, run.status, {})
    return VerifyRejectResponse(id=str(run.id), status=run.status)


@router.post("/runs/{run_id}/reject", response_model=VerifyRejectResponse, dependencies=[Depends(require_roles(VALIDATOR))])
async def reject_run(run_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    run = await db.get(ExtractionRun, run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
    old = run.status
    run = await ExtractionRunService.reject_run(db, run)
    await ActivityLogService.log_action(db, str(current_user.id), "REJECT_RUN", str(run.session_id), str(run.id), old, run.status, {})
    return VerifyRejectResponse(id=str(run.id), status=run.status)


@router.get("/sessions/{session_id}/files", response_model=List[RunFileResponse], dependencies=[Depends(get_current_active_user)])
async def list_files(session_id: str, version: Optional[int] = None, db: AsyncSession = Depends(get_db)):
    session = await db.get(ExtractionSession, session_id)
    if not session or session.is_deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    run_id = None
    if version is not None:
        run_stmt = select(ExtractionRun).where(ExtractionRun.session_id == session.id, ExtractionRun.version == version)
        run_res = await db.execute(run_stmt)
        run = run_res.scalar_one_or_none()
        if not run:
            return []
        run_id = run.id
    files_stmt = select(DocumentFile).where(DocumentFile.session_id == session.id, DocumentFile.is_deleted.is_(False))
    if run_id:
        files_stmt = files_stmt.where(DocumentFile.extraction_run_id == run_id)
    files_res = await db.execute(files_stmt)
    files = list(files_res.scalars().all())
    return [
        RunFileResponse(
            id=str(f.id),
            filename=f.filename,
            s3Url=f.s3_url,  # type: ignore[arg-type]
            mimeType=f.mime_type,  # type: ignore[arg-type]
            sizeBytes=f.size_bytes,  # type: ignore[arg-type]
            createdAt=f.created_at,  # type: ignore[arg-type]
            metadata=f.metadata_json,
        )
        for f in files
    ]


@router.delete("/files/{file_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_roles(ADMIN))])
async def delete_file(file_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    doc = await db.get(DocumentFile, file_id)
    if not doc or doc.is_deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
    doc.is_deleted = True
    doc.deleted_at = func.now()  # type: ignore[name-defined]
    await db.commit()
    await ActivityLogService.log_action(db, str(current_user.id), "DELETE_FILE", str(doc.session_id), str(doc.extraction_run_id), None, None, {"fileId": file_id})
    return None


@router.get("/activity", response_model=List[ActivityItem], dependencies=[Depends(require_roles(ADMIN))])
async def list_activity(db: AsyncSession = Depends(get_db)):
    stmt = select(ActivityLog).order_by(ActivityLog.created_at.desc()).limit(500)
    res = await db.execute(stmt)
    items = list(res.scalars().all())
    return [
        ActivityItem(
            id=str(a.id),
            userId=str(a.user_id) if a.user_id else None,  # type: ignore[arg-type]
            sessionId=str(a.session_id) if a.session_id else None,  # type: ignore[arg-type]
            runId=str(a.extraction_run_id) if a.extraction_run_id else None,  # type: ignore[arg-type]
            action=a.action,
            description=a.description,
            statusFrom=a.status_from,
            statusTo=a.status_to,
            metadata=a.metadata_json,
            createdAt=a.created_at,  # type: ignore[arg-type]
        )
        for a in items
    ]


