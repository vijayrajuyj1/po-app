import asyncio
import json
import uuid
from typing import List, Optional, Tuple

import boto3
from fastapi import HTTPException, UploadFile, status
from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from urllib.parse import urlparse, unquote

from apps.extraction.schemas import ProcessRequest
from models.extraction import Session as ExtractionSession, ExtractionRun, DocumentFile, ActivityLog, EXTRACTION_STATUS_VALUES
from models.purchase_order import PurchaseOrder
from settings.config import get_settings


class S3StorageService:
    @staticmethod
    def _get_client():
        settings = get_settings()
        # Prefer explicit credentials from settings when provided; otherwise let boto3 resolve env/instance profile.
        kwargs: dict = {}
        if getattr(settings, "AWS_REGION", None):
            kwargs["region_name"] = settings.AWS_REGION
        if getattr(settings, "AWS_ACCESS_KEY_ID", None) and getattr(settings, "AWS_SECRET_ACCESS_KEY", None):
            kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
            kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY

        return boto3.client("s3", **kwargs)

    @staticmethod
    async def upload_file(file: UploadFile, path: str) -> str:
        """
        Upload a file object to S3 at the given path. Returns the S3 URL.
        Offload blocking I/O to a thread.
        """
        client = S3StorageService._get_client()
        bucket = get_settings().AWS_S3_BUCKET  # must be configured in settings
        content = await file.read()

        def _upload():
            client.put_object(Bucket=bucket, Key=path, Body=content, ContentType=file.content_type or "application/octet-stream")
            # Construct URL using virtual-hosted–style URL
            region = client.meta.region_name or "us-east-1"
            return f"https://{bucket}.s3.{region}.amazonaws.com/{path}"

        return await asyncio.to_thread(_upload)


class ActivityLogService:
    @staticmethod
    async def log_action(
        db: AsyncSession,
        user_id: Optional[str],
        action: str,
        session_id: Optional[str],
        run_id: Optional[str],
        status_from: Optional[str],
        status_to: Optional[str],
        metadata: Optional[dict] = None,
    ) -> None:
        log = ActivityLog(
            user_id=uuid.UUID(str(user_id)) if user_id else None,
            session_id=uuid.UUID(str(session_id)) if session_id else None,
            extraction_run_id=uuid.UUID(str(run_id)) if run_id else None,
            action=action,
            description=metadata.get("description") if metadata else None,
            status_from=status_from,
            status_to=status_to,
            metadata_json=metadata or None,
            extra_args={},
        )
        db.add(log)
        await db.commit()


class SessionService:
    @staticmethod
    async def get_or_create_session(
        db: AsyncSession,
        po_id: str,
        vendor_id: str,
        current_user_id: Optional[str],
    ) -> Tuple[ExtractionSession, bool]:
        """
        Returns (session, created)
        """
        stmt = select(ExtractionSession).where(
            and_(ExtractionSession.po_id == po_id, ExtractionSession.is_deleted.is_(False))
        )
        res = await db.execute(stmt)
        session = res.scalar_one_or_none()
        if session:
            return session, False

        # Optionally fetch po_number for session_key
        po_number: Optional[str] = None
        po_res = await db.execute(select(PurchaseOrder).where(PurchaseOrder.id == po_id, PurchaseOrder.is_deleted.is_(False)))
        po = po_res.scalar_one_or_none()
        if po:
            po_number = po.number

        session = ExtractionSession(
            po_id=po_id,
            vendor_id=vendor_id,
            po_number=po_number,
            current_version=0,
            session_key=None,
            metadata_json=None,
            extra_args={},
            created_by=uuid.UUID(str(current_user_id)) if current_user_id else None,
            updated_by=uuid.UUID(str(current_user_id)) if current_user_id else None,
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
        return session, True

    @staticmethod
    async def increment_version(db: AsyncSession, session: ExtractionSession, current_user_id: Optional[str]) -> ExtractionSession:
        session.current_version = int(session.current_version or 0) + 1
        session.updated_by = uuid.UUID(str(current_user_id)) if current_user_id else None
        await db.commit()
        await db.refresh(session)
        return session

    @staticmethod
    async def update_session_key(db: AsyncSession, session: ExtractionSession) -> ExtractionSession:
        po_number = session.po_number or session.po_id
        session.session_key = f"{po_number}::v{session.current_version}"
        await db.commit()
        await db.refresh(session)
        return session


class ExtractionRunService:
    @staticmethod
    async def create_run(
        db: AsyncSession,
        session: ExtractionSession,
        selected_fields: Optional[List[str]],
        current_user_id: Optional[str],
        request_json: Optional[dict] = None,
    ) -> ExtractionRun:
        run = ExtractionRun(
            session_id=session.id,
            version=session.current_version,
            status="Processing",
            selected_fields=list(selected_fields or []),
            metadata_json=None,
            request_json=request_json or None,
            uploaded_files_count=0,
            reused_files_count=0,
            extra_args={},
            created_by=uuid.UUID(str(current_user_id)) if current_user_id else None,
            verified_by=None,
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)
        return run

    @staticmethod
    async def update_status(db: AsyncSession, run: ExtractionRun, new_status: str) -> ExtractionRun:
        if new_status not in EXTRACTION_STATUS_VALUES:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid status value.")
        old = run.status
        run.status = new_status
        await db.commit()
        await db.refresh(run)
        return run

    @staticmethod
    async def verify_run(db: AsyncSession, run: ExtractionRun, user_id: Optional[str]) -> ExtractionRun:
        run.status = "Verified"
        run.verified_by = uuid.UUID(str(user_id)) if user_id else None
        await db.commit()
        await db.refresh(run)
        return run

    @staticmethod
    async def reject_run(db: AsyncSession, run: ExtractionRun) -> ExtractionRun:
        run.status = "Failed"
        await db.commit()
        await db.refresh(run)
        return run

    @staticmethod
    async def set_file_counts(db: AsyncSession, run: ExtractionRun, uploaded_count: int, reused_count: int) -> ExtractionRun:
        run.uploaded_files_count = int(uploaded_count or 0)
        run.reused_files_count = int(reused_count or 0)
        await db.commit()
        await db.refresh(run)
        return run


class FileService:
    @staticmethod
    async def upload_files_to_s3(
        db: AsyncSession,
        files: List[UploadFile],
        session: ExtractionSession,
        run: ExtractionRun,
        user_id: Optional[str],
    ) -> List[DocumentFile]:
        if not files:
            return []
        uploaded: List[DocumentFile] = []
        for f in files:
            path = f"contracts/{session.id}/runs/{run.version}/{uuid.uuid4()}_{f.filename}"
            s3_url = await S3StorageService.upload_file(f, path)
            doc = DocumentFile(
                session_id=session.id,
                extraction_run_id=run.id,
                filename=f.filename,
                s3_url=s3_url,
                mime_type=f.content_type or "application/octet-stream",
                size_bytes=len(await f.read()) if hasattr(f, "file") else 0,
                uploaded_by=uuid.UUID(str(user_id)) if user_id else None,
                metadata_json={"reused": False},
                extra_args={},
            )
            db.add(doc)
            uploaded.append(doc)
        await db.commit()
        # refresh all with ids
        for d in uploaded:
            await db.refresh(d)
        return uploaded

    @staticmethod
    async def reuse_previous_files(
        db: AsyncSession,
        urls: List[str],
        session: ExtractionSession,
        run: ExtractionRun,
        user_id: Optional[str],
    ) -> List[DocumentFile]:
        reused: List[DocumentFile] = []
        for url in urls or []:
            # Try to recover original filename from our S3 key pattern "{uuid}_{filename}"
            path = urlparse(url).path or ""
            last = unquote(path.split("/")[-1]) if path else ""
            if "_" in last:
                original_name = last.split("_", 1)[1] or last
            else:
                original_name = last or "document.pdf"
            doc = DocumentFile(
                session_id=session.id,
                extraction_run_id=run.id,
                filename=original_name,
                s3_url=url,
                mime_type="application/pdf",
                size_bytes=0,
                uploaded_by=uuid.UUID(str(user_id)) if user_id else None,
                metadata_json={"reused": True, "source": url},
                extra_args={},
            )
            db.add(doc)
            reused.append(doc)
        await db.commit()
        for d in reused:
            await db.refresh(d)
        return reused


