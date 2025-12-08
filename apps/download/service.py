from typing import List
import csv
import io

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.download.schemas import SimpleFieldResponseOut, POResponsesRequest
from constants.roles import ADMIN, VALIDATOR
from models.extraction import Session as ExtractionSession, ExtractionRun
from models.field_response import FieldResponse
from models.purchase_order import PurchaseOrder
from models.user import User


async def session_exists_for_po(po_number: str, db: AsyncSession) -> bool:
    """
    Check whether a non-deleted session exists for the given PO number.
    """
    stmt = (
        select(ExtractionSession)
        .where(ExtractionSession.po_number == po_number, ExtractionSession.is_deleted.is_(False))
        .limit(1)
    )
    res = await db.execute(stmt)
    session = res.scalar_one_or_none()
    return bool(session)


async def generate_po_responses(
    payload: POResponsesRequest,
    db: AsyncSession,
    current_user: User,
):
    """
    Core business logic to fetch field responses for the given POs and
    return them in the requested format (json/csv/excel).
    """
    po_numbers = payload.po_numbers or []
    get_status = payload.get_status or ["Processed", "Verified"]
    response_type = payload.response_type or "csv"

    # 1) Find all POs by number (non-deleted only)
    po_stmt = select(PurchaseOrder).where(
        PurchaseOrder.number.in_([n for n in po_numbers if n]),
        PurchaseOrder.is_deleted.is_(False),
    )
    po_res = await db.execute(po_stmt)
    pos = list(po_res.scalars().all())
    if not pos:
        # Return an empty structure based on response_type
        if (response_type or "csv").lower() == "json":
            return []
        # For csv/excel we'll still return a CSV with only headers further below

    # 2) Work out which run statuses are allowed for this caller
    role_names = {r.name for r in (getattr(current_user, "roles", []) or [])}
    admin_or_validator = bool(role_names.intersection({ADMIN, VALIDATOR}))

    if admin_or_validator:
        # Normalize requested statuses to match Enum values
        normalized_statuses = set()
        for s in get_status or []:
            if not s:
                continue
            s_clean = s.strip().lower()
            if s_clean == "processed":
                normalized_statuses.add("Processed")
            elif s_clean == "verified":
                normalized_statuses.add("Verified")
            elif s_clean in {"to be verified", "to_be_verified"}:
                normalized_statuses.add("To be verified")
            elif s_clean == "processing":
                normalized_statuses.add("Processing")
            elif s_clean == "failed":
                normalized_statuses.add("Failed")

        # Fallback to the common pair if nothing valid was provided
        if not normalized_statuses:
            normalized_statuses = {"Processed", "Verified"}
    else:
        # End-user: only Verified runs, ignore requested get_status
        normalized_statuses = {"Verified"}

    results: List[SimpleFieldResponseOut] = []

    # 3) For each PO, fetch latest session and runs per allowed status, then responses
    for po in pos:
        # Latest non-deleted session for this PO
        session_stmt = (
            select(ExtractionSession)
            .where(ExtractionSession.po_id == po.id, ExtractionSession.is_deleted.is_(False))
            .order_by(ExtractionSession.updated_at.desc())
            .limit(1)
        )
        session_res = await db.execute(session_stmt)
        session = session_res.scalar_one_or_none()
        if not session:
            continue

        # For each allowed status, pick the latest run (highest version)
        run_ids = []
        for status in normalized_statuses:
            run_stmt = (
                select(ExtractionRun)
                .where(ExtractionRun.session_id == session.id, ExtractionRun.status == status)
                .order_by(ExtractionRun.version.desc())
                .limit(1)
            )
            run_res = await db.execute(run_stmt)
            run = run_res.scalar_one_or_none()
            if run:
                run_ids.append(run.id)

        if not run_ids:
            continue

        # Fetch field responses for all selected runs for this PO
        fr_stmt = select(FieldResponse).where(FieldResponse.extraction_run_id.in_(run_ids))
        fr_res = await db.execute(fr_stmt)
        responses = list(fr_res.scalars().all())

        for fr in responses:
            results.append(
                SimpleFieldResponseOut(
                    po_number=po.number,
                    category_name=fr.category.name if getattr(fr, "category", None) is not None else None,
                    question=fr.question,
                    answer=fr.answer,
                    status=getattr(fr.extraction_run, "status", None),
                )
            )

    # 4) Return in the requested format
    rt = (response_type or "csv").lower()

    if rt == "json":
        return results

    if rt not in {"csv", "excel"}:
        raise HTTPException(status_code=400, detail="Invalid response_type. Use one of: json, csv, excel.")

    # CSV (used also for 'excel' as Excel opens CSV directly)
    output = io.StringIO()
    writer = csv.writer(output)
    # Header
    writer.writerow(["po_number", "category_name", "question", "answer", "status"])
    # Rows
    for row in results:
        writer.writerow(
            [
                row.po_number,
                row.category_name or "",
                row.question or "",
                row.answer or "",
                row.status or "",
            ]
        )

    output.seek(0)
    if rt == "csv":
        filename = "po_field_responses.csv"
        media_type = "text/csv"
    else:  # excel
        filename = "po_field_responses.xlsx"
        media_type = "application/vnd.ms-excel"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"; charset=utf-8'},
    )


