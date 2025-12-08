from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.base import get_db
from models.user import User
from security.auth_backend import get_current_active_user, require_roles
from constants.roles import VALIDATOR, ADMIN, USER
from apps.download.schemas import POResponsesRequest
from apps.download.service import session_exists_for_po, generate_po_responses

router = APIRouter(prefix="/api/download", tags=["Download Dashboard"])



@router.get(
    "/dashboard/po/{po_id}/session-exists",
    dependencies=[Depends(require_roles(VALIDATOR, ADMIN))],
)
async def check_session_exists_for_po(
    po_number: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Check whether a non-deleted session exists for the given PO ID.
    Returns: {"exists": true/false}
    """
    exists = await session_exists_for_po(po_number, db)
    return {"exists": exists}


@router.post(
    "/dashboard/po/responses",
    dependencies=[Depends(require_roles(USER, VALIDATOR, ADMIN))],
)
async def list_field_responses_for_po_with_statuses(
    payload: POResponsesRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Given one or more PO numbers and optional run statuses (in the request body), return field responses from the
    latest ExtractionRun **per allowed status** for each PO.

    - ADMIN and VALIDATOR: for each status in `get_status`, pick the latest run with that status.
    - USER: restricted to runs with status == 'Verified' (ignores `get_status`).

    The outputs are merged, simplified to:
      [{po_number, category_name, question, answer, status}]

    Response type:
      - json  -> JSON array of the above objects
      - csv   -> downloadable CSV
      - excel -> downloadable CSV with Excel-compatible content-type
    """
    return await generate_po_responses(payload, db, current_user)


