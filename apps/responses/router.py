from typing import List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from apps.responses.schemas import (
    FieldResponseOut,
    ListResponsesQuery,
    CreateCommentRequest,
    CommentOut,
    CreateIssueRequest,
    IssueOut,
    EditResponseRequest,
    RegenerateResponseRequest,
    UpdateStatusRequest,
)
from apps.extraction.service import ActivityLogService
from models.base import get_db
from models.field_response import FieldResponse
from models.field_response_comment import FieldResponseComment
from models.field_response_issue import FieldResponseIssue
from models.field_response_edit import FieldResponseEdit
from models.user import User
from security.auth_backend import get_current_active_user, require_roles
from constants.roles import ADMIN, VALIDATOR


router = APIRouter(prefix="/api/responses", tags=["Field Responses"])


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


@router.get("", response_model=List[FieldResponseOut])
async def list_responses(
    sessionId: Optional[str] = Query(default=None),
    runId: Optional[str] = Query(default=None),
    categoryId: Optional[str] = Query(default=None),
    fieldId: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    where = []
    if sessionId:
        where.append(FieldResponse.session_id == uuid.UUID(sessionId))
    if runId:
        where.append(FieldResponse.extraction_run_id == uuid.UUID(runId))
    if categoryId:
        where.append(FieldResponse.category_id == uuid.UUID(categoryId))
    if fieldId:
        where.append(FieldResponse.field_id == uuid.UUID(fieldId))
    if status:
        where.append(FieldResponse.status == status)

    stmt = select(FieldResponse).where(and_(*where)) if where else select(FieldResponse)
    stmt = stmt.order_by(FieldResponse.updated_at.desc()).offset(offset).limit(limit)
    res = await db.execute(stmt)
    items = res.scalars().all()
    return [_serialize(fr) for fr in items]


@router.get("/{response_id}", response_model=FieldResponseOut)
async def get_response(response_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    fr = await db.get(FieldResponse, uuid.UUID(response_id))
    if not fr:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field response not found")
    return _serialize(fr)


@router.get("/{response_id}/comments", response_model=List[CommentOut])
async def list_comments(response_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    stmt = select(FieldResponseComment).where(FieldResponseComment.field_response_id == uuid.UUID(response_id)).order_by(FieldResponseComment.created_at.asc())
    res = await db.execute(stmt)
    comments = res.scalars().all()
    return [
        CommentOut(
            id=str(c.id),
            fieldResponseId=str(c.field_response_id),
            authorId=str(c.author_id) if c.author_id else None,
            content=c.content,
            createdAt=c.created_at,  # type: ignore[arg-type]
        )
        for c in comments
    ]


@router.post(
    "/{response_id}/comments",
    response_model=CommentOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_roles(VALIDATOR, ADMIN))],
)
async def create_comment(
    response_id: str,
    payload: CreateCommentRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    fr = await db.get(FieldResponse, uuid.UUID(response_id))
    if not fr:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field response not found")

    comment = FieldResponseComment(
        field_response_id=fr.id,
        author_id=current_user.id,
        content=payload.content,
    )
    db.add(comment)
    await db.commit()
    await db.refresh(comment)

    await ActivityLogService.log_action(
        db,
        str(current_user.id),
        "FIELD_RESPONSE_COMMENT",
        str(fr.session_id),
        str(fr.extraction_run_id),
        None,
        None,
        {"responseId": str(fr.id)},
    )

    return CommentOut(
        id=str(comment.id),
        fieldResponseId=str(comment.field_response_id),
        authorId=str(comment.author_id) if comment.author_id else None,
        content=comment.content,
        createdAt=comment.created_at,  # type: ignore[arg-type]
    )


@router.post(
    "/{response_id}/issues",
    response_model=IssueOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_roles(VALIDATOR, ADMIN))],
)
async def create_issue(
    response_id: str,
    payload: CreateIssueRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    fr = await db.get(FieldResponse, uuid.UUID(response_id))
    if not fr:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field response not found")

    issue = FieldResponseIssue(
        field_response_id=fr.id,
        reporter_id=current_user.id,
        title=payload.title,
        description=payload.description,
        severity=payload.severity,
        is_blocking=payload.isBlocking,
    )
    db.add(issue)
    await db.commit()
    await db.refresh(issue)

    await ActivityLogService.log_action(
        db,
        str(current_user.id),
        "FIELD_RESPONSE_ISSUE",
        str(fr.session_id),
        str(fr.extraction_run_id),
        None,
        None,
        {"responseId": str(fr.id), "issueId": str(issue.id), "severity": issue.severity},
    )

    return IssueOut(
        id=str(issue.id),
        fieldResponseId=str(issue.field_response_id),
        reporterId=str(issue.reporter_id) if issue.reporter_id else None,
        title=issue.title,
        description=issue.description,
        severity=issue.severity,
        status=issue.status,
        isBlocking=issue.is_blocking,
        createdAt=issue.created_at,  # type: ignore[arg-type]
        resolvedAt=issue.resolved_at,  # type: ignore[arg-type]
        resolvedBy=str(issue.resolved_by) if issue.resolved_by else None,
    )


async def _apply_edit(
    db: AsyncSession,
    fr: FieldResponse,
    actor: User,
    action: str,
    payload: EditResponseRequest | RegenerateResponseRequest,
) -> FieldResponseOut:
    edit = FieldResponseEdit(
        field_response_id=fr.id,
        actor_id=actor.id,
        action=action,
        reason=getattr(payload, "reason", None),
        before_answer=fr.answer,
        before_short_answer=fr.short_answer,
        before_citations=fr.citations,
        after_answer=payload.answer if payload.answer is not None else fr.answer,
        after_short_answer=payload.shortAnswer if payload.shortAnswer is not None else fr.short_answer,
        after_citations=payload.citations if payload.citations is not None else fr.citations,
    )
    db.add(edit)

    # Update FieldResponse
    fr.answer = edit.after_answer
    fr.short_answer = edit.after_short_answer
    fr.citations = edit.after_citations
    fr.is_modified = True
    fr.modified_by = actor.id
    # modified_at handled by DB trigger default on update; set explicitly for async correctness
    from datetime import datetime, timezone
    fr.modified_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(fr)

    await ActivityLogService.log_action(
        db,
        str(actor.id),
        f"FIELD_RESPONSE_{action}",
        str(fr.session_id),
        str(fr.extraction_run_id),
        None,
        None,
        {"responseId": str(fr.id), "editId": str(edit.id)},
    )

    return _serialize(fr)


@router.patch(
    "/{response_id}/edit",
    response_model=FieldResponseOut,
    dependencies=[Depends(require_roles(VALIDATOR, ADMIN))],
)
async def edit_response(
    response_id: str,
    payload: EditResponseRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    fr = await db.get(FieldResponse, uuid.UUID(response_id))
    if not fr:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field response not found")
    if fr.status == "Verified":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Verified responses cannot be edited")
    return await _apply_edit(db, fr, current_user, "EDIT", payload)


@router.post(
    "/{response_id}/regenerate",
    response_model=FieldResponseOut,
    dependencies=[Depends(require_roles(VALIDATOR, ADMIN))],
)
async def regenerate_response(
    response_id: str,
    payload: RegenerateResponseRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    fr = await db.get(FieldResponse, uuid.UUID(response_id))
    if not fr:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field response not found")
    if fr.status == "Verified":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Verified responses cannot be regenerated")
    return await _apply_edit(db, fr, current_user, "REGENERATE", payload)


@router.patch(
    "/{response_id}/status",
    response_model=FieldResponseOut,
    dependencies=[Depends(require_roles(VALIDATOR, ADMIN))],
)
async def update_status(
    response_id: str,
    payload: UpdateStatusRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    fr = await db.get(FieldResponse, uuid.UUID(response_id))
    if not fr:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field response not found")

    old = fr.status
    fr.status = payload.status
    await db.commit()
    await db.refresh(fr)

    await ActivityLogService.log_action(
        db,
        str(current_user.id),
        "FIELD_RESPONSE_STATUS",
        str(fr.session_id),
        str(fr.extraction_run_id),
        old,
        fr.status,
        {"responseId": str(fr.id)},
    )

    return _serialize(fr)


@router.post(
    "/{response_id}/verify",
    response_model=FieldResponseOut,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_roles(VALIDATOR, ADMIN))],
)
async def verify_response(
    response_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Mark a field response as Verified.
    - Sets status='Verified'
    - Stamps verifiedBy/verifiedAt
    """
    fr = await db.get(FieldResponse, uuid.UUID(response_id))
    if not fr:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field response not found")

    old = fr.status
    fr.status = "Verified"
    from datetime import datetime, timezone
    fr.verified_by = current_user.id
    fr.verified_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(fr)

    await ActivityLogService.log_action(
        db,
        str(current_user.id),
        "FIELD_RESPONSE_VERIFY",
        str(fr.session_id),
        str(fr.extraction_run_id),
        old,
        fr.status,
        {"responseId": str(fr.id)},
    )

    return _serialize(fr)

