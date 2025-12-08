from typing import List, Sequence

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from apps.team.schemas import (
    TeamUserOut,
    UpdateRolesRequest,
    UpdateStatusRequest,
    BulkRolesRequest,
    BulkStatusRequest,
    BulkIdsRequest,
    BulkResultItem,
    RolesResponse,
    StatusesResponse,
)
from apps.users.invite_service import generate_invitation
from constants.roles import ADMIN, VALIDATOR, USER
from constants.statuses import ACTIVE, PENDING, DISABLED
from email_services.email_client import EmailClient
from models.base import get_db
from models.user import Role, User
from security.auth_backend import require_roles, get_current_active_user
from settings.config import get_settings
from email_services.render import render_invite_email


router = APIRouter(prefix="/api/team", tags=["Team"], dependencies=[Depends(require_roles(ADMIN))])


def _serialize_user(u: User) -> TeamUserOut:
    """Shape a User ORM object into TeamUserOut."""
    roles = [r.name for r in (u.roles or [])]
    return TeamUserOut(id=str(u.id), name=u.name, email=u.email, roles=roles, status=u.status)  # type: ignore[arg-type]


@router.get("/users", response_model=List[TeamUserOut])
async def list_users(
    search: str | None = Query(None, description="Case-insensitive substring match on name or email"),
    roles: List[str] = Query(default=[], alias="role", description="Repeatable: role=ADMIN&role=USER"),
    statuses: List[str] = Query(default=[], alias="status", description="Repeatable: status=ACTIVE&status=PENDING"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns users filtered by optional search term, role, and status.
    Filters are AND-combined.
    """
    stmt = select(User).options(selectinload(User.roles))

    if search:
        like = f"%{search}%"
        stmt = stmt.where(or_(User.name.ilike(like), User.email.ilike(like)))

    if roles:
        requested_roles = {r.upper() for r in roles}
        if not requested_roles.issubset({ADMIN, VALIDATOR, USER}):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role filter")
        # Join on roles membership; OR semantics across selected roles
        stmt = stmt.join(User.roles).where(Role.name.in_(list(requested_roles)))

    if statuses:
        requested_statuses = {s.upper() for s in statuses}
        if not requested_statuses.issubset({ACTIVE, PENDING, DISABLED}):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid status filter")
        stmt = stmt.where(User.status.in_(list(requested_statuses)))

    res = await db.execute(stmt)
    users = res.scalars().unique().all()
    return [_serialize_user(u) for u in users]


@router.patch("/users/{user_id}/roles", response_model=TeamUserOut)
async def replace_user_roles(user_id: str, payload: UpdateRolesRequest, db: AsyncSession = Depends(get_db)):
    """
    Replace the role set for a single user.
    """
    res = await db.execute(select(User).options(selectinload(User.roles)).where(User.id == user_id))
    user = res.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    valid_names = {ADMIN, VALIDATOR, USER}
    requested = set(payload.roles)
    if not requested.issubset(valid_names):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="One or more roles are invalid")

    roles_res = await db.execute(select(Role).where(Role.name.in_(list(requested))))
    roles = roles_res.scalars().all()
    if len(roles) != len(requested):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="One or more roles do not exist")

    user.roles = roles
    await db.commit()
    await db.refresh(user)
    return _serialize_user(user)


@router.patch("/users/{user_id}/status", response_model=TeamUserOut)
async def update_user_status(user_id: str, payload: UpdateStatusRequest, db: AsyncSession = Depends(get_db)):
    """
    Set a user's status to ACTIVE, PENDING, or DISABLED.
    """
    res = await db.execute(select(User).options(selectinload(User.roles)).where(User.id == user_id))
    user = res.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    new_status = payload.status.upper()
    if new_status not in {ACTIVE, PENDING, DISABLED}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid status")

    user.status = new_status
    await db.commit()
    await db.refresh(user)
    return _serialize_user(user)


@router.post("/users/{user_id}/resend-invite")
async def resend_invite(
    user_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Resend an invite email for a Pending user.
    Returns a minimal OK payload for the UI.
    """
    res = await db.execute(select(User).options(selectinload(User.roles)).where(User.id == user_id))
    user = res.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if user.status != PENDING:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Invite can only be resent for PENDING users")

    # Roles required to be embedded in the invite
    role_names = [r.name for r in (user.roles or [])]
    token = await generate_invitation(db, type("InviteReq", (), {"email": user.email, "roles": role_names}))  # type: ignore[misc]

    # Use email client abstraction (stubbed) for sending
    client = EmailClient()
    settings = get_settings()
    origin = request.headers.get("origin") or (settings.FRONTEND_ORIGIN_DEFAULT or str(request.base_url).rstrip("/"))
    accept_path = settings.INVITE_ACCEPT_PATH
    invite_url = f"{origin}{accept_path}?token={token}"
    subject = f"You're invited to {settings.COMPANY_NAME or settings.PRODUCT_NAME}"
    html = render_invite_email(
        recipient_email=user.email,
        admin_name=current_user.name,
        admin_email=current_user.email,
        company_name=settings.COMPANY_NAME,
        product_name=settings.PRODUCT_NAME,
        invite_url=invite_url,
        expiry_hours=settings.INVITE_TOKEN_EXPIRE_HOURS,
    )
    await client.send_email(to=[user.email], subject=subject, html_body=html, from_email=current_user.email)

    return {"userId": str(user.id), "sent": True, "token": token}


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_user(user_id: str, db: AsyncSession = Depends(get_db)):
    """
    Permanently delete a user.
    """
    res = await db.execute(select(User).where(User.id == user_id))
    user = res.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    await db.delete(user)
    await db.commit()
    return None


@router.patch("/users:bulk-roles", response_model=List[BulkResultItem])
async def bulk_replace_roles(payload: BulkRolesRequest, db: AsyncSession = Depends(get_db)):
    """
    Replace roles for multiple users. Returns per-user results.
    """
    results: List[BulkResultItem] = []
    valid_names = {ADMIN, VALIDATOR, USER}
    requested = set(payload.roles)
    if not requested.issubset(valid_names):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="One or more roles are invalid")

    roles_res = await db.execute(select(Role).where(Role.name.in_(list(requested))))
    roles = roles_res.scalars().all()

    for uid in payload.userIds:
        res = await db.execute(select(User).where(User.id == uid))
        user = res.scalar_one_or_none()
        if not user:
            results.append(BulkResultItem(userId=uid, status="failed", reason="not_found"))
            continue
        user.roles = roles
        results.append(BulkResultItem(userId=uid, status="updated"))

    await db.commit()
    return results


@router.patch("/users:bulk-status", response_model=List[BulkResultItem])
async def bulk_change_status(payload: BulkStatusRequest, db: AsyncSession = Depends(get_db)):
    """
    Change status for multiple users. Returns per-user results.
    """
    new_status = payload.status.upper()
    if new_status not in {ACTIVE, PENDING, DISABLED}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid status")

    results: List[BulkResultItem] = []
    for uid in payload.userIds:
        res = await db.execute(select(User).where(User.id == uid))
        user = res.scalar_one_or_none()
        if not user:
            results.append(BulkResultItem(userId=uid, status="failed", reason="not_found"))
            continue
        user.status = new_status
        results.append(BulkResultItem(userId=uid, status="updated"))

    await db.commit()
    return results


@router.post("/users:bulk-resend-invites", response_model=List[BulkResultItem])
async def bulk_resend_invites(
    payload: BulkIdsRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Resend invites for a set of user IDs. Only PENDING users receive emails.
    """
    results: List[BulkResultItem] = []
    client = EmailClient()
    settings = get_settings()
    origin = request.headers.get("origin") or (settings.FRONTEND_ORIGIN_DEFAULT or str(request.base_url).rstrip("/"))
    accept_path = settings.INVITE_ACCEPT_PATH

    for uid in payload.userIds:
        res = await db.execute(select(User).options(selectinload(User.roles)).where(User.id == uid))
        user = res.scalar_one_or_none()
        if not user:
            results.append(BulkResultItem(userId=uid, status="failed", reason="not_found"))
            continue
        if user.status != PENDING:
            results.append(BulkResultItem(userId=uid, status="skipped", reason="not_pending"))
            continue

        role_names = [r.name for r in (user.roles or [])]
        token = await generate_invitation(db, type("InviteReq", (), {"email": user.email, "roles": role_names}))  # type: ignore[misc]
        invite_url = f"{origin}{accept_path}?inviteToken={token}"
        subject = f"You're invited to {settings.COMPANY_NAME or settings.PRODUCT_NAME}"
        html = render_invite_email(
            recipient_email=user.email,
            admin_name=current_user.name,
            admin_email=current_user.email,
            company_name=settings.COMPANY_NAME,
            product_name=settings.PRODUCT_NAME,
            invite_url=invite_url,
            expiry_hours=settings.INVITE_TOKEN_EXPIRE_HOURS,
        )
        await client.send_email(to=[user.email], subject=subject, html_body=html, from_email=current_user.email)
        results.append(BulkResultItem(userId=uid, status="sent"))

    return results


@router.delete("/users:bulk-remove", response_model=List[BulkResultItem])
async def bulk_remove(payload: BulkIdsRequest, db: AsyncSession = Depends(get_db)):
    """
    Remove multiple users. Returns per-user results.
    """
    results: List[BulkResultItem] = []
    for uid in payload.userIds:
        res = await db.execute(select(User).where(User.id == uid))
        user = res.scalar_one_or_none()
        if not user:
            results.append(BulkResultItem(userId=uid, status="failed", reason="not_found"))
            continue
        await db.delete(user)
        results.append(BulkResultItem(userId=uid, status="deleted"))
    await db.commit()
    return results


@router.get("/roles", response_model=RolesResponse)
async def list_roles():
    """
    Helper endpoint for client dropdowns.
    """
    return RolesResponse(roles=[ADMIN, VALIDATOR, USER])


@router.get("/statuses", response_model=StatusesResponse)
async def list_statuses():
    """
    Helper endpoint for client dropdowns.
    """
    return StatusesResponse(statuses=[ACTIVE, PENDING, DISABLED])


