from typing import Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from apps.users.exception import http_bad_request, http_conflict, http_unauthorized
from apps.users.schemas import UserRegister, UserLogin
from apps.users.utils import sanitize_user
from common.hashing import hash_password, verify_password
from common.jwt import create_access_refresh_tokens, verify_token, decode_invite_token
from models.user import User, Role
from security.password_rules import validate_password_strength, assert_passwords_match
from constants.roles import USER as ROLE_USER, ADMIN as ROLE_ADMIN, VALIDATOR as ROLE_VALIDATOR
from constants.statuses import ACTIVE, PENDING


async def register_user(db: AsyncSession, payload: UserRegister) -> dict:
    """
    Register a new user with validation and secure password hashing.
    """
    # Decode and validate invitation token
    try:
        invite = decode_invite_token(payload.inviteToken)
    except Exception as exc:
        raise http_bad_request("Invalid or expired invitation token.") from exc

    email_from_token = invite.get("email")
    token_roles = invite.get("roles") or []
    if not email_from_token or not isinstance(token_roles, list) or not token_roles:
        raise http_bad_request("Invitation token is missing required claims.")

    # Lookup by email (to support pending stub upgrade path)
    existing_email_res = await db.execute(select(User).where(User.email == email_from_token))
    existing_by_email = existing_email_res.scalar_one_or_none()

    # No username field in this project; nothing to validate there

    # Validate passwords
    assert_passwords_match(payload.password, payload.confirm_password)
    valid, message = validate_password_strength(payload.password)
    if not valid:
        raise http_bad_request(message)

    # Validate role names from token
    requested_roles = list(token_roles)
    valid_role_names = {ROLE_USER, ROLE_ADMIN, ROLE_VALIDATOR}
    if not set(requested_roles).issubset(valid_role_names):
        raise http_bad_request("One or more roles are invalid.")

    # Fetch Role objects
    roles_res = await db.execute(select(Role).where(Role.name.in_(requested_roles)))
    roles = roles_res.scalars().all()
    if len(roles) != len(set(requested_roles)):
        raise http_bad_request("One or more roles do not exist.")

    # If a pending stub exists for the email, upgrade it in-place
    if existing_by_email:
        if existing_by_email.status != PENDING or existing_by_email.is_verified:
            raise http_conflict("Email already in use.")

        existing_by_email.name = payload.name
        existing_by_email.password_hash = hash_password(payload.password)
        existing_by_email.is_verified = True
        existing_by_email.status = ACTIVE
        # Keep roles from token (overwrite to be safe)
        existing_by_email.roles = roles
        await db.commit()
        await db.refresh(existing_by_email)
        return sanitize_user(existing_by_email, roles=[r.name for r in roles])

    # Otherwise create a fresh user using email from invite
    user = User(
        email=email_from_token,
        name=payload.name,
        password_hash=hash_password(payload.password),
        is_verified=True,
        status=ACTIVE,
    )
    user.roles = roles
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return sanitize_user(user, roles=[r.name for r in roles])


async def authenticate_user(db: AsyncSession, payload: UserLogin) -> Tuple[dict, dict]:
    """
    Authenticate a user by email/password and return a token pair and user dict.
    """
    res = await db.execute(select(User).where(User.email == payload.email))
    user = res.scalar_one_or_none()
    if not user or not verify_password(payload.password, user.password_hash):
        raise http_unauthorized("Invalid email or password.")
    # Ensure roles are loaded for response and token
    roles_res = await db.execute(select(Role).join(User.roles).where(User.id == user.id))
    roles = {r.name for r in roles_res.scalars().all()}
    tokens = create_access_refresh_tokens(str(user.id), sorted(list(roles)))
    return tokens, sanitize_user(user, roles=sorted(list(roles)))


async def refresh_tokens(db: AsyncSession, refresh_token: str) -> dict:
    """
    Issue a new access/refresh token pair from a valid refresh token.
    """
    claims = verify_token(refresh_token, expected_token_type="refresh")
    user_id = claims.get("sub")
    roles_res = await db.execute(select(Role).join(User.roles).where(User.id == user_id))
    roles = {r.name for r in roles_res.scalars().all()}
    return create_access_refresh_tokens(user_id, sorted(list(roles)))


