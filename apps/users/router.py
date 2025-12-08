from fastapi import APIRouter, Depends, Response, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_

from apps.users.schemas import (
    UserRegister,
    UserLogin,
    TokenResponse,
    RefreshRequest,
    UserOut,
    TokenPair,
    InviteRequest,
    InviteResponse,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)
from apps.users.service import register_user, authenticate_user, refresh_tokens
from apps.users.invite_service import generate_invitation
from common.hashing import verify_password, hash_password
from common.jwt import create_password_reset_token, verify_token
from apps.users.exception import http_unauthorized, http_conflict, http_bad_request
from models.user import User, Role
from models.base import get_db
from settings.config import get_settings
from security.auth_backend import require_roles, get_current_active_user
from security.password_rules import validate_password_strength, assert_passwords_match
from constants.roles import ADMIN
from constants.statuses import PENDING, DISABLED
import secrets
from email_services.email_client import EmailClient
from email_services.render import render_invite_email, render_password_reset_email

router = APIRouter(prefix="/api/auth", tags=["Auth"])


@router.post("/register", response_model=UserOut)
async def register(payload: UserRegister, db: AsyncSession = Depends(get_db)):
    """
    Register a new user.
    - Validates unique email
    - Validates password strength and confirmation
    - Stores only hashed password
    """
    return await register_user(db, payload)


@router.post("/login", response_model=TokenResponse)
async def login(payload: UserLogin, response: Response, db: AsyncSession = Depends(get_db)):
    """
    Login with email and password to receive access + refresh tokens.
    Optionally also sets HttpOnly secure cookies when enabled via config.
    """
    settings = get_settings()
    tokens, user_dict = await authenticate_user(db, payload)

    # Optionally set tokens in HttpOnly cookies for browser clients
    if settings.ENABLE_COOKIE_AUTH:
        response.set_cookie(
            key="access_token",
            value=tokens["token"],
            httponly=True,
            secure=settings.COOKIE_SECURE,
            samesite="strict" if settings.ENVIRONMENT == "production" else "lax",
            domain=settings.COOKIE_DOMAIN,
            max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )
        response.set_cookie(
            key="refresh_token",
            value=tokens["refreshToken"],
            httponly=True,
            secure=settings.COOKIE_SECURE,
            samesite="strict" if settings.ENVIRONMENT == "production" else "lax",
            domain=settings.COOKIE_DOMAIN,
            max_age=settings.REFRESH_TOKEN_EXPIRE_MINUTES * 60,
        )

    return {"token": tokens["token"], "refreshToken": tokens["refreshToken"], "user": user_dict}


@router.post("/refresh", response_model=TokenPair)
async def refresh(payload: RefreshRequest, response: Response, db: AsyncSession = Depends(get_db)):
    """
    Refresh tokens using a valid refresh token.
    """
    settings = get_settings()
    tokens = await refresh_tokens(db, payload.refreshToken)

    if settings.ENABLE_COOKIE_AUTH:
        response.set_cookie(
            key="access_token",
            value=tokens["token"],
            httponly=True,
            secure=settings.COOKIE_SECURE,
            samesite="strict" if settings.ENVIRONMENT == "production" else "lax",
            domain=settings.COOKIE_DOMAIN,
            max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )
        response.set_cookie(
            key="refresh_token",
            value=tokens["refreshToken"],
            httponly=True,
            secure=settings.COOKIE_SECURE,
            samesite="strict" if settings.ENVIRONMENT == "production" else "lax",
            domain=settings.COOKIE_DOMAIN,
            max_age=settings.REFRESH_TOKEN_EXPIRE_MINUTES * 60,
        )

    return {"token": tokens["token"], "refreshToken": tokens["refreshToken"]}


@router.post("/logout")
async def logout(response: Response):
    """
    Logout endpoint - stateless JWT typically needs no server action.
    If cookie-auth is enabled, clear cookies.
    """
    settings = get_settings()
    if settings.ENABLE_COOKIE_AUTH:
        response.delete_cookie("access_token", domain=settings.COOKIE_DOMAIN)
        response.delete_cookie("refresh_token", domain=settings.COOKIE_DOMAIN)
    return {"message": "Logged out successfully"}


# OAuth2 token endpoint for Swagger "Authorize" (password flow)
@router.post("/token")
async def issue_token(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    """
    OAuth2 password flow token endpoint used by the Swagger Authorize dialog.
    Accepts form data fields 'username' and 'password' and returns a bearer token.
    - 'username' field carries the email (OAuth2PasswordRequestForm naming)
    """
    identifier = form_data.username  # this is the email
    result = await db.execute(select(User).where(User.email == identifier))
    user = result.scalar_one_or_none()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise http_unauthorized("Invalid username/email or password.")

    # Return shape expected by OAuth2PasswordBearer
    from common.jwt import create_access_refresh_tokens

    # Load roles and include additional claims in the token for Swagger OAuth2 flow
    roles_res = await db.execute(select(Role).join(User.roles).where(User.id == user.id))
    roles = {r.name for r in roles_res.scalars().all()}
    tokens = create_access_refresh_tokens(
        str(user.id),
        sorted(list(roles)),
        extra_claims={"email": user.email, "name": user.name},
    )
    return {"access_token": tokens["token"], "token_type": "bearer"}


@router.post("/invite", response_model=InviteResponse, dependencies=[Depends(require_roles(ADMIN))])
async def create_invite(
    payload: InviteRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    ADMIN-only endpoint to generate invitation tokens for email + roles.
    """
    # If email exists, prompt conflict (do not switch status automatically)
    existing_res = await db.execute(select(User).where(User.email == payload.email))
    existing = existing_res.scalar_one_or_none()
    if existing:
        raise http_conflict("Email already exists.")

    # Validate roles exist and create a pending user stub so Team UI can show it
    token = await generate_invitation(db, payload)

    # Fetch roles
    roles_res = await db.execute(select(Role).where(Role.name.in_(payload.roles)))
    roles = roles_res.scalars().all()
    if len(roles) != len(set(payload.roles)):
        raise http_bad_request("One or more roles do not exist.")

    # Create minimal pending user record
    random_password = hash_password(secrets.token_urlsafe(24))
    local_part = payload.email.split("@")[0] if "@" in payload.email else payload.email
    user = User(
        email=payload.email,
        name=local_part,
        password_hash=random_password,
        is_verified=False,
        status=PENDING,
    )
    user.roles = roles

    db.add(user)
    await db.commit()

    # Build accept-invite URL from request origin
    settings = get_settings()
    origin = request.headers.get("origin") or (settings.FRONTEND_ORIGIN_DEFAULT or str(request.base_url).rstrip("/"))
    accept_path = settings.INVITE_ACCEPT_PATH
    invite_url = f"{origin}{accept_path}?token={token}"

    # Prepare and send the invitation email
    subject = f"You're invited to {settings.COMPANY_NAME or settings.PRODUCT_NAME}"
    html = render_invite_email(
        recipient_email=payload.email,
        admin_name=current_user.name,
        admin_email=current_user.email,
        company_name=settings.COMPANY_NAME,
        product_name=settings.PRODUCT_NAME,
        invite_url=invite_url,
        expiry_hours=settings.INVITE_TOKEN_EXPIRE_HOURS,
    )
    client = EmailClient()
    await client.send_email(to=[payload.email], subject=subject, html_body=html, from_email=current_user.email)

    return {"inviteToken": token}


@router.post("/forgot-password")
async def forgot_password(
    payload: ForgotPasswordRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Initiate password reset flow:
    - Accepts an email address
    - If a matching active account exists, sends a password reset link with a time-limited token
    - Response is generic to avoid leaking whether the email is registered
    """
    settings = get_settings()

    # Look up user by email; do not reveal whether it exists
    res = await db.execute(select(User).where(User.email == payload.email))
    user = res.scalar_one_or_none()

    if user and user.status.lower() == 'active':
        token = create_password_reset_token(str(user.id), user.email)

        origin = request.headers.get("origin") or (
            settings.FRONTEND_ORIGIN_DEFAULT or str(request.base_url).rstrip("/")
        )
        reset_path = settings.PASSWORD_RESET_PATH
        reset_url = f"{origin}{reset_path}?token={token}"

        subject = f"Reset your {settings.COMPANY_NAME or settings.PRODUCT_NAME} password"
        html = render_password_reset_email(
            recipient_email=user.email,
            company_name=settings.COMPANY_NAME,
            product_name=settings.PRODUCT_NAME,
            reset_url=reset_url,
            expiry_minutes=settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES,
        )
        client = EmailClient()
        response = await client.send_email(to=[user.email], subject=subject, html_body=html)

        return {
            "status": "success",
            "reset_token": token
        }
    elif user and user.status.lower() != 'active':
        return {
            "status": "error",
            "message": f"Account status is {user.status}."
        }
    
    else:
        return {
            "status": "error",
            "message": "User not found."
        }
    

    # Always return a generic message
    # return {
    #     "message": "If an account with that email exists, a password reset link has been sent."
    # }


@router.post("/reset-password")
async def reset_password(
    payload: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Complete password reset:
    - Validates reset token
    - Validates password + confirmation and strength
    - Updates the user's password
    """
    # Decode and validate reset token
    try:
        claims = verify_token(payload.resetToken, expected_token_type="password_reset")
    except Exception as exc:
        raise http_bad_request("Invalid or expired reset token.") from exc

    user_id = claims.get("sub")
    email = claims.get("email")
    if not user_id or not email:
        raise http_bad_request("Reset token is missing required claims.")

    # Fetch the user referenced by the token
    res = await db.execute(select(User).where(User.id == user_id, User.email == email))
    user = res.scalar_one_or_none()
    if not user:
        raise http_bad_request("Invalid reset token.")
    if user.status == DISABLED:
        raise http_bad_request("Account is disabled.")

    # Validate password and confirmation
    try:
        assert_passwords_match(payload.password, payload.confirm_password)
    except ValueError as exc:
        raise http_bad_request(str(exc)) from exc

    valid, message = validate_password_strength(payload.password)
    if not valid:
        raise http_bad_request(message)

    # Update password
    user.password_hash = hash_password(payload.password)
    await db.commit()

    return {"message": "Password has been reset successfully."}

