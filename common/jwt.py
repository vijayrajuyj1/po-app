import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List

from jose import jwt, JWTError

from settings.config import get_settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_token(
    subject: str,
    token_type: str,
    expires_delta: timedelta,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a signed JWT with secure defaults and explicit claims.
    Validates issuer and audience on verification.
    """
    settings = get_settings()
    now = _utcnow()
    jti = str(uuid.uuid4())
    payload: Dict[str, Any] = {
        "sub": subject,
        "jti": jti,
        "type": token_type,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
    }
    if extra_claims:
        payload.update(extra_claims)
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token


def create_access_refresh_tokens(
    user_id: str,
    roles: Optional[List[str]] = None,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generate a short-lived access token and a long-lived refresh token.

    - roles: list of role names to embed as 'roles' claim
    - extra_claims: any additional custom claims to include in both tokens
    """
    settings = get_settings()
    claims: Dict[str, Any] = {}
    if roles is not None:
        claims["roles"] = roles
    if extra_claims:
        claims.update(extra_claims)

    access_token = create_token(
        subject=user_id,
        token_type="access",
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        extra_claims=claims,
    )
    refresh_token = create_token(
        subject=user_id,
        token_type="refresh",
        expires_delta=timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES),
        extra_claims=claims,
    )
    return {"token": access_token, "refreshToken": refresh_token}


# Invitation token utilities (separate secret and expiry)
def create_invite_token(email: str, roles: List[str], expires_hours: Optional[int] = None) -> str:
    """
    Create a JWT invitation token carrying target email and roles.
    Signed with a separate secret defined in settings.
    """
    settings = get_settings()
    now = _utcnow()
    exp_hours = expires_hours if expires_hours is not None else settings.INVITE_TOKEN_EXPIRE_HOURS
    payload: Dict[str, Any] = {
        "email": email,
        "roles": roles,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int((now + timedelta(hours=exp_hours)).timestamp()),
        "iss": settings.JWT_ISSUER,
        "aud": "api-base-invites",
        "type": "invite",
    }
    return jwt.encode(payload, settings.INVITE_TOKEN_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_invite_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate an invitation token.
    Ensures signature, expiration, issuer, and audience.
    """
    settings = get_settings()
    return jwt.decode(
        token,
        settings.INVITE_TOKEN_SECRET,
        algorithms=[settings.JWT_ALGORITHM],
        audience="api-base-invites",
        issuer=settings.JWT_ISSUER,
        options={"require_exp": True, "require_iat": True, "require_nbf": True},
    )


def create_password_reset_token(user_id: str, email: str) -> str:
    """
    Create a short-lived password reset token bound to a specific user + email.
    Uses the main JWT secret/audience with a distinct token_type for validation.
    """
    settings = get_settings()
    expires = timedelta(minutes=settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES)
    claims: Dict[str, Any] = {"email": email}
    return create_token(
        subject=user_id,
        token_type="password_reset",
        expires_delta=expires,
        extra_claims=claims,
    )

def verify_token(token: str, expected_token_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Verify a JWT's signature, expiration, issuer, and audience.
    Optionally enforce token type (e.g., 'access' or 'refresh').
    Returns decoded claims on success; raises JWTError on failure.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            audience=settings.JWT_AUDIENCE,
            issuer=settings.JWT_ISSUER,
            options={"require_exp": True, "require_sub": True, "require_iat": True, "require_nbf": True},
        )
        print(payload)
        if expected_token_type and payload.get("type") != expected_token_type:
            raise JWTError("Invalid token type.")
        return payload
    except JWTError as exc:
        # Re-raise for callers to handle uniformly
        raise exc


