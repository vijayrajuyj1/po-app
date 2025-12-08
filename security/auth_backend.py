from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from common.jwt import verify_token
from models.base import get_db
from models.user import User

# OAuth2PasswordBearer expects a tokenUrl for the interactive docs to work.
# Use a dedicated OAuth2-compatible token endpoint that accepts form data.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=True)


async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> User:
    """
    Dependency to retrieve the current authenticated user from the JWT.
    Validates the access token and fetches the associated user from DB.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = verify_token(token, expected_token_type="access")
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user: Optional[User] = await db.get(User, user_id)
    if not user:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Placeholder for additional user state checks.
    e.g., verify user.is_active == True if implementing soft-delete or blocked users.
    """
    return current_user


def require_roles(*allowed_roles: str):
    """
    Dependency factory to enforce that the current access token includes at least one allowed role.
    Usage:
      @router.get("/admin", dependencies=[Depends(require_roles("ADMIN"))])
    """
    async def _dependency(token: str = Depends(oauth2_scheme)):
        try:
            payload = verify_token(token, expected_token_type="access")
            roles = set(payload.get("roles") or [])
            print(roles)
            print(allowed_roles)
            if not roles.intersection(set(allowed_roles)):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        except JWTError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    return _dependency

