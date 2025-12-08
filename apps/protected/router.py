from fastapi import APIRouter, Depends

from apps.protected.service import get_profile
from models.user import User
from security.auth_backend import get_current_active_user, require_roles
from constants.roles import ADMIN, VALIDATOR, USER

router = APIRouter(prefix="/api/protected", tags=["Protected"])


@router.get("/me")
async def me(current_user: User = Depends(get_current_active_user)):
    """
    Example protected endpoint that returns the current user's profile.
    """
    return get_profile(current_user)


@router.get("/admin-only", dependencies=[Depends(require_roles(ADMIN))])
async def admin_only():
    return {"message": "Hello, admin!"}


@router.get("/validator-only", dependencies=[Depends(require_roles(VALIDATOR))])
async def validator_only():
    return {"message": "Hello, validator!"}


@router.get("/user-or-admin", dependencies=[Depends(require_roles(USER, ADMIN))])
async def user_or_admin():
    return {"message": "Hello, user or admin!"}


@router.get("/common", dependencies=[Depends(get_current_active_user)])
async def common():
    return {"message": "Hello, any authenticated user!"}

