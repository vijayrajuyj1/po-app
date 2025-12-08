from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from apps.users.schemas import InviteRequest
from common.jwt import create_invite_token
from constants.roles import ADMIN, VALIDATOR, USER
from models.user import Role


async def generate_invitation(db: AsyncSession, payload: InviteRequest) -> str:
    """
    Validate requested roles against Role table and return an invitation token.
    Stateless: does not persist any data, relies on JWT for integrity.
    """
    requested = set(payload.roles)
    valid_names = {ADMIN, VALIDATOR, USER}
    if not requested.issubset(valid_names):
        raise ValueError("One or more roles are invalid.")

    # Ensure roles exist in DB
    res = await db.execute(select(Role).where(Role.name.in_(list(requested))))
    roles = {r.name for r in res.scalars().all()}
    if roles != requested:
        raise ValueError("One or more roles do not exist.")

    return create_invite_token(payload.email, sorted(list(requested)))


