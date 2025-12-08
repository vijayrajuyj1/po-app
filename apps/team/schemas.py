from typing import List, Literal, Optional
from pydantic import BaseModel, EmailStr, Field, validator

from constants.roles import ADMIN, VALIDATOR, USER
from constants.statuses import ACTIVE, PENDING, DISABLED


RoleName = Literal[ADMIN, VALIDATOR, USER]
StatusName = Literal[ACTIVE, PENDING, DISABLED]


class TeamUserOut(BaseModel):
    """
    Minimal shape required by the Team table UI.
    """
    id: str
    name: str
    email: EmailStr
    roles: List[RoleName]
    status: StatusName


class UpdateRolesRequest(BaseModel):
    """
    Replaces a user's roles with the provided list.
    """
    roles: List[RoleName] = Field(..., min_items=1)

    @validator("roles")
    def unique_roles(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            raise ValueError("Duplicate roles are not allowed.")
        return v


class UpdateStatusRequest(BaseModel):
    """
    Sets a user's status.
    """
    status: StatusName


class BulkRolesRequest(BaseModel):
    userIds: List[str] = Field(..., min_items=1)
    roles: List[RoleName] = Field(..., min_items=1)

    @validator("roles")
    def unique_roles(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            raise ValueError("Duplicate roles are not allowed.")
        return v


class BulkStatusRequest(BaseModel):
    userIds: List[str] = Field(..., min_items=1)
    status: StatusName


class BulkIdsRequest(BaseModel):
    userIds: List[str] = Field(..., min_items=1)


class BulkResultItem(BaseModel):
    userId: str
    status: Literal["updated", "deleted", "sent", "skipped", "failed"]
    reason: Optional[str] = None


class RolesResponse(BaseModel):
    roles: List[RoleName]


class StatusesResponse(BaseModel):
    statuses: List[StatusName]


