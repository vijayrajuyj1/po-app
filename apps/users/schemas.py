from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserRegister(BaseModel):
    """
    Payload for user registration via invitation token.
    Email and roles are derived from inviteToken; not provided here.
    """
    inviteToken: str = Field(..., description="Invitation JWT")
    name: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    confirm_password: str = Field(..., min_length=8, max_length=128)


class UserLogin(BaseModel):
    """
    Payload for user login.
    """
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class UserOut(BaseModel):
    """
    Public user model returned to clients (no sensitive fields).
    """
    id: str
    email: EmailStr
    name: str
    is_verified: bool
    created_at: Optional[str]
    updated_at: Optional[str]
    roles: list[str] = []

    class Config:
        orm_mode = True


class TokenResponse(BaseModel):
    """
    Response model for token pair and user data.
    """
    token: str
    refreshToken: str
    user: UserOut


class RefreshRequest(BaseModel):
    """
    Payload for refreshing access tokens via refresh token.
    """
    refreshToken: str


class TokenPair(BaseModel):
    """
    Response model for just access and refresh tokens.
    Used by /api/auth/refresh endpoint.
    """
    token: str
    refreshToken: str


class InviteRequest(BaseModel):
    """
    Payload for generating an invitation token.
    """
    email: EmailStr
    roles: list[str]


class InviteResponse(BaseModel):
    """
    Response model for invitation token generation.
    """
    inviteToken: str


class ForgotPasswordRequest(BaseModel):
    """
    Payload for initiating a password reset.
    """
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """
    Payload for completing a password reset with a reset token.
    """
    resetToken: str = Field(..., description="Password reset JWT/token")
    password: str = Field(..., min_length=8, max_length=128)
    confirm_password: str = Field(..., min_length=8, max_length=128)


