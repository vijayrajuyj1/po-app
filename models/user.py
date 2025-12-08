import uuid
from datetime import datetime

from sqlalchemy import Column, String, Boolean, DateTime, func, UniqueConstraint, Table, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from models.base import Base
from constants.statuses import ACTIVE, PENDING, DISABLED


class User(Base):
    """
    User ORM model.
    - Uses UUID primary key
    - Unique email
    - Stores only a secure password hash (never plaintext)
    - Timestamps for auditing
    """

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("email", name="uq_users_email"),
        # Only email is unique; username removed
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    email = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_verified = Column(Boolean, nullable=False, default=True, server_default="true")
    # Team Management UI requires a simple status lifecycle
    # Stored as UPPERCASE strings to match server constants
    status = Column(String(32), nullable=False, default=ACTIVE, server_default=ACTIVE, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    def to_public_dict(self) -> dict:
        """
        Returns a sanitized dict without sensitive fields.
        """
        data = {
            "id": str(self.id),
            "email": self.email,
            "name": self.name,
            "is_verified": self.is_verified,
            "status": getattr(self, "status", ACTIVE),
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else str(self.created_at),
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else str(self.updated_at),
        }
        try:
            data["roles"] = [r.name for r in (getattr(self, "roles", []) or [])]
        except Exception:
            pass
        return data


# Association table for many-to-many User <-> Role
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("role_id", UUID(as_uuid=True), ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
)


class Role(Base):
    """
    Role ORM model with unique name. Valid values should be enforced at application layer.
    """
    __tablename__ = "roles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


# Add roles relationship to User (after Role is defined)
User.roles = relationship("Role", secondary=user_roles, lazy="selectin")



