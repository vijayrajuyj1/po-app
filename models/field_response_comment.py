import uuid
from sqlalchemy import Column, Text, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from models.base import Base


class FieldResponseComment(Base):
    """
    Comment left by a user on a specific FieldResponse.
    """
    __tablename__ = "field_response_comments"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    field_response_id = Column(PGUUID(as_uuid=True), ForeignKey("field_responses.id", ondelete="CASCADE"), nullable=False, index=True)
    author_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    content = Column(Text, nullable=False)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    field_response = relationship("FieldResponse", lazy="joined")


