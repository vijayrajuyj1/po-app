import uuid
from sqlalchemy import Column, Text, DateTime, String, Boolean, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from models.base import Base


class FieldResponseIssue(Base):
    """
    Issue report for a FieldResponse (e.g., incorrect extraction, missing citation).
    """
    __tablename__ = "field_response_issues"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    field_response_id = Column(PGUUID(as_uuid=True), ForeignKey("field_responses.id", ondelete="CASCADE"), nullable=False, index=True)
    reporter_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    severity = Column(String(20), nullable=True)  # e.g., low|medium|high
    status = Column(String(20), nullable=False, server_default="open")  # open|resolved|dismissed
    is_blocking = Column(Boolean, nullable=False, server_default="false")

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    field_response = relationship("FieldResponse", lazy="joined")


