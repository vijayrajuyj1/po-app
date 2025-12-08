import uuid
from sqlalchemy import Column, Text, DateTime, ForeignKey, JSON, String, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from models.base import Base


class FieldResponseEdit(Base):
    """
    Tracks edits and regenerations to FieldResponse answers.
    Stores before/after snapshots with actor and reason, for auditability.
    """
    __tablename__ = "field_response_edits"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    field_response_id = Column(PGUUID(as_uuid=True), ForeignKey("field_responses.id", ondelete="CASCADE"), nullable=False, index=True)
    actor_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    action = Column(String(30), nullable=False)  # EDIT | REGENERATE
    reason = Column(Text, nullable=True)

    before_answer = Column(Text, nullable=True)
    before_short_answer = Column(Text, nullable=True)
    before_citations = Column(JSON, nullable=True)

    after_answer = Column(Text, nullable=True)
    after_short_answer = Column(Text, nullable=True)
    after_citations = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    field_response = relationship("FieldResponse", lazy="joined")


