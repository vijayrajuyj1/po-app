import uuid
from sqlalchemy import Column, Text, DateTime, Boolean, Float, ForeignKey, JSON, Enum, func, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from models.base import Base


FIELD_RESPONSE_STATUS_VALUES = ("Pending", "To be verified", "Processing", "Processed", "Verified", "Failed")


class FieldResponse(Base):
    """
    Stores extraction responses for individual fields within an extraction run.
    Each response represents the answer to a specific field question, including:
    - The question asked
    - Full and short-form answers
    - Citations and confidence scores
    - Status tracking (Pending -> Processing -> To be verified -> Processed -> Verified)
    - Modification and verification history
    - User flagging for quality control
    """
    __tablename__ = "field_responses"
    __table_args__ = (
        UniqueConstraint("extraction_run_id", "field_id", name="uq_extraction_responses_run_field"),
        Index("idx_extraction_responses_run", "extraction_run_id"),
        Index("idx_extraction_responses_session", "session_id"),
        Index("idx_extraction_responses_session_category", "session_id", "category_id"),
        Index("idx_extraction_responses_status", "status"),
        Index("idx_extraction_responses_flagged", "is_flagged", postgresql_where=Column("is_flagged")),
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    extraction_run_id = Column(PGUUID(as_uuid=True), ForeignKey("extraction_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(PGUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    category_id = Column(PGUUID(as_uuid=True), ForeignKey("categories.id", ondelete="RESTRICT"), nullable=False, index=True)
    field_id = Column(PGUUID(as_uuid=True), ForeignKey("fields.id", ondelete="RESTRICT"), nullable=False, index=True)

    # Question
    question = Column(Text, nullable=True)

    # Answers
    answer = Column(Text, nullable=True)
    short_answer = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)

    # Citations (JSONB for flexible structure)
    citations = Column(JSON, nullable=True)

    # Status and workflow
    status = Column(
        Enum(*FIELD_RESPONSE_STATUS_VALUES, name="field_response_status"),
        nullable=False,
        default="Pending",
        server_default="Pending"
    )

    # Modification tracking
    is_modified = Column(Boolean, nullable=False, default=False, server_default="false")
    modified_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    modified_at = Column(DateTime(timezone=True), nullable=True)

    # Verification
    verified_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    verified_at = Column(DateTime(timezone=True), nullable=True)

    # Flagging (flexible JSON structure for flag details)
    is_flagged = Column(Boolean, nullable=False, default=False, server_default="false")
    flagged_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    flagged_at = Column(DateTime(timezone=True), nullable=True)
    flag_details = Column(JSON, nullable=True)

    # History and metadata
    answer_history = Column(JSON, nullable=True)
    metadata_json = Column("metadata", JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # Relationships
    extraction_run = relationship("ExtractionRun", lazy="joined")
    session = relationship("Session", lazy="joined")
    category = relationship("Category", lazy="joined")
    field = relationship("Field", lazy="joined")
