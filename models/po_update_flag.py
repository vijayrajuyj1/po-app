import uuid
from sqlalchemy import Column, Text, DateTime, String, ForeignKey, func, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from models.base import Base


class POUpdateFlag(Base):
    """
    User-submitted flag indicating that a PO (extraction session/run) needs updating.
    Tracks who flagged, reason, status, and resolution metadata.
    """
    __tablename__ = "po_update_flags"
    __table_args__ = (
        Index("ix_flag_run", "extraction_run_id"),
        Index("ix_flag_session", "session_id"),
        Index("ix_flag_status", "status"),
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(PGUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    extraction_run_id = Column(PGUUID(as_uuid=True), ForeignKey("extraction_runs.id", ondelete="CASCADE"), nullable=False, index=True)

    reason = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, server_default="open")  # open|processing|resolved|dismissed

    flagged_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Resolution
    admin_note = Column(Text, nullable=True)
    resolved_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships (optional for convenient joined loads)
    run = relationship("ExtractionRun", lazy="joined")
    session = relationship("Session", lazy="joined")


