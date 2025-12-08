import uuid
from sqlalchemy import Column, String, DateTime, Boolean, Integer, ForeignKey, JSON, Enum, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from models.base import Base


class Session(Base):
    """
    One session per PO; versioned via ExtractionRun.
    Stores vendor and PO identifiers, optional po_number, and session_key "{po_number}::v{current_version}".
    Includes soft-delete and extra_args for forward-compatible flags.
    """
    __tablename__ = "sessions"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    po_id = Column(String(50), ForeignKey("purchase_orders.id", ondelete="RESTRICT"), nullable=False, index=True)
    vendor_id = Column(String(50), ForeignKey("vendors.id", ondelete="RESTRICT"), nullable=False, index=True)
    po_number = Column(String(100), nullable=True, index=True)

    current_version = Column(Integer, nullable=False, default=0, server_default="0")
    session_key = Column(String(255), nullable=True, index=True)

    metadata_json = Column("metadata", JSON, nullable=True)
    extra_args = Column(JSON, nullable=False, default=dict, server_default="{}")

    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    updated_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    is_deleted = Column(Boolean, nullable=False, default=False, server_default="false", index=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    runs = relationship("ExtractionRun", back_populates="session", lazy="selectin", cascade="all, delete-orphan")


EXTRACTION_STATUS_VALUES = ("To be verified", "Processing", "Processed", "Verified", "Failed")


class ExtractionRun(Base):
    """
    A versioned run for a given Session. Tracks status lifecycle and selected_fields.
    """
    __tablename__ = "extraction_runs"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(PGUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    version = Column(Integer, nullable=False, index=True)
    status = Column(Enum(*EXTRACTION_STATUS_VALUES, name="extraction_run_status"), nullable=False, default="Processing", server_default="Processing")

    selected_fields = Column(JSON, nullable=True)  # normalized list of {"categoryId": ..., "fieldIds": [...]}
    metadata_json = Column("metadata", JSON, nullable=True)
    # Store the full request payload for the run (selected_fields/previous_file_urls/metadata, etc.)
    request_json = Column(JSON, nullable=True)
    # Store separate counts for uploaded vs reused files
    uploaded_files_count = Column(Integer, nullable=False, default=0, server_default="0")
    reused_files_count = Column(Integer, nullable=False, default=0, server_default="0")
    extra_args = Column(JSON, nullable=False, default=dict, server_default="{}")

    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    verified_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    session = relationship("Session", back_populates="runs", lazy="joined")
    files = relationship("DocumentFile", back_populates="run", lazy="selectin", cascade="all, delete-orphan")


class DocumentFile(Base):
    """
    Files grouped under a Session and specific ExtractionRun version.
    Supports soft-delete and marks reused vs uploaded files via metadata.
    """
    __tablename__ = "document_files"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(PGUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    extraction_run_id = Column(PGUUID(as_uuid=True), ForeignKey("extraction_runs.id", ondelete="CASCADE"), nullable=False, index=True)

    filename = Column(String(500), nullable=False)
    s3_url = Column(String(1000), nullable=False, index=True)
    mime_type = Column(String(100), nullable=False)
    size_bytes = Column(Integer, nullable=False)

    uploaded_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    metadata_json = Column("metadata", JSON, nullable=True)
    extra_args = Column(JSON, nullable=False, default=dict, server_default="{}")

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    is_deleted = Column(Boolean, nullable=False, default=False, server_default="false", index=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    session = relationship("Session", lazy="joined")
    run = relationship("ExtractionRun", back_populates="files", lazy="joined")


class ActivityLog(Base):
    """
    Append-only audit log of actions across sessions and runs.
    """
    __tablename__ = "activity_logs"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    session_id = Column(PGUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    extraction_run_id = Column(PGUUID(as_uuid=True), ForeignKey("extraction_runs.id", ondelete="SET NULL"), nullable=True, index=True)

    action = Column(String(100), nullable=False)
    description = Column(String(1000), nullable=True)
    status_from = Column(String(50), nullable=True)
    status_to = Column(String(50), nullable=True)
    metadata_json = Column("metadata", JSON, nullable=True)
    extra_args = Column(JSON, nullable=False, default=dict, server_default="{}")

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


