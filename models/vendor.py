from sqlalchemy import Column, String, DateTime, Boolean, func
from sqlalchemy.orm import relationship

from models.base import Base


class Vendor(Base):
    """
    Vendor model with soft delete support.
    Uses custom string IDs like 'VND-001'.
    """
    __tablename__ = "vendors"

    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    address = Column(String(500), nullable=True)
    contact_email = Column(String(255), nullable=True, index=True)
    phone = Column(String(50), nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # Soft delete fields
    is_deleted = Column(Boolean, nullable=False, server_default="false", index=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # Relationship to POs
    purchase_orders = relationship(
        "PurchaseOrder",
        back_populates="vendor",
        lazy="selectin",
    )


