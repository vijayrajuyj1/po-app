from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, func, UniqueConstraint
from sqlalchemy.orm import relationship

from models.base import Base


class PurchaseOrder(Base):
    """
    Purchase Order model with soft delete support.
    Uses custom string IDs like 'PO-2024-001'.
    """
    __tablename__ = "purchase_orders"
    __table_args__ = (
        UniqueConstraint("number", name="uq_purchase_orders_number"),
    )

    id = Column(String(50), primary_key=True, index=True)
    number = Column(String(100), nullable=False, index=True, unique=True)
    vendor_id = Column(String(50), ForeignKey("vendors.id", ondelete="RESTRICT"), nullable=False, index=True)
    # status removed; derive from latest extraction run

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # Soft delete fields
    is_deleted = Column(Boolean, nullable=False, server_default="false", index=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    vendor = relationship("Vendor", back_populates="purchase_orders", lazy="joined")


