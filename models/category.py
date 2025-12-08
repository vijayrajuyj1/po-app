import uuid
from sqlalchemy import Column, String, Text, DateTime, Integer, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from models.base import Base


class Category(Base):
    """
    Category model representing a logical group of fields.
    - name: unique name for the category
    - description: optional description
    - sort_order: integer used for drag-and-drop ordering in UI
    """
    __tablename__ = "categories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    sort_order = Column(Integer, nullable=False, server_default="0")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # Relationship to fields with cascade delete and ordering by sort_order
    fields = relationship(
        "Field",
        back_populates="category",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Field.sort_order",
        lazy="selectin",
    )


