import uuid
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship

from models.base import Base


class Field(Base):
    """
    Field model representing an extractable field within a category.
    - question: human-readable question for extraction
    - keywords: list of keywords to help extraction (ARRAY of TEXT)
    - extraction_instructions: required detailed extraction steps/instructions
    - ai_prompt: optional AI prompt used to generate/assist instructions
    - sort_order: integer used for ordering within category
    """
    __tablename__ = "fields"
    __table_args__ = (
        UniqueConstraint("category_id", "question", name="uq_fields_category_id_question"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    category_id = Column(UUID(as_uuid=True), ForeignKey("categories.id", ondelete="CASCADE"), nullable=False, index=True)

    question = Column(String(500), nullable=False)
    keywords = Column(ARRAY(Text), nullable=False, server_default="{}")  # empty list by default
    extraction_instructions = Column(Text, nullable=False)
    ai_prompt = Column(Text, nullable=True)
    sort_order = Column(Integer, nullable=False, server_default="0")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    category = relationship("Category", back_populates="fields", lazy="joined")


