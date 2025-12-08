from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field as PydanticField, constr
from apps.fields.schemas import FieldResponseSchema


class CategoryCreateSchema(BaseModel):
    name: constr(strip_whitespace=True, min_length=1, max_length=255)
    description: Optional[str] = None


class CategoryUpdateSchema(BaseModel):
    name: Optional[constr(strip_whitespace=True, min_length=1, max_length=255)] = None
    description: Optional[str] = None
    sort_order: Optional[int] = PydanticField(default=None, ge=0)


class CategoryResponseSchema(BaseModel):
    id: str
    name: str
    description: Optional[str]
    sort_order: int
    field_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CategoryWithFieldsSchema(BaseModel):
    id: str
    name: str
    description: Optional[str]
    sort_order: int
    fields: List[FieldResponseSchema]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


