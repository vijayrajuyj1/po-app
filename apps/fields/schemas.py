from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field as PydanticField, constr


class FieldCreateSchema(BaseModel):
    question: constr(strip_whitespace=True, min_length=1, max_length=500)
    keywords: List[constr(strip_whitespace=True, min_length=1)] = []
    extraction_instructions: constr(strip_whitespace=True, min_length=1)
    ai_prompt: Optional[str] = None
    sort_order: Optional[int] = PydanticField(default=None, ge=0)


class FieldUpdateSchema(BaseModel):
    question: Optional[constr(strip_whitespace=True, min_length=1, max_length=500)] = None
    keywords: Optional[List[constr(strip_whitespace=True, min_length=1)]] = None
    extraction_instructions: Optional[constr(strip_whitespace=True, min_length=1)] = None
    ai_prompt: Optional[str] = None
    sort_order: Optional[int] = PydanticField(default=None, ge=0)


class FieldResponseSchema(BaseModel):
    id: str
    category_id: str
    category_name: str
    question: str
    keywords: List[str]
    extraction_instructions: str
    ai_prompt: Optional[str]
    sort_order: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


