from typing import List
from pydantic import BaseModel, constr


class AIGenerationRequestSchema(BaseModel):
    question: constr(strip_whitespace=True, min_length=1)
    keywords: List[constr(strip_whitespace=True, min_length=1)] = []


class AIGenerationResponseSchema(BaseModel):
    extraction_instructions: str


