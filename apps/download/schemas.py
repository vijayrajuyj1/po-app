from typing import List, Optional

from pydantic import BaseModel


class SimpleFieldResponseOut(BaseModel):
    po_number: str
    category_name: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    status: Optional[str] = None


class POResponsesRequest(BaseModel):
    po_numbers: List[str]
    get_status: Optional[List[str]] = None
    response_type: str = "csv"


