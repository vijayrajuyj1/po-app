from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field as PydanticField


class CreatePOFlagRequest(BaseModel):
    runId: str
    sessionId: Optional[str] = None
    reason: str = PydanticField(min_length=5, max_length=5000)


class POFlagOut(BaseModel):
    id: str
    runId: str
    sessionId: Optional[str] = None
    reason: str
    status: str
    flaggedBy: Optional[str] = None
    createdAt: datetime
    adminNote: Optional[str] = None
    resolvedBy: Optional[str] = None
    resolvedAt: Optional[datetime] = None

    class Config:
        from_attributes = True
        populate_by_name = True


class UpdatePOFlagStatusRequest(BaseModel):
    status: str  # open|processing|resolved|dismissed
    adminNote: Optional[str] = None


