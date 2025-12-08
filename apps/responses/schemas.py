from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel, Field as PydanticField


class FieldResponseOut(BaseModel):
    id: str
    extractionRunId: str
    sessionId: str
    categoryId: str
    categoryName: Optional[str] = None
    fieldId: str
    question: Optional[str] = None
    answer: Optional[str] = None
    shortAnswer: Optional[str] = None
    confidenceScore: Optional[float] = None
    citations: Optional[list[dict] | dict] = None
    status: str
    isModified: bool
    modifiedBy: Optional[str] = None
    modifiedAt: Optional[datetime] = None
    verifiedBy: Optional[str] = None
    verifiedAt: Optional[datetime] = None
    createdAt: datetime
    updatedAt: datetime

    class Config:
        from_attributes = True
        populate_by_name = True


class ListResponsesQuery(BaseModel):
    sessionId: Optional[str] = None
    runId: Optional[str] = None
    categoryId: Optional[str] = None
    fieldId: Optional[str] = None
    status: Optional[str] = None
    limit: int = PydanticField(default=100, ge=1, le=500)
    offset: int = PydanticField(default=0, ge=0)


class CreateCommentRequest(BaseModel):
    content: str = PydanticField(min_length=1, max_length=4000)


class CommentOut(BaseModel):
    id: str
    fieldResponseId: str
    authorId: Optional[str] = None
    content: str
    createdAt: datetime

    class Config:
        from_attributes = True
        populate_by_name = True


class CreateIssueRequest(BaseModel):
    title: str = PydanticField(min_length=1, max_length=200)
    description: Optional[str] = None
    severity: Optional[str] = PydanticField(default=None, description="low|medium|high")
    isBlocking: bool = False


class IssueOut(BaseModel):
    id: str
    fieldResponseId: str
    reporterId: Optional[str] = None
    title: str
    description: Optional[str] = None
    severity: Optional[str] = None
    status: str
    isBlocking: bool
    createdAt: datetime
    resolvedAt: Optional[datetime] = None
    resolvedBy: Optional[str] = None

    class Config:
        from_attributes = True
        populate_by_name = True


class EditResponseRequest(BaseModel):
    answer: Optional[str] = None
    shortAnswer: Optional[str] = None
    citations: Optional[list[dict] | dict] = None
    reason: Optional[str] = None


class RegenerateResponseRequest(BaseModel):
    answer: Optional[str] = None
    shortAnswer: Optional[str] = None
    citations: Optional[list[dict] | dict] = None
    reason: Optional[str] = None


class UpdateStatusRequest(BaseModel):
    status: str


