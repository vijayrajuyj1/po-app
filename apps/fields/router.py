from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from apps.fields.schemas import FieldCreateSchema, FieldUpdateSchema, FieldResponseSchema
from apps.fields.service import FieldService
from models.base import get_db
from security.auth_backend import require_roles
from constants.roles import ADMIN


router = APIRouter(prefix="/api", tags=["Fields"], dependencies=[Depends(require_roles(ADMIN))])


@router.get("/fields", response_model=List[FieldResponseSchema])
async def list_all_fields(db: AsyncSession = Depends(get_db)):
    """
    List all fields across all categories.
    """
    fields = await FieldService.list_all_fields(db)
    return [
        FieldResponseSchema(
            id=str(f.id),
            category_id=str(f.category_id),
            category_name=f.category.name,
            question=f.question,
            keywords=list(f.keywords or []),
            extraction_instructions=f.extraction_instructions,
            ai_prompt=f.ai_prompt,
            sort_order=f.sort_order,
            created_at=f.created_at,
            updated_at=f.updated_at,
        )
        for f in fields
    ]


@router.post("/categories/{category_id}/fields", response_model=FieldResponseSchema, status_code=status.HTTP_201_CREATED)
async def create_field(category_id: str, payload: FieldCreateSchema, db: AsyncSession = Depends(get_db)):
    field = await FieldService.create_field(db, category_id, payload)
    return FieldResponseSchema(
        id=str(field.id),
        category_id=str(field.category_id),
        category_name=field.category.name,
        question=field.question,
        keywords=list(field.keywords or []),
        extraction_instructions=field.extraction_instructions,
        ai_prompt=field.ai_prompt,
        sort_order=field.sort_order,
        created_at=field.created_at,
        updated_at=field.updated_at,
    )


@router.get("/categories/{category_id}/fields", response_model=List[FieldResponseSchema])
async def list_fields(category_id: str, db: AsyncSession = Depends(get_db)):
    fields = await FieldService.list_fields(db, category_id)
    return [
        FieldResponseSchema(
            id=str(f.id),
            category_id=str(f.category_id),
            category_name=f.category.name,
            question=f.question,
            keywords=list(f.keywords or []),
            extraction_instructions=f.extraction_instructions,
            ai_prompt=f.ai_prompt,
            sort_order=f.sort_order,
            created_at=f.created_at,
            updated_at=f.updated_at,
        )
        for f in fields
    ]


@router.get("/fields/{field_id}", response_model=FieldResponseSchema)
async def get_field(field_id: str, db: AsyncSession = Depends(get_db)):
    f = await FieldService.get_field(db, field_id)
    return FieldResponseSchema(
        id=str(f.id),
        category_id=str(f.category_id),
        category_name=f.category.name,
        question=f.question,
        keywords=list(f.keywords or []),
        extraction_instructions=f.extraction_instructions,
        ai_prompt=f.ai_prompt,
        sort_order=f.sort_order,
        created_at=f.created_at,
        updated_at=f.updated_at,
    )


@router.patch("/fields/{field_id}", response_model=FieldResponseSchema)
async def update_field(field_id: str, payload: FieldUpdateSchema, db: AsyncSession = Depends(get_db)):
    f = await FieldService.update_field(db, field_id, payload)
    return FieldResponseSchema(
        id=str(f.id),
        category_id=str(f.category_id),
        question=f.question,
        keywords=list(f.keywords or []),
        extraction_instructions=f.extraction_instructions,
        ai_prompt=f.ai_prompt,
        sort_order=f.sort_order,
        created_at=f.created_at,
        updated_at=f.updated_at,
    )


@router.delete("/fields/{field_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_field(field_id: str, db: AsyncSession = Depends(get_db)):
    await FieldService.delete_field(db, field_id)
    return None


@router.patch("/categories/{category_id}/fields/reorder", status_code=status.HTTP_204_NO_CONTENT)
async def reorder_fields(category_id: str, body: Dict[str, List[str]], db: AsyncSession = Depends(get_db)):
    """
    Body: { "ordered_ids": ["uuid1", "uuid2", ...] }
    """
    ordered_ids = body.get("ordered_ids") or []
    if not isinstance(ordered_ids, list) or not all(isinstance(x, str) for x in ordered_ids):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ordered_ids must be a list of strings.")
    await FieldService.reorder_fields(db, category_id, ordered_ids)
    return None


