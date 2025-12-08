from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from apps.categories.schemas import (
    CategoryCreateSchema,
    CategoryUpdateSchema,
    CategoryResponseSchema,
    CategoryWithFieldsSchema,
)
from apps.fields.schemas import FieldResponseSchema
from apps.categories.service import CategoryService
from models.base import get_db
from security.auth_backend import require_roles
from constants.roles import ADMIN
from models.category import Category
from models.field import Field


router = APIRouter(prefix="/api/categories", tags=["Categories"], dependencies=[Depends(require_roles(ADMIN))])


@router.post("", response_model=CategoryResponseSchema, status_code=status.HTTP_201_CREATED)
async def create_category(payload: CategoryCreateSchema, db: AsyncSession = Depends(get_db)):
    category = await CategoryService.create_category(db, payload)
    # Field count is zero on creation
    return CategoryResponseSchema(
        id=str(category.id),
        name=category.name,
        description=category.description,
        sort_order=category.sort_order,
        field_count=0,
        created_at=category.created_at,
        updated_at=category.updated_at,
    )


@router.get("", response_model=List[CategoryResponseSchema])
async def list_categories(db: AsyncSession = Depends(get_db)):
    rows = await CategoryService.list_categories(db)
    results: List[CategoryResponseSchema] = []
    for category, field_count in rows:
        results.append(
            CategoryResponseSchema(
                id=str(category.id),
                name=category.name,
                description=category.description,
                sort_order=category.sort_order,
                field_count=int(field_count or 0),
                created_at=category.created_at,
                updated_at=category.updated_at,
            )
        )
    return results


@router.get("/{category_id}", response_model=CategoryWithFieldsSchema)
async def get_category(category_id: str, db: AsyncSession = Depends(get_db)):
    category = await CategoryService.get_category_with_fields(db, category_id)
    fields = [
        FieldResponseSchema(
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
        for f in (category.fields or [])
    ]
    return CategoryWithFieldsSchema(
        id=str(category.id),
        name=category.name,
        description=category.description,
        sort_order=category.sort_order,
        fields=fields,
        created_at=category.created_at,
        updated_at=category.updated_at,
    )


@router.patch("/{category_id}", response_model=CategoryWithFieldsSchema)
async def update_category(category_id: str, payload: CategoryUpdateSchema, db: AsyncSession = Depends(get_db)):
    category = await CategoryService.update_category(db, category_id, payload)
    # Reload with fields
    category = await CategoryService.get_category_with_fields(db, category_id)
    fields = [
        FieldResponseSchema(
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
        for f in (category.fields or [])
    ]
    return CategoryWithFieldsSchema(
        id=str(category.id),
        name=category.name,
        description=category.description,
        sort_order=category.sort_order,
        fields=fields,
        created_at=category.created_at,
        updated_at=category.updated_at,
    )


@router.delete("/{category_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_category(category_id: str, db: AsyncSession = Depends(get_db)):
    await CategoryService.delete_category(db, category_id)
    return None


@router.patch("/reorder", status_code=status.HTTP_204_NO_CONTENT)
async def reorder_categories(body: Dict[str, List[str]], db: AsyncSession = Depends(get_db)):
    """
    Body: { "ordered_ids": ["uuid1", "uuid2", ...] }
    """
    ordered_ids = body.get("ordered_ids") or []
    if not isinstance(ordered_ids, list) or not all(isinstance(x, str) for x in ordered_ids):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ordered_ids must be a list of strings.")
    await CategoryService.reorder_categories(db, ordered_ids)
    return None


