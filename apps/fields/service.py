from typing import List, Optional
from fastapi import HTTPException, status
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from apps.fields.schemas import FieldCreateSchema, FieldUpdateSchema
from models.category import Category
from models.field import Field


class FieldService:
    @staticmethod
    async def _get_next_sort_order(db: AsyncSession, category_id: str) -> int:
        res = await db.execute(
            select(func.coalesce(func.max(Field.sort_order), -1)).where(Field.category_id == category_id)
        )
        return int(res.scalar_one() or -1) + 1

    @staticmethod
    async def create_field(db: AsyncSession, category_id: str, payload: FieldCreateSchema) -> Field:
        # Ensure category exists
        category = await db.get(Category, category_id)
        if not category:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Category not found.")

        # Validate unique question within category
        existing = await db.execute(
            select(Field).where(Field.category_id == category_id, func.lower(Field.question) == func.lower(payload.question))
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Field question must be unique within the category.")

        sort_order = payload.sort_order if payload.sort_order is not None else await FieldService._get_next_sort_order(db, category_id)
        field = Field(
            category_id=category_id,
            question=payload.question.strip(),
            keywords=list(payload.keywords or []),
            extraction_instructions=payload.extraction_instructions.strip(),
            ai_prompt=payload.ai_prompt,
            sort_order=sort_order,
        )
        db.add(field)
        await db.commit()
        await db.refresh(field)
        return field

    @staticmethod
    async def list_fields(db: AsyncSession, category_id: str) -> List[Field]:
        # Ensure category exists
        category = await db.get(Category, category_id)
        if not category:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Category not found.")
        res = await db.execute(select(Field).where(Field.category_id == category_id).order_by(Field.sort_order, Field.created_at))
        return list(res.scalars().all())

    @staticmethod
    async def list_all_fields(db: AsyncSession) -> List[Field]:
        res = await db.execute(select(Field).order_by(Field.category_id, Field.sort_order, Field.created_at))
        return list(res.scalars().all())

    @staticmethod
    async def get_field(db: AsyncSession, field_id: str) -> Field:
        field = await db.get(Field, field_id)
        if not field:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field not found.")
        return field

    @staticmethod
    async def update_field(db: AsyncSession, field_id: str, payload: FieldUpdateSchema) -> Field:
        field = await db.get(Field, field_id)
        if not field:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field not found.")

        if payload.question and payload.question.strip().lower() != (field.question or "").strip().lower():
            # Validate uniqueness within category
            existing = await db.execute(
                select(Field).where(
                    Field.category_id == field.category_id,
                    func.lower(Field.question) == func.lower(payload.question),
                    Field.id != field_id,
                )
            )
            if existing.scalar_one_or_none():
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Field question must be unique within the category.")
            field.question = payload.question.strip()

        if payload.keywords is not None:
            field.keywords = list(payload.keywords)
        if payload.extraction_instructions is not None:
            field.extraction_instructions = payload.extraction_instructions.strip()
        if payload.ai_prompt is not None:
            field.ai_prompt = payload.ai_prompt
        if payload.sort_order is not None:
            field.sort_order = payload.sort_order

        await db.commit()
        await db.refresh(field)
        return field

    @staticmethod
    async def delete_field(db: AsyncSession, field_id: str) -> None:
        field = await db.get(Field, field_id)
        if not field:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field not found.")
        await db.delete(field)
        await db.commit()

    @staticmethod
    async def reorder_fields(db: AsyncSession, category_id: str, ordered_ids: List[str]) -> None:
        # Validate category
        category = await db.get(Category, category_id)
        if not category:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Category not found.")
        # Update sort orders
        for index, fid in enumerate(ordered_ids):
            await db.execute(update(Field).where(Field.id == fid, Field.category_id == category_id).values(sort_order=index))
        await db.commit()


