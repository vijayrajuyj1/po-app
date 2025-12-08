from typing import List, Optional, Tuple
from fastapi import HTTPException, status
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from apps.categories.schemas import CategoryCreateSchema, CategoryUpdateSchema
from apps.fields.schemas import FieldResponseSchema
from models.category import Category
from models.field import Field


class CategoryService:
    @staticmethod
    async def _get_next_sort_order(db: AsyncSession) -> int:
        res = await db.execute(select(func.coalesce(func.max(Category.sort_order), -1)))
        return int(res.scalar_one() or -1) + 1

    @staticmethod
    async def create_category(db: AsyncSession, payload: CategoryCreateSchema) -> Category:
        # Enforce unique category name
        existing_res = await db.execute(select(Category).where(func.lower(Category.name) == func.lower(payload.name)))
        if existing_res.scalar_one_or_none():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Category name must be unique.")

        sort_order = await CategoryService._get_next_sort_order(db)
        category = Category(name=payload.name.strip(), description=payload.description, sort_order=sort_order)
        db.add(category)
        await db.commit()
        await db.refresh(category)
        return category

    @staticmethod
    async def list_categories(db: AsyncSession) -> List[Tuple[Category, int]]:
        # Return list of (Category, field_count)
        stmt = (
            select(Category, func.count(Field.id))
            .outerjoin(Field, Field.category_id == Category.id)
            .group_by(Category.id)
            .order_by(Category.sort_order, Category.name)
        )
        res = await db.execute(stmt)
        return list(res.all())

    @staticmethod
    async def get_category_with_fields(db: AsyncSession, category_id: str) -> Category:
        stmt = select(Category).options(selectinload(Category.fields)).where(Category.id == category_id)
        res = await db.execute(stmt)
        category = res.scalar_one_or_none()
        if not category:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Category not found.")
        return category

    @staticmethod
    async def update_category(db: AsyncSession, category_id: str, payload: CategoryUpdateSchema) -> Category:
        category = await db.get(Category, category_id)
        if not category:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Category not found.")

        if payload.name and payload.name.strip().lower() != (category.name or "").strip().lower():
            # Validate unique name
            existing_res = await db.execute(
                select(Category).where(func.lower(Category.name) == func.lower(payload.name), Category.id != category_id)
            )
            if existing_res.scalar_one_or_none():
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Category name must be unique.")
            category.name = payload.name.strip()

        if payload.description is not None:
            category.description = payload.description

        if payload.sort_order is not None:
            category.sort_order = payload.sort_order

        await db.commit()
        await db.refresh(category)
        return category

    @staticmethod
    async def delete_category(db: AsyncSession, category_id: str) -> None:
        # Deleting category should cascade to fields due to FK ondelete and relationship cascade
        category = await db.get(Category, category_id)
        if not category:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Category not found.")
        await db.delete(category)
        await db.commit()

    @staticmethod
    async def reorder_categories(db: AsyncSession, ordered_ids: List[str]) -> None:
        """
        Reorder categories according to the given ordered IDs list (drag-and-drop).
        """
        # Use a single transaction with batch updates
        for index, cid in enumerate(ordered_ids):
            await db.execute(update(Category).where(Category.id == cid).values(sort_order=index))
        await db.commit()


