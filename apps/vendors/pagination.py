from typing import Any, Dict, Tuple
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession


async def paginate_select(
    db: AsyncSession,
    base_stmt,
    count_stmt,
    page: int,
    size: int,
) -> Tuple[list[Any], Dict[str, int]]:
    """
    Simple async pagination helper for SQLAlchemy 2.0 style select statements.
    Returns (items, pagination_dict)
    """
    page = max(1, page or 1)
    size = max(1, min(size or 10, 100))
    total_res = await db.execute(count_stmt)
    total = int(total_res.scalar_one() or 0)
    items_res = await db.execute(base_stmt.limit(size).offset((page - 1) * size))
    items = list(items_res.scalars().all())
    return items, {"page": page, "size": size, "total": total}


