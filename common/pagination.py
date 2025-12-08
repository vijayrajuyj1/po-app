from dataclasses import dataclass
from math import ceil
from typing import Any, Iterable, List, Optional, Tuple, Type

from sqlalchemy import asc, desc
from sqlalchemy.orm import Query


@dataclass
class PaginationParams:
    """
    Reusable pagination/sorting/search parameters.
    """
    page: int = 1
    size: int = 10
    search: Optional[str] = None
    sort_by: Optional[str] = None
    sort_order: str = "asc"  # 'asc' or 'desc'


def apply_sorting(query: Query, model: Type[Any], params: PaginationParams) -> Query:
    """
    Apply dynamic sorting to a SQLAlchemy query based on params.
    """
    if params.sort_by and hasattr(model, params.sort_by):
        column = getattr(model, params.sort_by)
        direction = asc if params.sort_order.lower() != "desc" else desc
        query = query.order_by(direction(column))
    return query


def paginate_query(query: Query, params: PaginationParams) -> Tuple[List[Any], int, int, int, int]:
    """
    Apply LIMIT/OFFSET pagination and return items with paging info.
    Returns: (items, total, page, size, total_pages)
    """
    total = query.count()
    page = max(1, params.page)
    size = max(1, min(params.size, 100))
    items = query.limit(size).offset((page - 1) * size).all()
    total_pages = ceil(total / size) if size else 1
    return items, total, page, size, total_pages


