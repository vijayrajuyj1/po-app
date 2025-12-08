from typing import Any, Dict, List, Optional


def success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    """
    Standard success response envelope.
    """
    return {"message": message, "data": data}


def paginated_response(
    items: List[Any],
    total: int,
    page: int,
    size: int,
    total_pages: int,
    message: str = "Success",
) -> Dict[str, Any]:
    """
    Standard paginated response envelope.
    """
    return {
        "message": message,
        "data": items,
        "meta": {"total": total, "page": page, "size": size, "total_pages": total_pages},
    }


def error_response(message: str, details: Optional[Any] = None) -> Dict[str, Any]:
    """
    Standard error response envelope.
    """
    payload: Dict[str, Any] = {"message": message}
    if details is not None:
        payload["details"] = details
    return payload


