from fastapi import HTTPException, status


def http_conflict(detail: str) -> HTTPException:
    """
    409 Conflict response shortcut.
    """
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


def http_bad_request(detail: str) -> HTTPException:
    """
    400 Bad Request response shortcut.
    """
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def http_unauthorized(detail: str = "Invalid credentials") -> HTTPException:
    """
    401 Unauthorized response shortcut.
    """
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


