from models.user import User


def sanitize_user(user: User, roles: list[str] | None = None) -> dict:
    """
    Convert a User ORM object into a public-safe dict.
    """
    data = user.to_public_dict()
    if roles is not None:
        data["roles"] = roles
    else:
        data.setdefault("roles", [])
    return data


