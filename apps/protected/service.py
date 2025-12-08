from models.user import User


def get_profile(user: User) -> dict:
    """
    Return a minimal profile payload for the current user.
    """
    return user.to_public_dict()


