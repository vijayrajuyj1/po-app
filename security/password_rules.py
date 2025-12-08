import re
from typing import Tuple


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """
    Validate password strength against required policy:
      - At least 8 characters
      - At least one uppercase letter
      - At least one lowercase letter
      - At least one digit
    Returns a tuple (is_valid, message). Message is empty when valid.
    """
    if not isinstance(password, str) or not password:
        return False, "Password is required."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must include at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must include at least one lowercase letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must include at least one numeric digit."
    return True, ""


def assert_passwords_match(password: str, confirm_password: str) -> None:
    """
    Raise ValueError if the password and confirmation do not match.
    """
    if password != confirm_password:
        raise ValueError("Password and confirmation do not match.")


