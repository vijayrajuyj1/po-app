import bcrypt


def hash_password(plain_password: str) -> str:
    """
    Hash a plaintext password using bcrypt with a secure salt.
    """
    if not isinstance(plain_password, str):
        raise TypeError("Password must be a string")
    # bcrypt expects bytes
    password_bytes = plain_password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=12)
    password_hash = bcrypt.hashpw(password_bytes, salt)
    return password_hash.decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    """
    Verify a plaintext password against a bcrypt hash.
    """
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


