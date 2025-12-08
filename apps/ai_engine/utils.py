"""Utility functions for AI Engine"""

import secrets
from pathlib import Path


def generate_unique_filename(original_filename: str, suffix_length: int = 8) -> str:
    """Generate a unique filename by adding a random suffix

    Args:
        original_filename: Original filename to make unique
        suffix_length: Length of random suffix in characters (default: 8)

    Returns:
        Filename with random suffix before extension

    Example:
        generate_unique_filename("document.pdf") -> "document_a1b2c3d4.pdf"

    TODO: Add filename preprocessing in future version:
        - Remove or replace special characters (e.g., !, @, #, $, %, etc.)
        - Replace spaces with underscores or remove them
        - Truncate filename length to a threshold (e.g., 100 characters)
        - Handle unicode characters and ensure filesystem compatibility
        - Validate and sanitize filename to prevent path traversal attacks
        - Consider using a library like `slugify` for robust filename sanitization
    """
    file_path = Path(original_filename)
    file_stem = file_path.stem
    file_extension = file_path.suffix
    random_suffix = secrets.token_hex(suffix_length // 2)  # token_hex returns 2 chars per byte
    return f"{file_stem}_{random_suffix}{file_extension}"
