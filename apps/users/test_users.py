import re

from security.password_rules import validate_password_strength


def test_password_rules_valid():
    ok, msg = validate_password_strength("Password123")
    assert ok is True
    assert msg == ""


def test_password_rules_short():
    ok, msg = validate_password_strength("Pw123")
    assert ok is False
    assert "at least 8 characters" in msg


def test_password_rules_no_uppercase():
    ok, msg = validate_password_strength("password123")
    assert ok is False
    assert "uppercase" in msg


def test_password_rules_no_lowercase():
    ok, msg = validate_password_strength("PASSWORD123")
    assert ok is False
    assert "lowercase" in msg


def test_password_rules_no_digit():
    ok, msg = validate_password_strength("Passwordxxx")
    assert ok is False
    assert "numeric digit" in msg


