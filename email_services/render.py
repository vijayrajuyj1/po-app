from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from settings.config import get_settings


_TEMPLATES_DIR = Path(__file__).parent / "templates"
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)


def render_invite_email(
    recipient_email: str,
    admin_name: str,
    admin_email: str,
    company_name: Optional[str],
    product_name: str,
    invite_url: str,
    expiry_hours: int,
) -> str:
    """
    Render the invitation email HTML body from template.
    """
    template = _env.get_template("invite.html")
    brand_company = company_name or product_name
    return template.render(
        recipient_email=recipient_email,
        admin_name=admin_name,
        admin_email=admin_email,
        company_name=brand_company,
        product_name=product_name,
        invite_url=invite_url,
        expiry_hours=expiry_hours,
    )


def render_password_reset_email(
    recipient_email: str,
    company_name: Optional[str],
    product_name: str,
    reset_url: str,
    expiry_minutes: int,
) -> str:
    """
    Render the password reset email HTML body from template.
    """
    template = _env.get_template("password_reset.html")
    brand_company = company_name or product_name
    return template.render(
        recipient_email=recipient_email,
        company_name=brand_company,
        product_name=product_name,
        reset_url=reset_url,
        expiry_minutes=expiry_minutes,
    )


