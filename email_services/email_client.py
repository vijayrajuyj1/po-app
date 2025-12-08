from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Sequence

import aiosmtplib

from settings.config import get_settings


class EmailClient:
    """
    SMTP-based email client.
    Reads configuration from settings and sends HTML emails asynchronously.
    """

    async def send_email(
        self,
        to: Sequence[str],
        subject: str,
        html_body: str,
        from_email: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        if not to:
            return None

        sender_email = from_email or settings.SMTP_FROM_EMAIL or settings.ADMIN_EMAIL
        sender_name = settings.SMTP_FROM_NAME or settings.COMPANY_NAME or settings.PRODUCT_NAME or "PO System"
        from_header = f"{sender_name} <{sender_email}>"

        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = from_header
        message["To"] = ", ".join(to)

        # HTML part (no plain text variant for brevity)
        html_part = MIMEText(html_body, "html", "utf-8")
        message.attach(html_part)

        host = settings.SMTP_HOST or ""
        port = settings.SMTP_PORT or 587
        username = settings.SMTP_USERNAME or None
        password = settings.SMTP_PASSWORD or None

        # Prefer SSL if explicitly enabled; otherwise use STARTTLS when SMTP_USE_TLS is True
        if settings.SMTP_USE_SSL:
            await aiosmtplib.send(
                message,
                hostname=host,
                port=port,
                username=username,
                password=password,
                use_tls=True,
            )
            return None

        # STARTTLS flow
        await aiosmtplib.send(
            message,
            hostname=host,
            port=port,
            start_tls=settings.SMTP_USE_TLS,
            username=username,
            password=password,
        )
        return None


