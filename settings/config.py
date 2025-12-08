from functools import lru_cache
from typing import List, Optional
from urllib.parse import quote_plus
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized application configuration loaded from environment variables.
    Uses Pydantic's BaseSettings for robust env parsing and validation.
    """

    # App
    APP_NAME: str = "API Base"
    PRODUCT_NAME: str = "PO Contract Extractor"
    COMPANY_NAME: Optional[str] = None
    ENVIRONMENT: str = "development"  # development | staging | production
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Database (support single URL or split parts)
    DATABASE_URL: Optional[str] = None
    DB_SCHEME: str = "postgresql+psycopg"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_NAME: str = "api_base"

    # JWT / Auth
    JWT_SECRET_KEY: str = "change-this-secret-in-env"  # MUST be overridden in production
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 20  # 20 minutes for access tokens
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days for refresh tokens
    JWT_ISSUER: str = "api-base"
    JWT_AUDIENCE: str = "api-base-users"
    ENABLE_COOKIE_AUTH: bool = False  # Optionally also set tokens as HttpOnly cookies
    COOKIE_SECURE: bool = True  # Send cookies only on HTTPS in production
    COOKIE_DOMAIN: Optional[str] = None

    # CORS
    # Comma-separated origins, e.g. "http://localhost:3000,https://myapp.com"
    ALLOWED_ORIGINS: str = "http://localhost:3000"

    # Security middleware toggles
    ENABLE_RATE_LIMITER: bool = True
    RATE_LIMIT_REQUESTS: int = 100  # requests
    RATE_LIMIT_WINDOW_SECONDS: int = 60  # per this many seconds
    RATE_LIMIT_STORAGE_URI: Optional[str] = None  # e.g., "redis://localhost:6379"

    # Invitation tokens
    INVITE_TOKEN_SECRET: str = "change-this-invite-secret"  # MUST be overridden
    INVITE_TOKEN_EXPIRE_HOURS: int = 48
    INVITE_ACCEPT_PATH: str = "/account-setup"

    # Password reset tokens
    PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: int = 60
    PASSWORD_RESET_PATH: str = "/reset-password"

    # Email / SMTP
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_USE_TLS: bool = True
    SMTP_USE_SSL: bool = False
    SMTP_FROM_EMAIL: Optional[str] = None
    SMTP_FROM_NAME: Optional[str] = None

    # Frontend base fallback (used if request has no Origin header)
    FRONTEND_ORIGIN_DEFAULT: Optional[str] = None

    # Bootstrap admin (development convenience; override in env for production)
    ADMIN_EMAIL: str = "admin@example.com"
    ADMIN_PASSWORD: str = "Admin@12345"

    # Storage / S3
    AWS_S3_BUCKET: Optional[str] = None
    AWS_REGION: Optional[str] = None
    # Optional explicit credentials (boto3 can also read from environment/instance profile)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_SESSION_TOKEN: Optional[str] = None

    # Pydantic v2 settings configuration
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    @property
    def allowed_origins_list(self) -> List[str]:
        """
        Parses comma-separated origins into a list. Trims spaces, omits empties.
        """
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    def build_database_url(self) -> str:
        """
        Compose a SQLAlchemy URL from individual DB_* parts when DATABASE_URL is not provided.
        """
        if self.DATABASE_URL:
            return str(self.DATABASE_URL)
        return f"{self.DB_SCHEME}://{self.DB_USER}:{quote_plus(self.DB_PASSWORD)}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @field_validator("DEBUG", mode="before")
    def _normalize_debug(cls, v):
        # Accept "1", "true", "True", etc.
        if isinstance(v, str):
            return v.lower() in ("1", "true", "yes", "on")
        return bool(v)


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance to avoid re-parsing env on each import.
    """
    return Settings()


