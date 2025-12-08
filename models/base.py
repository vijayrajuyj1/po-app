import logging
from typing import AsyncGenerator

from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from settings.config import get_settings

# Alembic-friendly naming convention to ensure stable constraint names
naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=naming_convention)
Base = declarative_base(metadata=metadata)

logger = logging.getLogger(__name__)


def _make_engine():
    """
    Create SQLAlchemy ASYNC engine using DATABASE_URL from settings (psycopg3).
    """
    settings = get_settings()
    engine = create_async_engine(
        settings.DATABASE_URL or settings.build_database_url(),
        pool_pre_ping=True,  # Validate connections before use
        future=True,
    )
    logger.info("SQLAlchemy async engine created")
    return engine


# Session factory and engine are module-level singletons
engine = _make_engine()
SessionLocal = async_sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an Async SQLAlchemy session and ensures it's closed.
    """
    async with SessionLocal() as db:
        yield db


