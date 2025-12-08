import time
from collections import deque
from typing import Deque, Dict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from apps.protected.router import router as protected_router
from apps.users.router import router as users_router
from apps.team.router import router as team_router
from apps.categories.router import router as categories_router
from apps.fields.router import router as fields_router
from apps.ai.router import router as ai_router
from apps.vendors.router import router as vendors_router
from apps.pos.router import router as pos_router
from apps.extraction.router import router as extraction_router
from apps.responses.router import router as responses_router
from apps.flags.router import router as flags_router
from apps.admin.router import router as admin_router
from apps.validator.router import router as validator_router
from apps.user_dashboard.router import router as user_dashboard_router
from apps.storage.router import router as storage_router
from apps.download.router import router as download_router
from models.base import Base, engine, SessionLocal
from settings.config import get_settings
from utils.logging import setup_logging
from models.user import User, Role
from models.field_response import FieldResponse
from sqlalchemy import select
from constants.roles import ADMIN
from constants.statuses import ACTIVE
from common.hashing import hash_password


def create_app() -> FastAPI:
    """
    Application factory to build a FastAPI app with all middlewares and routers.
    """
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    app = FastAPI(
        title=settings.APP_NAME,
        debug=settings.DEBUG,
        version="1.0.0",
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["Authorization"],
    )

    # Security headers middleware (helmet-like)
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    # Optional SlowAPI rate limiter (preferred over custom limiter)
    if settings.ENABLE_RATE_LIMITER:
        # Build a default limit string from settings, using common time units
        req = settings.RATE_LIMIT_REQUESTS
        win = settings.RATE_LIMIT_WINDOW_SECONDS
        if win == 1:
            default_limit = f"{req}/second"
        elif win == 60:
            default_limit = f"{req}/minute"
        elif win == 3600:
            default_limit = f"{req}/hour"
        elif win == 86400:
            default_limit = f"{req}/day"
        else:
            # Fallback: human-friendly seconds window
            default_limit = f"{req} per {win} seconds"

        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[default_limit],
            storage_uri=settings.RATE_LIMIT_STORAGE_URI or None,
        )
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        app.add_middleware(SlowAPIMiddleware)

    # Routers
    app.include_router(users_router)
    app.include_router(protected_router)
    app.include_router(team_router)
    app.include_router(categories_router)
    app.include_router(fields_router)
    app.include_router(ai_router)
    app.include_router(vendors_router)
    app.include_router(pos_router)
    app.include_router(extraction_router)
    app.include_router(responses_router)
    app.include_router(flags_router)
    app.include_router(admin_router)
    app.include_router(validator_router)
    app.include_router(user_dashboard_router)
    app.include_router(storage_router)
    app.include_router(download_router)
    # Ensure tables exist (for local/dev). In prod, use Alembic migrations.
    @app.on_event("startup")
    async def on_startup():
        # Create tables in dev when using the async engine
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Seed a default admin user if not present
        async with SessionLocal() as db:
            # ensure ADMIN role exists
            role_res = await db.execute(select(Role).where(Role.name == ADMIN))
            admin_role = role_res.scalar_one_or_none()
            if not admin_role:
                # roles are seeded by migration, but create if missing
                admin_role = Role(name=ADMIN)
                db.add(admin_role)
                await db.commit()
                await db.refresh(admin_role)

            user_res = await db.execute(select(User).where(User.email == settings.ADMIN_EMAIL))
            admin_user = user_res.scalar_one_or_none()
            if not admin_user:
                admin_user = User(
                    email=settings.ADMIN_EMAIL,
                    name="Administrator",
                    password_hash=hash_password(settings.ADMIN_PASSWORD),
                    is_verified=True,
                    status=ACTIVE,
                )
                admin_user.roles = [admin_role]
                db.add(admin_user)
                await db.commit()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()


