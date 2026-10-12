"""FastAPI application entrypoint for TensorLens."""
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.core.database import init_db, SessionLocal
from app.services.storage import seed_defaults
from app.api.routes import api_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle events."""
    # Ensure database schema is created and default profiles are seeded
    init_db()
    with SessionLocal() as db:
        seed_defaults(db)
    yield


app = FastAPI(
    title="TensorLens API",
    description="Layer-Streaming Large Language Model Inference & VRAM Optimization Workbench",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST API router
app.include_router(api_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Safely format unhandled errors without leaking sensitive information."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal system error occurred. Telemetry has been logged."}
    )


@app.get("/")
def root_info() -> dict:
    """Root metadata response."""
    return {
        "name": "TensorLens",
        "description": "Layer-Streaming Large Language Model Inference & VRAM Optimization Workbench",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/api/health",
    }
