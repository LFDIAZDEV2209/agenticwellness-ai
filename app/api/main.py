"""Aplicación FastAPI del servicio de IA."""

import asyncio
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    admin,
    agents,
    chat,
    content_translation,
    health,
    ingest,
    internal,
    lab_exam,
    nutrition_draft,
    patient_summary,
    proactive,
    program_narrative,
    threads,
    vita_attachments,
    vita_documents,
    vita_voice,
    voice,
)
from app.core.config import get_settings
from app.core.logging import setup_logging

# Windows: psycopg async solo funciona con SelectorEventLoop. uvicorn 0.36+
# fuerza ProactorEventLoop al arrancar, pero cualquier `asyncio.run` interno
# del proceso usa la policy — fijarla en import evita fallos de psycopg en
# helpers que crean loops propios.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging()
    # Fallo rápido: una tool que no convierte al esquema del proveedor tumba el
    # arranque (el deploy no pasa health check) en lugar de un 500 por turno.
    from app.tools.catalog import validate_tool_catalog

    validate_tool_catalog()
    settings = get_settings()
    if settings.auto_migrate and settings.database_url:
        from app.db.engine import run_db_migrations

        await asyncio.to_thread(run_db_migrations)
    if (
        settings.environment.lower() != "test"
        and settings.vita_runtime_enabled
        and settings.openai_api_key
    ):
        from app.rag.embeddings import prewarm_embeddings

        await prewarm_embeddings()
    from app.history.service import retention_worker
    from app.vita.attachments import attachment_purge_worker

    if settings.environment.lower() == "test" or not settings.database_url:
        async with retention_worker():
            yield
        return
    async with retention_worker(), attachment_purge_worker():
        yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Servicio de asistentes de Agentic Wellness IA con LangGraph y FastAPI.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS: allowlist explícita desde configuración (settings.cors_origins).
    # No hay clientes de navegador del AI Service: el backend .NET lo consume
    # server-to-server. En producción el servicio debe escuchar solo en red
    # interna (docs/observability.md) y CORS_ORIGINS define la allowlist.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = settings.api_prefix
    app.include_router(health.router, prefix=prefix)
    app.include_router(chat.router, prefix=prefix)
    app.include_router(lab_exam.router, prefix=prefix)
    app.include_router(threads.router, prefix=prefix)
    app.include_router(agents.router, prefix=prefix)
    app.include_router(ingest.router, prefix=prefix)
    app.include_router(admin.router, prefix=prefix)

    # Endpoints internos (backend → AI Service); van fuera del prefix público.
    app.include_router(internal.router)
    app.include_router(proactive.router)
    app.include_router(lab_exam.router)
    app.include_router(voice.router)
    app.include_router(vita_documents.router)
    app.include_router(vita_attachments.router)
    app.include_router(vita_voice.router)
    app.include_router(program_narrative.router)
    app.include_router(content_translation.router)
    app.include_router(nutrition_draft.router)
    app.include_router(patient_summary.router)

    return app
