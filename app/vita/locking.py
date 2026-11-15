"""Exclusión por hilo entre réplicas y canales, con lock transaccional PostgreSQL."""

import hashlib
from contextlib import asynccontextmanager

from anyio import CancelScope
from sqlalchemy import text

from app.db.engine import engine


class ThreadBusyError(Exception):
    pass


@asynccontextmanager
async def thread_guard(storage_thread_id: str):
    key = int.from_bytes(
        hashlib.sha256(storage_thread_id.encode()).digest()[:8], "big", signed=True
    )
    connection = await engine.connect()
    try:
        await connection.begin()
        acquired = await connection.scalar(
            text("SELECT pg_try_advisory_xact_lock(:key)"),
            {"key": key},
        )
        if not acquired:
            raise ThreadBusyError("vita_thread_busy")
        yield
    finally:
        # Solo contiene el advisory lock: close hace rollback y devuelve la
        # conexión al pool. Proteger ambos pasos de la cancelación ASGI/AnyIO;
        # no mantener un CancelScope abierto a través del yield del generador.
        with CancelScope(shield=True):
            await connection.close()


class LockedGraph:
    def __init__(self, graph, reason: str):
        self.graph = graph
        self.reason = reason

    def __getattr__(self, name):
        return getattr(self.graph, name)

    async def ainvoke(self, *args, config, **kwargs):
        async with thread_guard(config["configurable"]["thread_id"]):
            if config["configurable"].get("vita_history") is not None:
                from app.history.service import history_service

                await history_service.before_turn_locked(self.graph, config)
            return await self.graph.ainvoke(*args, config=config, **kwargs)

    async def astream(self, *args, config, **kwargs):
        stream = self.graph.astream(*args, config=config, **kwargs)
        async with thread_guard(config["configurable"]["thread_id"]):
            try:
                if config["configurable"].get("vita_history") is not None:
                    from app.history.service import history_service

                    await history_service.before_turn_locked(self.graph, config)
                async for item in stream:
                    yield item
            finally:
                with CancelScope(shield=True):
                    await stream.aclose()
