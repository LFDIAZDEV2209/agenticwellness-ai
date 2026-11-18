"""Exclusión por hilo entre réplicas y canales, con lock transaccional PostgreSQL."""

import asyncio
import hashlib
from contextlib import asynccontextmanager

from anyio import CancelScope
from sqlalchemy import text

from app.db.engine import engine

VITA_HANDOFF_WAIT_SECONDS = 1.0


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


@asynccontextmanager
async def wait_thread_guard(storage_thread_id: str):
    """Margen para el cierre entre canales; el hilo sigue siendo exclusivo."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + VITA_HANDOFF_WAIT_SECONDS
    while True:
        lock = thread_guard(storage_thread_id)
        try:
            await lock.__aenter__()
            break
        except ThreadBusyError:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise
            await asyncio.sleep(min(0.05, remaining))
    try:
        yield
    finally:
        with CancelScope(shield=True):
            await lock.__aexit__(None, None, None)


class LockedGraph:
    def __init__(self, graph, reason: str):
        self.graph = graph
        self.reason = reason

    def __getattr__(self, name):
        return getattr(self.graph, name)

    async def ainvoke(self, *args, config, **kwargs):
        configurable = config["configurable"]
        has_history = configurable.get("vita_history") is not None
        guard = wait_thread_guard if has_history else thread_guard
        async with guard(configurable["thread_id"]):
            if has_history:
                from app.history.service import history_service

                await history_service.before_turn_locked(self.graph, config)
            return await self.graph.ainvoke(*args, config=config, **kwargs)

    async def astream(self, *args, config, **kwargs):
        stream = self.graph.astream(*args, config=config, **kwargs)
        configurable = config["configurable"]
        has_history = configurable.get("vita_history") is not None
        guard = wait_thread_guard if has_history else thread_guard
        async with guard(configurable["thread_id"]):
            try:
                if has_history:
                    from app.history.service import history_service

                    await history_service.before_turn_locked(self.graph, config)
                async for item in stream:
                    yield item
            finally:
                with CancelScope(shield=True):
                    await stream.aclose()
