"""Ejecución de UNA admisión, con cancelación y fencing durable de checkpoints.

No muta grafos cacheados. Root pasa este wrapper en lugar de turn.graph al
adaptador; conserva su stream y sus filtros. Los waits no retienen transacciones.
"""

import asyncio
from contextlib import asynccontextmanager, suppress
from copy import copy

from anyio import CancelScope
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.vita.locking import LockedGraph, ThreadBusyError, thread_guard
from app.voice.store import TurnAdmission, VoiceContextError, VoiceStore


@asynccontextmanager
async def _thread_slot(store, admission, config, *, timeout=20.0):
    binding = admission.binding
    key = f"{binding.user_id}::{binding.thread_id}"
    configurable = config["configurable"]
    if (
        configurable["thread_id"] != key
        or configurable.get("user_id") != binding.user_id
        or configurable.get("voice_session_id") != binding.session_id
    ):
        raise VoiceContextError("voice_checkpoint_binding", 403)
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        await store.check_current(admission, completed_ok=admission.action == "replay")
        lock = thread_guard(key)
        try:
            await lock.__aenter__()
            break
        except ThreadBusyError:
            if asyncio.get_running_loop().time() >= deadline:
                raise VoiceContextError("voice_thread_wait_timeout", 503) from None
            await asyncio.sleep(0.05)
    try:
        yield
    finally:
        with CancelScope(shield=True):
            await lock.__aexit__(None, None, None)


async def prepare_admitted_turn(store, admission, config, saver, *, timeout=20.0):
    """ANTES de routing; supersede ya confirmado, espera al viejo sin otro LLM.

    Captura la revisión tras obtener el slot. Un borrado entre prepare y execute
    sigue invalidando el request, sin recrear una sesión ni restaurar contexto.
    """
    from app.history.service import history_service

    if admission.action not in {"run", "control", "replay"}:
        raise VoiceContextError("voice_admission_not_executable", 409)
    async with _thread_slot(store, admission, config, timeout=timeout):
        binding = admission.binding
        record = await history_service.ensure_locked(
            binding.user_id, binding.thread_id, saver, opt_in=True
        )
        await store.check_current(admission, completed_ok=admission.action == "replay")
        config["configurable"]["vita_history"] = {
            "user_id": binding.user_id,
            "thread_id": binding.thread_id,
            "revision": record.history_revision,
        }
        await history_service.activity_locked(binding.user_id, binding.thread_id)


class FencedSaver(BaseCheckpointSaver):
    def __init__(self, saver, store: VoiceStore, admission: TurnAdmission):
        super().__init__(serde=saver.serde)
        self.saver, self.store, self.admission = saver, store, admission

    @property
    def config_specs(self):
        return self.saver.config_specs

    def get_next_version(self, current, channel):
        return self.saver.get_next_version(current, channel)

    def _check_thread(self, config):
        binding = self.admission.binding
        if config["configurable"]["thread_id"] != f"{binding.user_id}::{binding.thread_id}":
            raise VoiceContextError("voice_checkpoint_binding", 403)

    async def aget_tuple(self, config):
        self._check_thread(config)
        return await self.saver.aget_tuple(config)

    async def alist(self, config, **kwargs):
        self._check_thread(config)
        async for item in self.saver.alist(config, **kwargs):
            yield item

    async def aput(self, config, *args, **kwargs):
        self._check_thread(config)
        async with self.store.write_fence(self.admission):
            return await self.saver.aput(config, *args, **kwargs)

    async def aput_writes(self, config, *args, **kwargs):
        self._check_thread(config)
        async with self.store.write_fence(self.admission):
            return await self.saver.aput_writes(config, *args, **kwargs)


class AdmittedVoiceGraph:
    """Solo action=run puede ejecutar; un wait/replay jamás inicia otro LLM."""

    def __init__(self, graph, store: VoiceStore, admission: TurnAdmission, *, wait_timeout=20.0):
        if admission.action != "run":
            raise VoiceContextError("voice_run_claim_required", 409)
        while isinstance(graph, LockedGraph):
            graph = graph.graph
        self.graph = copy(graph)
        self.graph.checkpointer = FencedSaver(graph.checkpointer, store, admission)
        self.store, self.admission = store, admission
        self.wait_timeout = wait_timeout
        self.checkpointer = self.graph.checkpointer

    @asynccontextmanager
    async def _execution(self, config):
        from app.history.service import history_service

        if config["configurable"].get("vita_history") is None:
            raise VoiceContextError("voice_history_not_prepared", 409)
        async with _thread_slot(self.store, self.admission, config, timeout=self.wait_timeout):
            await self.store.check_current(self.admission)
            await history_service.before_turn_locked(self.graph, config)
            yield

    async def _until_current(self, awaitable):
        pending = asyncio.ensure_future(awaitable)
        try:
            while not pending.done():
                await asyncio.wait({pending}, timeout=0.05)
                await self.store.check_current(self.admission)
            await self.store.check_current(self.admission)
            return await pending
        finally:
            with CancelScope(shield=True):
                if not pending.done():
                    pending.cancel()
                # Recupera errores de tareas finalizadas antes de revocar.
                # Conserva la excepción original/CancelledError del llamador.
                with suppress(asyncio.CancelledError, Exception):
                    await pending

    async def ainvoke(self, *args, config, **kwargs):
        async with self._execution(config):
            return await self._until_current(self.graph.ainvoke(*args, config=config, **kwargs))

    async def astream(self, *args, config, **kwargs):
        async with self._execution(config):
            stream = self.graph.astream(*args, config=config, **kwargs)
            try:
                while True:
                    try:
                        item = await self._until_current(anext(stream))
                    except StopAsyncIteration:
                        break
                    yield item
            finally:
                with CancelScope(shield=True):
                    await stream.aclose()
