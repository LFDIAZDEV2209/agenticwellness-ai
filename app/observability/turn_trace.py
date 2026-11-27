"""Traza por turno (tarea 2.7) y persistencia en el event log append-only (tarea 2.4).

`TurnTraceCollector` es un callback de LangChain que se adjunta a la config del
grafo. Registra SOLO metadata técnica: nodo, tool, modelo, estado, latencia,
tokens, códigos de error y huellas de salidas rechazadas. Por diseño nunca lee ni guarda
entradas/salidas (mensajes del paciente, respuestas, resultados de tools,
prompts ni excepciones con su texto): lo que no se captura no puede filtrarse.

`persist_turn` inserta el turno y sus eventos en `ai.vita_turns` /
`ai.vita_turn_events` en una sola transacción, best-effort: un fallo del log
nunca rompe ni retrasa con errores el turno del canal.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from anyio import fail_after
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import ToolMessage

from app.llm.cost import SessionStats
from app.safety.vita_delivery import FALLBACK_REASONS, SAFETY_EVENT

logger = logging.getLogger(__name__)

# Eventos de política emitidos por el nodo de tools (adispatch_custom_event).
POLICY_EVENT = "vita_tool_policy"
MAX_EVENTS_PER_TURN = 200
_WRITE_TIMEOUT_SECONDS = 1
_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_NAME = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")


def safe_code(value: object) -> str | None:
    """Solo códigos tipo `snake_case` cortos; cualquier otro texto se descarta."""
    return value if isinstance(value, str) and _CODE.fullmatch(value) else None


def safe_name(value: object, fallback: str = "unknown") -> str:
    return value if isinstance(value, str) and _NAME.fullmatch(value) else fallback


def error_code(error: BaseException) -> str:
    """Código de error sin mensaje: el texto de una excepción puede contener PHI."""
    return f"exc_{type(error).__name__}".lower()[:64]


@dataclass
class TraceEvent:
    seq: int
    kind: str
    name: str
    status: str
    latency_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    error_code: str | None = None


@dataclass
class TurnTrace:
    """Turno completo listo para persistir; `to_log()` es la vista para logs."""

    thread_id: str
    channel: str
    patient_id: str | None
    agent_id: str | None
    status: str
    started_at: datetime
    finished_at: datetime
    latency_ms: int
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    error_code: str | None
    execution_id: str | None
    message_ref: str | None
    events: list[TraceEvent] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        """Métricas sin claves que enlacen con filas privadas de la conversación."""
        return {
            "channel": self.channel,
            "agent": self.agent_id,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "error_code": self.error_code,
            "events": [
                {
                    "kind": e.kind,
                    "name": e.name,
                    "status": e.status,
                    "latency_ms": e.latency_ms,
                    "tokens_in": e.tokens_in,
                    "tokens_out": e.tokens_out,
                    "error_code": e.error_code,
                }
                for e in self.events
            ],
        }


class TurnTraceCollector(AsyncCallbackHandler):
    """Recolector de eventos de un turno (instancia nueva por turno; sin estado global)."""

    raise_error = False

    def __init__(
        self,
        *,
        thread_id: str,
        channel: str,
        patient_id: str | None = None,
        agent_id: str | None = None,
        execution_id: str | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.thread_id = thread_id
        self.channel = channel
        self.patient_id = patient_id
        self.agent_id = agent_id
        self.execution_id = execution_id
        self._clock = clock
        self._started_wall = datetime.now(UTC)
        self._started = clock()
        self._open: dict[UUID, tuple[str, str, float]] = {}
        self._open_nodes: set[str] = set()
        self.events: list[TraceEvent] = []

    # -- helpers ---------------------------------------------------------------
    def _append(self, kind: str, name: str, status: str, **extra: Any) -> None:
        if len(self.events) >= MAX_EVENTS_PER_TURN:
            return
        self.events.append(TraceEvent(len(self.events) + 1, kind, name, status, **extra))

    def _close(self, run_id: UUID, status: str, **extra: Any) -> None:
        opened = self._open.pop(run_id, None)
        if opened is None:
            return
        kind, name, started = opened
        if kind == "node":
            self._open_nodes.discard(name)
        latency = int((self._clock() - started) * 1000)
        self._append(kind, name, status, latency_ms=latency, **extra)

    # -- nodos del grafo -------------------------------------------------------
    async def on_chain_start(
        self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, **kwargs
    ) -> None:
        node = (metadata or {}).get("langgraph_node")
        # Solo el runnable del nodo (no sus sub-runnables internos ni ToolNode anidado).
        if node and kwargs.get("name") == node and node not in self._open_nodes:
            self._open_nodes.add(node)
            self._open[run_id] = ("node", safe_name(node), self._clock())

    async def on_chain_end(self, outputs, *, run_id, **kwargs) -> None:
        self._close(run_id, "ok")

    async def on_chain_error(self, error, *, run_id, **kwargs) -> None:
        self._close(run_id, "error", error_code=error_code(error))

    # -- tools -----------------------------------------------------------------
    async def on_tool_start(self, serialized, input_str, *, run_id, **kwargs) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name")
        self._open[run_id] = ("tool", safe_name(name), self._clock())

    async def on_tool_end(self, output, *, run_id, **kwargs) -> None:
        status = "ok"
        if isinstance(output, ToolMessage):
            if output.status == "error":
                status = "error"
            # El estado semántico se toma de un conjunto cerrado. Los campos
            # de texto del resultado, incluido `code`, no se guardan.
            try:
                payload = json.loads(output.content) if isinstance(output.content, str) else None
            except (ValueError, TypeError):
                payload = None
            if isinstance(payload, dict) and payload.get("status") in (
                "unavailable",
                "forbidden",
                "error",
            ):
                status = "error"
        self._close(run_id, status, error_code="tool_unavailable" if status == "error" else None)

    async def on_tool_error(self, error, *, run_id, **kwargs) -> None:
        self._close(run_id, "error", error_code=error_code(error))

    # -- LLM -------------------------------------------------------------------
    async def on_chat_model_start(
        self, serialized, messages, *, run_id, metadata=None, **kwargs
    ) -> None:
        model = (metadata or {}).get("ls_model_name") or (serialized or {}).get("name")
        self._open[run_id] = ("llm", safe_name(model, "llm"), self._clock())

    async def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in getattr(response, "generations", []) or []:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    tokens_in += int(usage.get("input_tokens", 0) or 0)
                    tokens_out += int(usage.get("output_tokens", 0) or 0)
        self._close(run_id, "ok", tokens_in=tokens_in, tokens_out=tokens_out)

    async def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._close(run_id, "error", error_code=error_code(error))

    # -- política del nodo de tools -------------------------------------------
    def record_safety_block(self, digest: object, reason: object) -> None:
        if (
            isinstance(digest, str)
            and re.fullmatch(r"sha256:[a-f0-9]{64}", digest)
            and isinstance(reason, str)
            and reason in FALLBACK_REASONS
        ):
            self._append("policy", digest, "blocked", latency_ms=0, error_code=reason)

    async def on_custom_event(self, name, data, *, run_id, **kwargs) -> None:
        if not isinstance(data, dict):
            return
        if name == SAFETY_EVENT:
            self.record_safety_block(data.get("fingerprint"), data.get("reason"))
            return
        if name != POLICY_EVENT:
            return
        self._append(
            "policy",
            safe_name(data.get("tool")),
            safe_code(data.get("decision")) or "blocked",
            latency_ms=0,
            error_code=safe_code(data.get("code")),
        )

    # -- cierre ----------------------------------------------------------------
    def finish(
        self,
        *,
        status: str,
        error_code: str | None = None,
        model: str | None = None,
        message_ref: str | None = None,
        agent_id: str | None = None,
    ) -> TurnTrace:
        """Cierra el turno; los runs abiertos (cancelación/timeout) quedan `interrupted`."""
        for run_id in list(self._open):
            self._close(run_id, "interrupted")
        latency = max(0, int((self._clock() - self._started) * 1000))
        totals = SessionStats()
        cost = 0.0
        has_usage = False
        for event in self.events:
            if event.kind == "llm" and event.tokens_in is not None:
                stats = SessionStats(model=model or event.name)
                stats.add(input_tokens=event.tokens_in, output_tokens=event.tokens_out or 0)
                totals.add(input_tokens=stats.input_tokens, output_tokens=stats.output_tokens)
                has_usage |= bool(stats.total_tokens)
                cost += stats.estimate_cost_usd()
        return TurnTrace(
            thread_id=self.thread_id,
            channel=self.channel,
            patient_id=self.patient_id,
            agent_id=safe_name(agent_id or self.agent_id, "base"),
            status=status,
            started_at=self._started_wall,
            finished_at=self._started_wall + timedelta(milliseconds=latency),
            latency_ms=latency,
            tokens_in=totals.input_tokens,
            tokens_out=totals.output_tokens,
            cost_usd=round(cost, 6) if has_usage else None,
            error_code=safe_code(error_code) if error_code else None,
            execution_id=self.execution_id,
            message_ref=message_ref,
            events=list(self.events),
        )


def attach_collector(config: dict, collector: TurnTraceCollector) -> dict:
    """Añade el recolector a `config["callbacks"]` (lista) sin pisar los existentes."""
    config["callbacks"] = [*(config.get("callbacks") or []), collector]
    return config


async def persist_turn(
    trace: TurnTrace, *, session_factory: Callable[[], Any] | None = None
) -> str | None:
    """Inserta turno + eventos (append-only). Best-effort: nunca propaga errores."""
    from app.core.config import get_settings

    logger.info("vita_turn %s", json.dumps(trace.to_log(), ensure_ascii=False))
    from app.db.models import VitaTurn, VitaTurnEvent

    turn_id = str(uuid.uuid4())
    try:
        if session_factory is None:
            if not get_settings().vita_turn_log_enabled:
                return None
            from app.db.engine import async_session as session_factory
        with fail_after(_WRITE_TIMEOUT_SECONDS):
            async with session_factory() as session:
                session.add(
                    VitaTurn(
                        id=turn_id,
                        thread_id=trace.thread_id,
                        patient_id=trace.patient_id,
                        channel=trace.channel,
                        agent_id=trace.agent_id,
                        status=trace.status,
                        started_at=trace.started_at,
                        finished_at=trace.finished_at,
                        latency_ms=trace.latency_ms,
                        tokens_in=trace.tokens_in,
                        tokens_out=trace.tokens_out,
                        cost_usd=trace.cost_usd,
                        error_code=trace.error_code,
                        execution_id=trace.execution_id,
                        message_ref=trace.message_ref,
                    )
                )
                await session.flush()
                session.add_all(
                    VitaTurnEvent(
                        turn_id=turn_id,
                        seq=event.seq,
                        kind=event.kind,
                        name=event.name,
                        status=event.status,
                        latency_ms=event.latency_ms,
                        tokens_in=event.tokens_in,
                        tokens_out=event.tokens_out,
                        error_code=event.error_code,
                    )
                    for event in trace.events
                )
                await session.commit()
    except Exception as exc:
        # Solo el tipo: el texto del driver podría incluir parámetros.
        logger.warning("vita_turn_log_failed error=%s", type(exc).__name__)
        return None
    return turn_id
