"""Rutas de chat: respuesta completa y streaming SSE."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import Any

from anyio import CancelScope
from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy.exc import SQLAlchemyError

from app.agents.runtime_registry import AgentRuntimeError
from app.agents.runtime_registry import registry as runtime_registry
from app.api.deps import get_db_session, get_graph
from app.api.message_text import extract_message_text
from app.api.schemas import (
    ChatAttachmentResult,
    ChatRequest,
    ChatResponse,
    ChatSuggestion,
    FeedbackRequest,
    FeedbackResponse,
)
from app.api.security import require_internal_key
from app.api.streaming import ClosingStreamingResponse as StreamingResponse
from app.core.config import get_settings
from app.core.errors import CoppAiError
from app.core.logging import get_logger
from app.graph.language import detect_turn_language
from app.history.service import history_service
from app.memory.adaptive import AdaptiveMemoryService
from app.observability.executions import ExecutionTracker
from app.observability.turn_trace import (
    TurnTraceCollector,
    attach_collector,
    error_code,
    persist_turn,
)
from app.safety.guardrails import RateLimiter
from app.safety.vita_delivery import fallback_message, safety_reason
from app.vita.locking import ThreadBusyError
from app.voice.store import VoiceContextError

logger = get_logger(__name__)

# Canal interno: solo el backend .NET (con X-Internal-Key) puede invocar el
# chat. El frontend NUNCA llama a este servicio directo.
router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(require_internal_key)])

_rate_limiters: dict[str, RateLimiter] = {}
_rate_limiters_lock = threading.Lock()


_MAX_ACTIVE_LIMITERS = 10_000


def _prune_inactive_limiters() -> None:
    """Evita la fuga de memoria: elimina limiters sin actividad reciente cuando
    el dict supera el umbral."""
    if len(_rate_limiters) < _MAX_ACTIVE_LIMITERS:
        return
    for key, limiter in list(_rate_limiters.items()):
        if limiter.remaining == limiter.max_requests:  # sin actividad en la ventana
            del _rate_limiters[key]


def _rate_limiter_for(client_key: str) -> RateLimiter:
    with _rate_limiters_lock:
        _prune_inactive_limiters()
        limiter = _rate_limiters.get(client_key)
        if limiter is None:
            limiter = RateLimiter()
            _rate_limiters[client_key] = limiter
        return limiter


def _client_key(request: Request, user_id: str | None) -> str:
    """Clave de rate limit: el `user_id` real (inyectado por el backend desde
    el JWT) cuando existe; si no, la IP como fallback (p. ej. health/uso sin
    identidad). Varios usuarios tras una misma IP no se penalizan entre sí."""
    if user_id:
        return f"user:{user_id}"
    return f"ip:{request.client.host}" if request.client else "ip:unknown"


def _check_rate_limit(request: Request, user_id: str | None) -> None:
    if not _rate_limiter_for(_client_key(request, user_id)).check():
        raise HTTPException(
            status_code=429,
            detail="Demasiadas solicitudes. Espera un momento antes de continuar.",
        )


def _storage_thread_id(user_id: str | None, thread_id: str) -> str:
    """Clave interna del thread en el checkpointer.

    Se antepone el `user_id` real (proviene del backend, que lo deriva del
    JWT): aunque dos usuarios enviaran el mismo `thread_id` desde el cliente,
    jamás comparten historial ni estado — aislamiento de memoria/threads
    garantizado a nivel de almacenamiento, no solo de UI.
    """
    if user_id:
        return f"{user_id}::{thread_id}"
    return thread_id


def _normalize_language(language: str | None) -> str | None:
    """Normaliza el idioma de la app ('es' | 'en'); None/vacío/desconocido ⇒ None.

    Mismo convenio que `lab_exam._normalize_language`, pero sin default a 'es':
    aquí None significa "el frontend no pidió idioma" y el agente mantiene su
    regla de espejo (comportamiento byte-idéntico al estándar).
    """
    value = (language or "").strip().lower()
    return value if value in ("es", "en") else None


def _build_config(
    thread_id: str,
    request: ChatRequest | None = None,
    recursion_limit: int | None = None,
) -> dict:
    settings = get_settings()
    storage_thread_id = _storage_thread_id(
        request.user_id if request is not None else None,
        thread_id,
    )
    configurable = {"thread_id": storage_thread_id}

    # Aislamiento multi-agente: el checkpointer y las tools separan el estado
    # por usuario/paciente/instancia además del thread.
    if request is not None:
        # El canal viene fijado en el request (no de `agent`, que el routing de
        # Vita reescribe con el especialista elegido).
        configurable["message_channel"] = request.channel
        configurable["vita_safety"] = _uses_vita_safety(request)
        if request.channel_message_id:
            configurable["channel_message_id"] = request.channel_message_id
        if request.user_id:
            configurable["user_id"] = request.user_id
        if request.patient_id:
            configurable["patient_id"] = request.patient_id
        if request.agent_instance_id:
            configurable["agent_instance_id"] = request.agent_instance_id
        # Contexto del control de programa (backend .NET): viaja por config y
        # NUNCA persiste en el checkpointer — es guía de UN turno. None cuando
        # el turno no corresponde a un control (comportamiento byte-idéntico).
        configurable["control_context"] = (
            request.control_context.model_dump() if request.control_context is not None else None
        )
        # Idioma de la app del paciente: señal EXPLÍCITA por turno (permite
        # cambiar de idioma a mitad de conversación). Viaja por `configurable`
        # igual que control_context — nunca persiste en el checkpointer.
        configurable["language"] = _normalize_language(request.language)

    return {
        "configurable": configurable,
        "recursion_limit": recursion_limit or settings.recursion_limit,
    }


def _vita_channel_enabled(request: ChatRequest) -> bool:
    """WhatsApp entra al runtime Vita solo con su flag y un usuario vinculado.

    Un vínculo sin cuenta de la app (sin `user_id`) conserva el perfil legado en vez
    de recibir 403: el paciente sigue teniendo respuesta educativa por WhatsApp.
    """
    if request.channel != "whatsapp":
        return True
    return get_settings().vita_whatsapp_enabled and bool(request.user_id)


def _is_vita_request(request: ChatRequest) -> bool:
    return (
        get_settings().vita_runtime_enabled
        and request.supports_vita
        and _vita_channel_enabled(request)
    )


def _uses_vita_safety(request: ChatRequest) -> bool:
    # La política no desaparece al desactivar el routing Vita. El playground
    # ERP no se convierte en un canal de paciente por el flag global.
    return request.supports_vita is True or request.channel == "whatsapp"


def _provider_fallback_state(request: ChatRequest) -> dict:
    language = detect_turn_language({"input": request.message}) or request.language
    return {"messages": [fallback_message("provider_failure", language)]}


def _last_ai_message(state: dict | None) -> AIMessage | None:
    messages = (state or {}).get("messages", [])
    start = (state or {}).get("turn_start_index", 0)
    if (
        len(messages) > start
        and isinstance(messages[-1], AIMessage)
        and not messages[-1].tool_calls
    ):
        return messages[-1]
    return None


def _history_opt_in(request: ChatRequest) -> bool:
    # WhatsApp comparte el hilo Vita de la app: participa del mismo corte
    # mensual/revisión para no escribir sobre un historial pendiente de reset.
    return request.supports_vita and _vita_channel_enabled(request)


def _redact_execution(request: ChatRequest) -> bool:
    return _history_opt_in(request) or request.channel == "whatsapp"


async def _prepare_chat_history(request: ChatRequest, thread_id: str, graph):
    if not _history_opt_in(request):
        return None
    if not request.user_id:
        raise HTTPException(status_code=403, detail="vita_identity_required")
    config = _build_config(thread_id, request)
    try:
        await history_service.prepare_turn(request.user_id, thread_id, graph.checkpointer, config)
    except ThreadBusyError as exc:
        raise HTTPException(status_code=409, detail="vita_thread_busy") from exc
    except VoiceContextError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="history_state_unavailable") from exc
    return config["configurable"]["vita_history"]


async def _prepare_attachments(request: ChatRequest, thread_id: str, session):
    """Valida acceso y extrae el texto de los adjuntos efímeros del turno.

    El paciente sale del contexto autenticado que envía el backend, nunca del
    modelo. Un adjunto ajeno, expirado o inexistente es indistinguible (404).
    """
    if not request.attachment_ids:
        return None
    if request.channel != "text":
        # Adjuntos por WhatsApp pertenecen a 6.6; aquí solo el chat de la app.
        raise HTTPException(status_code=403, detail="vita_identity_required")
    if not get_settings().vita_attachments_enabled:
        raise HTTPException(status_code=403, detail="vita_attachments_disabled")
    from app.tools.retrieval import authorized_patient_id

    patient_id = authorized_patient_id(
        {"user_id": request.user_id, "patient_id": request.patient_id, "message_channel": "text"}
    )
    if not _history_opt_in(request) or patient_id is None:
        raise HTTPException(status_code=403, detail="vita_identity_required")
    from app.vita.attachment_context import build_turn_context, extract_attachment
    from app.vita.attachments import AttachmentStore

    ids = list(dict.fromkeys(request.attachment_ids))
    try:
        rows = await AttachmentStore(session).get_many(
            ids, patient_id=uuid.UUID(patient_id), user_id=request.user_id
        )
    except SQLAlchemyError as exc:
        logger.warning("vita_attachment_store_unavailable error=%s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="attachment_store_unavailable") from None
    if len(rows) != len(ids) or any(row.thread_id not in (None, thread_id) for row in rows):
        raise HTTPException(status_code=404, detail="vita_attachment_not_found")
    extracted = await asyncio.gather(*(extract_attachment(row) for row in rows))
    return build_turn_context(extracted)


def _attachment_results(context) -> list[ChatAttachmentResult]:
    if context is None:
        return []
    return [
        ChatAttachmentResult(attachment_id=s.attachment_id, kind=s.kind, readable=s.readable)
        for s in context.summaries
    ]


def _with_transcript_lock(graph, *, force=False):
    if force or get_settings().vita_transcript_sync_enabled:
        from app.vita.locking import LockedGraph

        if not isinstance(graph, LockedGraph):
            return LockedGraph(graph, "legacy")
    return graph


async def _resolve_graph(
    request: ChatRequest,
    session,
    base_graph=None,
    thread_id: str | None = None,
) -> tuple[Any, int, str | None, str | None, str | None]:
    """Resuelve el grafo a ejecutar.

    Si `agent_type_id` viene, usa el runtime multi-agente (compilado/cacheado
    por el registry) y su recursion_limit configurado; si no, el grafo base
    (perfil `agent`).

    Returns:
        (graph, recursion_limit, version_id, provider, model)
    """
    if _is_vita_request(request):
        from app.vita.client import VitaAccessError
        from app.vita.runtime import resolve_chat_turn

        if not request.user_id:
            raise HTTPException(status_code=403, detail="vita_identity_required")
        try:
            turn = await resolve_chat_turn(request, session, base_graph, thread_id)
        except VitaAccessError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        request.agent = turn.agent_id
        request.agent_type_id = turn.agent_id
        return turn.graph, turn.recursion_limit, turn.version_id, turn.provider, turn.model

    if request.agent_type_id:
        try:
            compiled = await runtime_registry.get_agent(request.agent_type_id, session)
            cfg = compiled.runtime_config
            return (
                compiled.graph,
                compiled.runtime_config.recursion_limit,
                compiled.version_id,
                cfg.provider,
                cfg.model,
            )
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return None, get_settings().recursion_limit, None, None, None


def _extract_answer(state: dict) -> str:
    messages = state.get("messages", [])
    if not messages:
        return ""
    last = messages[-1]
    return extract_message_text(getattr(last, "content", ""))


def _normalize_suggestions(raw: Any) -> list[ChatSuggestion]:
    """Normaliza las sugerencias crudas del estado a `ChatSuggestion`.

    Las entradas malformadas se descartan de forma defensiva (nunca deben
    tumbar la respuesta del chat ni el evento done del stream).
    """
    suggestions: list[ChatSuggestion] = []
    if not isinstance(raw, list):
        return suggestions
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            suggestions.append(ChatSuggestion.model_validate(item))
        except Exception:
            logger.debug("Sugerencia de chat descartada (malformada): %s", item)
    return suggestions


async def _bounded_stream(graph, inputs, *, config, vita: bool = False):
    settings = get_settings()
    timeout = settings.vita_turn_timeout if vita else settings.llm_invoke_timeout
    stream = graph.astream(
        inputs,
        config=config,
        stream_mode=["updates", "messages", "values", "tasks"],
    )
    try:
        async with asyncio.timeout(timeout):
            async for item in stream:
                yield item
    finally:
        with CancelScope(shield=True):
            await stream.aclose()


def _log_execution_error(message: str, *, vita: bool) -> None:
    if vita:
        logger.error(message)
    else:
        logger.exception(message)


def _turn_collector(request: ChatRequest, thread_id: str, execution_id: str) -> TurnTraceCollector:
    """Traza técnica del turno (sin contenido); el canal sale de la política única."""
    return TurnTraceCollector(
        thread_id=_storage_thread_id(request.user_id, thread_id),
        channel=request.channel or "text",
        patient_id=request.patient_id or request.user_id,
        agent_id=request.agent_type_id or request.agent or "base",
        execution_id=execution_id,
    )


def _turn_outcome(state: dict | None) -> tuple[str | None, str | None]:
    """(message_ref, error_code) del último AIMessage del turno, si existe."""
    state = state or {}
    messages = state.get("messages", [])[state.get("turn_start_index", 0) :]
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            code = message.additional_kwargs.get("error_code")
            if code not in ("tool_catalog_unavailable", "tool_budget_exceeded"):
                code = None
            return message.id, code
    return None, None


async def _close_turn(
    collector: TurnTraceCollector,
    *,
    status: str,
    error: str | None,
    state: dict | None,
    model: str | None,
) -> None:
    message_ref, state_error = _turn_outcome(state if status == "completed" else None)
    reason = safety_reason(_last_ai_message(state))
    if reason:
        status = "failed" if reason in {"provider_failure", "empty_answer"} else "rejected"
        error = reason
    elif status == "completed" and (state or {}).get("guardrail", {}).get("safe") is False:
        status, error = "rejected", "guardrail_rejected"
    trace = collector.finish(
        status=status,
        error_code=error or state_error,
        model=model,
        message_ref=message_ref,
        agent_id=(state or {}).get("agent"),
    )
    await persist_turn(trace)


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
    session=Depends(get_db_session),
    base_graph=Depends(get_graph),
) -> ChatResponse:
    """Ejecuta el agente de principio a fin y devuelve la respuesta completa."""
    _check_rate_limit(http_request, request.user_id)
    thread_id = request.thread_id or str(uuid.uuid4())

    # Correlation ID propagado por el backend (X-Correlation-ID) para trazar
    # la cadena Frontend → Backend → AI en los logs.
    correlation_id = http_request.headers.get("x-correlation-id")
    if correlation_id:
        logger.info(
            "chat: correlation_id=%s user=%s thread=%s agent=%s",
            correlation_id,
            request.user_id,
            thread_id,
            request.agent_type_id or request.agent or "base",
        )

    attachment_context = await _prepare_attachments(request, thread_id, session)
    history_binding = await _prepare_chat_history(request, thread_id, base_graph)
    graph, recursion_limit, version_id, provider, model = await _resolve_graph(
        request, session, base_graph, thread_id
    )
    if graph is None:
        graph = base_graph
    graph = _with_transcript_lock(graph, force=history_binding is not None)

    protected = _uses_vita_safety(request)
    tracker = ExecutionTracker(session, redact_content=_redact_execution(request) or protected)
    execution_id = await tracker.start(
        thread_id=thread_id,
        agent_type_id=request.agent_type_id or request.agent or "base",
        version_id=version_id,
        agent_instance_id=request.agent_instance_id,
        user_id=request.user_id,
        provider=provider,
        model=model,
        message=request.message,
    )
    started = time.perf_counter()
    config = _build_config(thread_id, request, recursion_limit)
    if history_binding is not None:
        config["configurable"]["vita_history"] = history_binding
    config["configurable"]["execution_id"] = execution_id
    if attachment_context is not None:
        config["configurable"]["vita_attachments"] = attachment_context
    collector = _turn_collector(request, thread_id, execution_id)
    attach_collector(config, collector)
    turn_status, turn_error, result = "failed", "exc_unknown", None
    try:
        result = await asyncio.wait_for(
            graph.ainvoke(
                {
                    "input": request.message,
                    "agent": request.agent or request.agent_type_id or "base",
                },
                config=config,
            ),
            timeout=(
                get_settings().vita_turn_timeout
                if _is_vita_request(request)
                else get_settings().llm_invoke_timeout
            ),
        )
        if protected and _last_ai_message(result) is None:
            result = _provider_fallback_state(request)
        await tracker.complete(
            execution_id,
            result_state=result,
            model=model,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        turn_status, turn_error = "completed", None
    except asyncio.CancelledError:
        turn_status, turn_error = "cancelled", "cancelled"
        await asyncio.shield(
            tracker.fail(
                execution_id,
                error="cancelled",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        )
        await asyncio.shield(session.commit())
        raise
    except (VoiceContextError, ThreadBusyError) as exc:
        turn_status, turn_error = "rejected", "history_turn_rejected"
        await tracker.fail(execution_id, error="history_turn_rejected", latency_ms=0)
        await session.commit()
        raise HTTPException(status_code=getattr(exc, "status", 409), detail=str(exc)) from exc
    except Exception as exc:
        turn_error = "turn_timeout" if isinstance(exc, TimeoutError) else error_code(exc)
        _log_execution_error(
            "Error ejecutando el grafo", vita=_redact_execution(request) or protected
        )
        await tracker.fail(
            execution_id,
            error=turn_error if protected else str(exc),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        if not protected:
            await session.commit()
            detail = (
                str(exc)
                if isinstance(exc, CoppAiError)
                else "No fue posible procesar la solicitud."
            )
            raise HTTPException(status_code=500, detail=detail) from exc
        result = _provider_fallback_state(request)

    finally:
        # Event log append-only: best-effort, nunca cambia la respuesta del canal.
        await asyncio.shield(
            _close_turn(collector, status=turn_status, error=turn_error, state=result, model=model)
        )
    await session.commit()

    tools_used = list(result.get("tools_used", []))
    # Señal del control de programa: SOLO cuando el turno trae contexto Y el
    # modelo llamó la tool no-op de rechazo explícito. Sin contexto ⇒ None.
    control_signal = (
        "declined"
        if request.control_context is not None and "mark_control_declined" in tools_used
        else None
    )

    return ChatResponse(
        thread_id=thread_id,
        answer=_extract_answer(result),
        actions=result.get("app_actions", []),
        sources=result.get("data_sources", []),
        version_id=version_id,
        agent=result.get("agent") or request.agent or request.agent_type_id or "base",
        tools_used=tools_used,
        model=result.get("provider") or model,
        execution_id=execution_id,
        suggestions=_normalize_suggestions(result.get("suggestions", [])),
        control_signal=control_signal,
        attachments=_attachment_results(attachment_context),
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(
    request: FeedbackRequest,
    session=Depends(get_db_session),
) -> FeedbackResponse:
    """Registra el feedback del usuario y alimenta la adaptive memory.

    Si vienen `agent_type_id` + `trigger` + `response`, el rating se convierte
    en una experiencia aprendida (patrón trigger → response con outcome
    success/error según el rating). El feedback siempre se persiste.
    """
    adaptive = AdaptiveMemoryService(session)
    await adaptive.save_feedback(
        thread_id=request.thread_id,
        rating=request.rating,
        comment=request.comment,
        user_id=request.user_id,
        execution_id=request.execution_id,
    )

    experience_saved = False
    outcome: str | None = None
    if request.agent_type_id and request.trigger and request.response:
        try:
            experience = await adaptive.save_experience(
                agent_type_id=request.agent_type_id,
                version_id=None,
                trigger=request.trigger,
                response=request.response,
                rating=request.rating,
            )
            experience_saved = True
            outcome = experience.outcome
        except Exception:
            logger.exception(
                "No se pudo guardar la experiencia para el agente %s",
                request.agent_type_id,
            )

    await session.commit()
    return FeedbackResponse(
        thread_id=request.thread_id,
        rating=request.rating,
        experience_saved=experience_saved,
        outcome=outcome,
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
    session=Depends(get_db_session),
    base_graph=Depends(get_graph),
) -> StreamingResponse:
    """Streaming del agente vía Server-Sent Events (tokens + nodos en vivo)."""
    _check_rate_limit(http_request, request.user_id)
    thread_id = request.thread_id or str(uuid.uuid4())

    correlation_id = http_request.headers.get("x-correlation-id")
    if correlation_id:
        logger.info(
            "chat/stream: correlation_id=%s user=%s thread=%s agent=%s",
            correlation_id,
            request.user_id,
            thread_id,
            request.agent_type_id or request.agent or "base",
        )

    attachment_context = await _prepare_attachments(request, thread_id, session)
    history_binding = await _prepare_chat_history(request, thread_id, base_graph)
    graph, recursion_limit, version_id, provider, model = await _resolve_graph(
        request, session, base_graph, thread_id
    )
    if graph is None:
        graph = base_graph
    graph = _with_transcript_lock(graph, force=history_binding is not None)

    protected = _uses_vita_safety(request)
    tracker = ExecutionTracker(session, redact_content=_redact_execution(request) or protected)
    execution_id = await tracker.start(
        thread_id=thread_id,
        agent_type_id=request.agent_type_id or request.agent or "base",
        version_id=version_id,
        agent_instance_id=request.agent_instance_id,
        user_id=request.user_id,
        provider=provider,
        model=model,
        message=request.message,
    )

    started = time.perf_counter()
    flow_trace: list[dict[str, object]] = []
    collector = _turn_collector(request, thread_id, execution_id)

    async def event_generator():
        final_state: dict | None = None
        error: str | None = None
        turn_status, turn_error = "cancelled", "stream_interrupted"
        # Trazabilidad de nodos para la visualización de flujos en vivo:
        # `stream_mode="tasks"` emite inicio (trae `triggers`) y fin (trae
        # `result`) de cada nodo; se traduce a eventos `flow` con step/ts.
        flow_step = 0
        running_tasks: dict[str, dict[str, Any]] = {}
        stream = None
        try:
            yield "event: start\ndata: {}\n\n"
            config = _build_config(thread_id, request, recursion_limit)
            if history_binding is not None:
                config["configurable"]["vita_history"] = history_binding
            config["configurable"]["execution_id"] = execution_id
            if attachment_context is not None:
                config["configurable"]["vita_attachments"] = attachment_context
            attach_collector(config, collector)
            agent_key = request.agent or request.agent_type_id or "base"

            stream = _bounded_stream(
                graph,
                {"input": request.message, "agent": agent_key},
                config=config,
                vita=_is_vita_request(request),
            )
            async for mode, chunk in stream:
                if mode == "values":
                    # Estado completo tras cada super-step; el último es el final.
                    if isinstance(chunk, dict):
                        final_state = chunk
                    continue
                if mode == "messages":
                    if protected:
                        continue  # Nunca emitir callbacks crudos del modelo o del planificador.
                    message_chunk, metadata = chunk
                    # Solo transmitir tokens de AIMessage/AIMessageChunk, no HumanMessage
                    if not isinstance(message_chunk, (AIMessage, AIMessageChunk)):
                        continue
                    content = getattr(message_chunk, "content", "")
                    # Claude devuelve content como lista de bloques, extraer texto
                    if isinstance(content, list):
                        text = "".join(
                            block.get("text", "")
                            for block in content
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    else:
                        text = content if isinstance(content, str) else ""
                    if text:
                        token_payload: dict[str, object] = {"type": "token", "content": text}
                        # Nodo que emite el token (el front marca el nodo activo).
                        node_name = (
                            metadata.get("langgraph_node") if isinstance(metadata, dict) else None
                        )
                        if node_name:
                            token_payload["node"] = node_name
                        payload = json.dumps(token_payload, ensure_ascii=False)
                        yield f"event: message\ndata: {payload}\n\n"
                elif mode == "updates":
                    for node_name in chunk:
                        payload = json.dumps({"type": "node", "node": node_name})
                        yield f"event: node\ndata: {payload}\n\n"
                elif mode == "tasks":
                    if not isinstance(chunk, dict):
                        continue
                    node_name = str(chunk.get("name") or "")
                    task_id = str(chunk.get("id") or "")
                    # Tasks internos de LangGraph (`__start__`, etc.) fuera.
                    if not node_name or not task_id or node_name.startswith("__"):
                        continue
                    now_ms = int(time.time() * 1000)
                    if "triggers" in chunk:
                        flow_step += 1
                        running_tasks[task_id] = {
                            "node": node_name,
                            "started_pc": time.perf_counter(),
                            "step": flow_step,
                            "ts": now_ms,
                        }
                        flow_trace.append(
                            {
                                "node": node_name,
                                "phase": "start",
                                "step": flow_step,
                                "ts": now_ms,
                            }
                        )
                        payload = json.dumps(
                            {
                                "type": "flow",
                                "node": node_name,
                                "phase": "start",
                                "step": flow_step,
                                "ts": now_ms,
                            },
                            ensure_ascii=False,
                        )
                        yield f"event: flow\ndata: {payload}\n\n"
                    else:
                        info = running_tasks.pop(task_id, None)
                        entry: dict[str, object] = {
                            "node": node_name,
                            "phase": "end",
                            "step": info["step"] if info else flow_step,
                            "ts": now_ms,
                        }
                        if info is not None:
                            entry["duration_ms"] = int(
                                (time.perf_counter() - float(info["started_pc"])) * 1000
                            )
                        flow_trace.append(entry)
                        payload = json.dumps({"type": "flow", **entry}, ensure_ascii=False)
                        yield f"event: flow\ndata: {payload}\n\n"
            if protected and _last_ai_message(final_state) is None:
                final_state = _provider_fallback_state(request)
            turn_status, turn_error = (
                ("completed", None) if final_state is not None else ("failed", "stream_interrupted")
            )
        except Exception as exc:
            turn_status, turn_error = (
                "failed",
                ("turn_timeout" if isinstance(exc, TimeoutError) else error_code(exc)),
            )
            _log_execution_error("Error en el stream del agente", vita=_redact_execution(request))
            error = "No fue posible procesar la solicitud."
            if protected and not isinstance(exc, (VoiceContextError, ThreadBusyError)):
                final_state = _provider_fallback_state(request)
            else:
                final_state = None
                payload = json.dumps({"type": "error", "error": error}, ensure_ascii=False)
                yield f"event: error\ndata: {payload}\n\n"
        finally:
            with CancelScope(shield=True):
                try:
                    if stream is not None:
                        await stream.aclose()
                finally:
                    await _close_turn(
                        collector,
                        status=turn_status,
                        error=turn_error,
                        state=final_state,
                        model=model,
                    )
            latency_ms = int((time.perf_counter() - started) * 1000)
            if turn_status == "completed":
                await tracker.complete(
                    execution_id,
                    result_state=final_state,
                    model=model,
                    latency_ms=latency_ms,
                    flow_trace=flow_trace or None,
                )
            else:
                await tracker.fail(
                    execution_id,
                    error=error or "Stream interrumpido sin respuesta final.",
                    latency_ms=latency_ms,
                )
            await session.commit()
        done_payload = {
            "thread_id": thread_id,
            "execution_id": execution_id,
            "agent": request.agent_type_id or request.agent,
            "version_id": version_id,
            "routing_reason": getattr(graph, "reason", None),
            "actions": final_state.get("app_actions", []) if final_state else [],
            "sources": final_state.get("data_sources", []) if final_state else [],
            "suggestions": [
                s.model_dump()
                for s in _normalize_suggestions(
                    final_state.get("suggestions", []) if final_state else []
                )
            ],
        }
        if protected and final_state is not None:
            answer = _extract_answer(final_state)
            done_payload["answer"] = answer
            payload = json.dumps({"type": "token", "content": answer}, ensure_ascii=False)
            yield f"event: message\ndata: {payload}\n\n"
        if attachment_context is not None:
            done_payload["attachments"] = [
                a.model_dump() for a in _attachment_results(attachment_context)
            ]
        # Señal del control de programa (UC-001 'Controles'): evento propio
        # DESPUÉS del stream de tokens y ANTES de `done`, solo cuando el turno
        # traía contexto Y el modelo llamó la tool de rechazo explícito. El
        # backend la consume y no la reenvía; la app móvil ignora líneas
        # desconocidas. Sin contexto ⇒ nunca se emite.
        if (
            request.control_context is not None
            and final_state is not None
            and "mark_control_declined" in (final_state.get("tools_used") or [])
        ):
            yield "event: control_signal\ndata: declined\n\n"
        yield f"event: done\ndata: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
