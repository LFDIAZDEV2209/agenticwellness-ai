"""Custom LLM detrás de BACK: cabeceras ligadas a sesión, SSE sin prompts internos."""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from contextlib import suppress
from typing import Literal

from anyio import CancelScope
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from langchain_core.messages import AIMessage, AIMessageChunk
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.api.deps import get_db_session, get_graph
from app.api.message_text import extract_message_text
from app.api.routes.chat import _build_config, _check_rate_limit, _extract_answer
from app.api.schemas import ChatRequest
from app.api.streaming import ClosingStreamingResponse as StreamingResponse
from app.core.config import get_settings
from app.history.service import history_service, storage_key
from app.vita.client import VitaAccessError, read_catalog
from app.vita.locking import ThreadBusyError, thread_guard
from app.vita.runtime import resolve_turn
from app.voice.context import verify_context
from app.voice.input import compose_voice_input
from app.voice.language import (
    conversation_language,
    language_chunk,
    language_switch,
    latest_user_index,
    valid_language_followup,
)
from app.voice.store import VoiceBinding, VoiceContextError, voice_store
from app.voice.turns import AdmittedVoiceGraph, prepare_admitted_turn

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/internal/voice/v1", tags=["voice-runtime"])


def _log_failure(phase: str, exc: Exception) -> None:
    # Solo códigos propios y tipos: una excepción del proveedor puede contener PHI.
    code = str(exc) if isinstance(exc, (VoiceContextError, ThreadBusyError)) else None
    logger.warning(
        "vita_voice_failure phase=%s error_type=%s code=%s", phase, type(exc).__name__, code
    )


class ProviderMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None = Field(default=None, max_length=8000)
    tool_calls: list[dict] = Field(default_factory=list, max_length=4)
    tool_call_id: str | None = Field(default=None, max_length=200)


class CompletionRequest(BaseModel):
    # Campos OpenAI del proveedor se ignoran: jamás sobreescriben modelo/tools/KB/prompt.
    model_config = ConfigDict(extra="ignore")
    stream: Literal[True]
    messages: list[ProviderMessage] = Field(min_length=1, max_length=128)
    tools: list[dict] = Field(default_factory=list, max_length=16)

    @model_validator(mode="before")
    @classmethod
    def reject_identity(cls, value):
        if isinstance(value, dict) and any(
            key in value
            for key in (
                "user_id",
                "userId",
                "patient_id",
                "patientId",
                "thread_id",
                "threadId",
                "session_token",
                "sessionToken",
                "conversation_id",
                "conversationId",
            )
        ):
            raise ValueError("identity_must_use_signed_session_headers")
        return value

    @model_validator(mode="after")
    def validate_history(self):
        index = latest_user_index(self.messages)
        if index < 0 or not (self.messages[index].content or "").strip():
            raise ValueError("last_user_message_required")
        if self.messages[-1].role != "user" and not valid_language_followup(self.messages):
            raise ValueError("last_user_message_required")
        if sum(len(m.content or "") for m in self.messages) > 64_000:
            raise ValueError("history_limit")
        return self


def history_identity(messages: list[ProviderMessage]) -> tuple[int, str, str]:
    inputs = [m.content or "" for m in messages if m.role == "user"]

    def digest(values):
        return hashlib.sha256(
            json.dumps(
                values,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    return len(inputs), digest(inputs), digest(inputs[:-1])


def require_voice_service(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if (
        not settings.voice_enabled
        or not settings.vita_runtime_enabled
        or not settings.vita_voice_enabled
        or not settings.vita_voice_transport_verified
        or len(settings.vita_voice_service_key.encode()) < 32
    ):
        raise HTTPException(status_code=503, detail="vita_voice_disabled_or_unverified")
    expected = f"Bearer {settings.vita_voice_service_key}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="voice_service_auth_required")


def _chunk(response_id: str, content: str | None = None, *, done=False) -> str:
    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "vita",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content is not None else {},
                "finish_reason": "stop" if done else None,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat/completions", dependencies=[Depends(require_voice_service)])
async def completion(
    payload: CompletionRequest,
    http_request: Request,
    x_vita_session: str = Header(min_length=1),
    x_vita_conversation: str = Header(min_length=1, max_length=200),
    x_vita_turn: int = Header(ge=0, le=1000000),
    session=Depends(get_db_session),
    base_graph=Depends(get_graph),
) -> StreamingResponse:
    request_started = time.perf_counter()
    settings = get_settings()
    try:
        session_id, expiry = verify_context(x_vita_session, settings.vita_voice_context_secret)
        record = await voice_store.get(session_id)
        if (
            record.runtime_mode != "vita"
            or record.conversation_id != x_vita_conversation
            or int(record.expires_at.timestamp()) != expiry
        ):
            raise VoiceContextError("voice_context_binding")
        await read_catalog(record.user_id)
        _check_rate_limit(http_request, record.user_id)
        ordinal, prefix, previous = history_identity(payload.messages)
        switch = language_switch(payload.messages, payload.tools, record.language)
        binding = VoiceBinding(record.id, record.user_id, record.thread_id, record.conversation_id)
        admission = await voice_store.admit_turn(
            binding,
            ordinal,
            x_vita_turn,
            prefix,
            previous,
            phase="control" if switch else "utterance",
        )
        if admission.action == "wait":
            admission = await voice_store.wait_for_turn(
                admission, timeout=settings.vita_turn_timeout
            )
    except VoiceContextError as exc:
        # Solo códigos de control propios: nunca transcript, firma ni identidad.
        logger.warning(
            "vita_voice_rejected code=%s http=%s flags=%s", str(exc), exc.status, exc.metadata
        )
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    except VitaAccessError as exc:
        raise HTTPException(status_code=403, detail="voice_access_unavailable") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="voice_turn_wait_timeout") from exc

    logger.info("vita_voice_admission action=%s flags=%s", admission.action, admission.metadata)

    def obsolete_response():
        async def finish():
            yield _chunk(admission.response_message_id, done=True)
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            finish(), media_type="text/event-stream", headers={"Cache-Control": "no-store"}
        )

    if admission.action == "obsolete":
        return obsolete_response()

    request = ChatRequest(
        message=compose_voice_input(payload.messages),
        user_id=record.user_id,
        patient_id=record.patient_id,
        thread_id=record.thread_id,
        language=conversation_language(payload.messages, record.language),
    )
    config = _build_config(record.thread_id, request)
    config["configurable"].update(
        {
            "user_message_id": f"voice:{session_id}:{ordinal}:user",
            "assistant_message_id": admission.response_message_id,
            "message_channel": "voice",
            "voice_session_id": record.id,
            "voice_user_text": payload.messages[latest_user_index(payload.messages)].content,
        }
    )
    try:
        await prepare_admitted_turn(
            voice_store,
            admission,
            config,
            base_graph.checkpointer,
            timeout=settings.vita_turn_timeout,
        )
    except asyncio.CancelledError:
        if admission.action == "run":
            with CancelScope(shield=True):
                with suppress(VoiceContextError):
                    await voice_store.finish_admission(admission, "interrupted")
        raise
    except (VoiceContextError, ThreadBusyError) as exc:
        if admission.action == "run":
            with suppress(VoiceContextError):
                await voice_store.finish_admission(admission, "interrupted")
        if str(exc) == "voice_turn_obsolete":
            return obsolete_response()
        raise HTTPException(status_code=getattr(exc, "status", 409), detail=str(exc)) from exc
    except Exception as exc:
        if admission.action == "run":
            with suppress(VoiceContextError):
                await voice_store.finish_admission(admission, "interrupted")
        raise HTTPException(status_code=503, detail="history_state_unavailable") from exc

    if admission.action == "control":

        async def switch_language():
            response_id = admission.response_message_id
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "vita",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [language_chunk(response_id, switch)]},
                        "finish_reason": None,
                    }
                ],
            }
            try:
                async with thread_guard(storage_key(record.user_id, record.thread_id)):
                    await history_service.before_turn_locked(base_graph, config)
                    await voice_store.check_current(admission)
                    yield f"data: {json.dumps(data)}\n\n"
                    data["choices"] = [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
                    yield f"data: {json.dumps(data)}\n\n"
                    yield "data: [DONE]\n\n"
            except (VoiceContextError, ThreadBusyError) as exc:
                if str(exc) == "voice_turn_obsolete":
                    yield _chunk(response_id, done=True)
                    yield "data: [DONE]\n\n"
                else:
                    _log_failure("language", exc)
                    yield 'data: {"error":{"code":"voice_turn_interrupted"}}\n\n'

        return StreamingResponse(
            switch_language(), media_type="text/event-stream", headers={"Cache-Control": "no-store"}
        )

    if admission.action == "replay":

        async def replay():
            try:
                async with thread_guard(config["configurable"]["thread_id"]):
                    await history_service.before_turn_locked(base_graph, config)
                    await voice_store.check_current(admission, completed_ok=True)
                    current = await base_graph.aget_state(config)
                    replay_message = next(
                        (
                            m
                            for m in (current.values or {}).get("messages", [])
                            if m.id == admission.response_message_id
                        ),
                        None,
                    )
                    if replay_message is None:
                        raise VoiceContextError("voice_receipt_without_checkpoint", 409)
                    yield _chunk(
                        admission.response_message_id, extract_message_text(replay_message.content)
                    )
                    yield _chunk(admission.response_message_id, done=True)
                    yield "data: [DONE]\n\n"
            except (VoiceContextError, ThreadBusyError) as exc:
                if str(exc) == "voice_turn_obsolete":
                    yield _chunk(admission.response_message_id, done=True)
                    yield "data: [DONE]\n\n"
                else:
                    _log_failure("replay", exc)
                    yield 'data: {"error":{"code":"voice_turn_interrupted"}}\n\n'

        return StreamingResponse(replay(), media_type="text/event-stream")
    try:
        turn = await resolve_turn(
            user_id=record.user_id,
            text=request.message,
            session=session,
            base_graph=base_graph,
            config=config,
            channel="voice",
        )
        config["recursion_limit"] = turn.recursion_limit
    except asyncio.CancelledError:
        with CancelScope(shield=True):
            with suppress(VoiceContextError):
                await voice_store.finish_admission(admission, "interrupted")
        raise
    except Exception as exc:
        await voice_store.finish_admission(admission, "interrupted")
        if str(exc) == "voice_turn_obsolete":
            return obsolete_response()
        _log_failure("resolve", exc)
        raise HTTPException(status_code=503, detail="voice_runtime_unavailable") from exc

    async def events():
        started = request_started
        first_token_ms = None
        text_chars = 0
        state, completed = None, False
        next_access_check = time.monotonic() + 1
        graph = AdmittedVoiceGraph(
            turn.graph, voice_store, admission, wait_timeout=settings.vita_turn_timeout
        )
        stream = graph.astream(
            {"input": request.message, "agent": turn.agent_id},
            config=config,
            stream_mode=["messages", "values"],
        )
        pending = None
        try:
            async with asyncio.timeout(settings.vita_turn_timeout):
                while True:
                    pending = asyncio.create_task(anext(stream))
                    while not pending.done():
                        await asyncio.wait({pending}, timeout=0.25)
                        # Lectura durable en otra conexión: la revocación de otra
                        # réplica cancela hasta una tool/LLM que aún no emite tokens.
                        await voice_store.check_current(admission)
                        if time.monotonic() >= next_access_check:
                            await read_catalog(record.user_id)
                            next_access_check = time.monotonic() + 1
                    try:
                        mode, chunk = pending.result()
                    except StopAsyncIteration:
                        break
                    if mode == "values":
                        state = chunk
                    elif mode == "messages":
                        message, metadata = chunk
                        if (
                            metadata.get("langgraph_node") == "answer"
                            and isinstance(message, (AIMessage, AIMessageChunk))
                            and not getattr(message, "tool_calls", None)
                            and not getattr(message, "tool_call_chunks", None)
                            # LangGraph vuelve a emitir el AIMessage persistido
                            # con nuestro ID estable al terminar los chunks del
                            # modelo. Ese snapshot no es otro fragmento hablado.
                            and (isinstance(message, AIMessageChunk) or first_token_ms is None)
                        ):
                            text = extract_message_text(message.content)
                            if not text:
                                continue
                            if first_token_ms is None:
                                first_token_ms = int((time.perf_counter() - started) * 1000)
                            await voice_store.check_current(admission)
                            text_chars += len(text)
                            yield _chunk(admission.response_message_id, text)
                if state is None:
                    raise VoiceContextError("voice_stream_incomplete", 502)
                if first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - started) * 1000)
                    await voice_store.check_current(admission)
                    text = _extract_answer(state)
                    text_chars += len(text)
                    yield _chunk(admission.response_message_id, text)
                if not await voice_store.finish_admission(admission, "completed"):
                    raise VoiceContextError("voice_turn_obsolete", 409)
                completed = True
                yield _chunk(admission.response_message_id, done=True)
                yield "data: [DONE]\n\n"
        except (VoiceContextError, VitaAccessError, TimeoutError, ThreadBusyError) as exc:
            if str(exc) == "voice_turn_obsolete":
                yield _chunk(admission.response_message_id, done=True)
                yield "data: [DONE]\n\n"
            else:
                _log_failure("generate", exc)
                yield 'data: {"error":{"code":"voice_turn_interrupted"}}\n\n'
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Nunca serializar errores del proveedor/modelo: pueden contener PHI.
            _log_failure("generate", exc)
            yield 'data: {"error":{"code":"voice_runtime_error"}}\n\n'
        finally:
            # Starlette cancela dentro de un CancelScope. Sin shield, ese scope
            # puede interrumpir la limpieza antes de propagar la cancelación a
            # la tool y cerrar el stream/checkpoint.
            with CancelScope(shield=True):
                if pending:
                    if not pending.done():
                        pending.cancel()
                    # También recuperar la excepción si el wrapper acabó antes
                    # del sondeo de revocación del adaptador.
                    with suppress(asyncio.CancelledError, Exception):
                        await pending
                with suppress(Exception):
                    await stream.aclose()
                if not completed:
                    with suppress(VoiceContextError):
                        await voice_store.finish_admission(admission, "interrupted")
            logger.info(
                "vita_voice_turn agent=%s version=%s reason=%s "
                "first_token_ms=%s duration_ms=%s status=%s tools=%s source_count=%s "
                "language=%s text_chars=%s",
                turn.agent_id,
                turn.version_id,
                turn.reason,
                first_token_ms,
                int((time.perf_counter() - started) * 1000),
                "generated" if completed else "interrupted",
                (state or {}).get("tools_used", []),
                len((state or {}).get("data_sources", [])),
                request.language,
                text_chars,
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )
