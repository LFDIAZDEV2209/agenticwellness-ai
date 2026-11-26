"""HTTP ASGI del adaptador con transporte/modelo/store simulados.

Estos tests no acreditan concurrencia PostgreSQL: el store SQL se verifica en
integración aislada antes de activar el modo. No se accede a la BD compartida.
"""

import asyncio
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import Field

from app.api.deps import get_db_session, get_graph
from app.api.main import create_app
from app.api.routes import vita_voice
from app.core.config import get_settings
from app.graph.graph import build_graph
from app.vita.locking import ThreadBusyError
from app.vita.runtime import ResolvedTurn
from app.voice.context import AUDIENCE, sign_context, verify_context
from app.voice.store import TurnAdmission, VoiceContextError
from tests.fakes import RecordingFakeModel
from tests.test_api import FakeDbSession
from tests.vita_safety_cases import (
    HIDDEN_MARKER,
    PRIOR_ANSWER,
    SAFETY_CASES,
    ChunkedSafetyModel,
    assert_context_delivery,
    assert_safe_delivery,
    sse_payloads,
)

SECRET, SERVICE_KEY = "context-fixture-" * 3, "service-fixture-" * 3
SESSION_ID = "11111111-1111-1111-1111-111111111111"


class SimulatedStore:
    """Solo doble contractual, sin pretensión de representar locks entre réplicas."""

    def __init__(self):
        self.record = SimpleNamespace(
            id=SESSION_ID,
            user_id="owner",
            patient_id="patient",
            thread_id="shared",
            conversation_id="conv_fixture",
            agent_id="agent_vita_fixture",
            version_id=None,
            runtime_mode="vita",
            language="es",
            revoked=False,
            expires_at=datetime.fromtimestamp(int(time.time()) + 300, UTC),
        )
        self.receipts = {}
        self.lock = asyncio.Lock()
        self.wait_started = asyncio.Event()

    async def get(self, sid):
        if (
            sid != self.record.id
            or self.record.revoked
            or self.record.expires_at <= datetime.now(UTC)
        ):
            raise VoiceContextError("voice_session_inactive")
        return self.record

    async def _binding(self, binding):
        record = await self.get(binding.session_id)
        if (binding.user_id, binding.thread_id, binding.conversation_id) != (
            record.user_id,
            record.thread_id,
            record.conversation_id,
        ):
            raise VoiceContextError("voice_context_binding")

    async def admit_turn(
        self, binding, ordinal, provider_turn, prefix, previous, *, phase="utterance"
    ):
        async with self.lock:
            await self._binding(binding)
            latest = self.receipts[max(self.receipts)] if self.receipts else None
            existing = self.receipts.get(ordinal)
            prior = self.receipts.get(ordinal - 1) if existing else latest
            flags = {
                "previous_hash_matches": previous
                == (prior.prefix_hash if prior else hashlib.sha256(b"[]").hexdigest()),
                "counter_reused": latest is not None and latest.provider_turn == provider_turn,
                "existing_completed": existing is not None and existing.status == "completed",
                "latest_ordinal_matches": latest is not None and latest.ordinal == ordinal,
                "prefix_hash_matches": existing is not None and existing.prefix_hash == prefix,
            }
            if existing:
                if not flags["previous_hash_matches"]:
                    raise VoiceContextError("voice_turn_conflict", 409, metadata=flags)
                if existing.prefix_hash != prefix:
                    if latest.ordinal != ordinal or existing.status not in {
                        "running",
                        "control",
                        "completed",
                        "interrupted",
                    }:
                        raise VoiceContextError("voice_turn_conflict", 409, metadata=flags)
                    existing.prefix_hash = prefix
                    existing.response_message_id = (
                        f"voice:{binding.session_id}:{ordinal}:bot:{uuid.uuid4().hex}"
                    )
                    existing.status = "control" if phase == "control" else "running"
                    action = "control" if phase == "control" else "run"
                elif latest.ordinal != ordinal or existing.status == "superseded":
                    action = "obsolete"
                elif existing.status == "interrupted":
                    existing.status = "running"
                    existing.response_message_id = (
                        f"voice:{binding.session_id}:{ordinal}:bot:{uuid.uuid4().hex}"
                    )
                    action = "run"
                elif existing.status == "completed":
                    action = "replay"
                elif existing.status == "running":
                    action = "wait" if phase == "utterance" else "obsolete"
                elif phase == "control":
                    action = "control"
                else:
                    existing.status = "running"
                    action = "run"
                turn = existing
            else:
                if ordinal != (latest.ordinal + 1 if latest else 1):
                    raise VoiceContextError("voice_history_sequence", 409, metadata=flags)
                if not flags["previous_hash_matches"]:
                    raise VoiceContextError("voice_history_changed", 409, metadata=flags)
                if latest and latest.status in {"control", "running"}:
                    latest.status = "superseded"
                turn = SimpleNamespace(
                    ordinal=ordinal,
                    provider_turn=provider_turn,
                    prefix_hash=prefix,
                    status="control" if phase == "control" else "running",
                    response_message_id=f"voice:{binding.session_id}:{ordinal}:bot",
                )
                self.receipts[ordinal] = turn
                action = "control" if phase == "control" else "run"
            return TurnAdmission(
                action, binding, ordinal, prefix, turn.response_message_id, **flags
            )

    async def _current(self, admission, *, completed_ok=False):
        await self._binding(admission.binding)
        latest = self.receipts[max(self.receipts)] if self.receipts else None
        allowed = {"running", "completed"} if completed_ok else {"running"}
        if admission.action == "control":
            allowed = {"control"}
        if (
            latest is None
            or latest.ordinal != admission.ordinal
            or latest.prefix_hash != admission.prefix_hash
            or latest.response_message_id != admission.response_message_id
            or latest.status not in allowed
        ):
            raise VoiceContextError("voice_turn_obsolete", 409, metadata=admission.metadata)

    async def check_current(self, admission, *, completed_ok=False):
        async with self.lock:
            await self._current(admission, completed_ok=completed_ok)

    @asynccontextmanager
    async def write_fence(self, admission):
        assert admission.action == "run"
        async with self.lock:
            await self._current(admission)
            yield

    async def wait_for_turn(self, admission, *, timeout=20, interval=0.001):
        assert admission.action == "wait"
        self.wait_started.set()
        async with asyncio.timeout(timeout):
            while True:
                async with self.lock:
                    await self._binding(admission.binding)
                    latest = self.receipts[max(self.receipts)]
                    if (
                        latest.ordinal != admission.ordinal
                        or latest.prefix_hash != admission.prefix_hash
                        or latest.response_message_id != admission.response_message_id
                        or latest.status in {"superseded", "interrupted"}
                    ):
                        return replace(admission, action="obsolete")
                    if latest.status == "completed":
                        return replace(admission, action="replay")
                await asyncio.sleep(interval)

    async def finish_admission(self, admission, status):
        assert admission.action == "run"
        async with self.lock:
            await self._binding(admission.binding)
            latest = self.receipts[max(self.receipts)]
            if (
                latest.ordinal != admission.ordinal
                or latest.prefix_hash != admission.prefix_hash
                or latest.response_message_id != admission.response_message_id
                or latest.status != "running"
            ):
                return False
            latest.status = status
            return True

    async def revoke(self, sid, user_id):
        if sid != self.record.id or user_id != self.record.user_id:
            raise VoiceContextError("voice_session_not_found", 404)
        self.record.revoked = True


@pytest.fixture
def rig(monkeypatch):
    # Cada escenario HTTP tiene usuarios sintéticos propios; no debe heredar
    # el contador global consumido por escenarios anteriores de toda la suite.
    monkeypatch.setattr("app.api.routes.chat._rate_limiters", {})
    settings = get_settings()
    for name, value in {
        "voice_enabled": True,
        "vita_runtime_enabled": True,
        "vita_voice_enabled": True,
        "vita_voice_transport_verified": True,
        "vita_voice_service_key": SERVICE_KEY,
        "vita_voice_context_secret": SECRET,
    }.items():
        monkeypatch.setattr(settings, name, value)
    store = SimulatedStore()
    monkeypatch.setattr(vita_voice, "voice_store", store)
    monkeypatch.setattr("app.api.routes.voice.voice_store", store)
    monkeypatch.setattr(vita_voice, "read_catalog", AsyncMock(return_value=[]))
    history = SimpleNamespace(
        before_turn_locked=AsyncMock(),
        activity_locked=AsyncMock(),
        ensure_locked=AsyncMock(return_value=SimpleNamespace(history_revision=0)),
    )
    monkeypatch.setattr(vita_voice, "history_service", history)
    monkeypatch.setattr("app.history.service.history_service", history)

    thread_lock = asyncio.Lock()

    @asynccontextmanager
    async def guarded(_):
        if thread_lock.locked():
            raise ThreadBusyError("vita_thread_busy")
        async with thread_lock:
            yield

    monkeypatch.setattr("app.voice.turns.thread_guard", guarded)
    model = RecordingFakeModel(
        responses=[
            AIMessage(content="Respuesta uno"),
            AIMessage(content="Respuesta dos"),
        ]
    )
    graph = build_graph(model=model, checkpointer=MemorySaver())

    async def resolve(**_):
        return ResolvedTurn(
            graph, "authorized-agent", "active-version", "fake", "fake", 25, "explicit"
        )

    resolver = AsyncMock(side_effect=resolve)
    monkeypatch.setattr(vita_voice, "resolve_turn", resolver)
    app = create_app()
    app.dependency_overrides[get_graph] = lambda: graph
    app.dependency_overrides[get_db_session] = FakeDbSession
    token = sign_context(SESSION_ID, int(store.record.expires_at.timestamp()), SECRET)
    headers = {
        "Authorization": f"Bearer {SERVICE_KEY}",
        "X-Vita-Session": token,
        "X-Vita-Conversation": "conv_fixture",
        "X-Vita-Turn": "1",
    }
    return SimpleNamespace(
        app=app,
        store=store,
        model=model,
        graph=graph,
        headers=headers,
        resolver=resolver,
        history=history,
        monkeypatch=monkeypatch,
    )


def body(*inputs, **extra):
    return {"stream": True, "messages": [{"role": "user", "content": t} for t in inputs], **extra}


async def post(rig, payload=None, headers=None):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=rig.app), base_url="http://test"
    ) as client:
        return await client.post(
            "/internal/voice/v1/chat/completions",
            json=payload or body("Hola"),
            headers=rig.headers if headers is None else headers,
        )


LANGUAGE_TOOLS = [{"type": "function", "function": {"name": "language_detection"}}]


@pytest.mark.parametrize(
    ("language", "utterance", "answer"),
    [
        ("es", "¿Cómo se llama la aplicación?", "La aplicación se llama Agentic Wellness IA."),
        ("es", "Vita, ¿cómo se llama esta app?", "La aplicación se llama Agentic Wellness IA."),
        (
            "es",
            "¿Esta app sigue siendo COPP-ADRESD?",
            "La aplicación se llama Agentic Wellness IA.",
        ),
        (
            "es",
            "¿Estoy usando la app de CoppAdressd?",
            "La aplicación se llama Agentic Wellness IA.",
        ),
        ("es", "¿Estoy usando ANTARES?", "La aplicación se llama Agentic Wellness IA."),
        ("es", "CoppAdressd", "La aplicación se llama Agentic Wellness IA."),
        ("en", "What's this app called?", "The app is called Agentic Wellness IA."),
        ("en", "ANTARES?", "The app is called Agentic Wellness IA."),
        ("en", "Is this app still called COPP ADRESD?", "The app is called Agentic Wellness IA."),
        ("en", "Is this ANTARES?", "The app is called Agentic Wellness IA."),
    ],
)
async def test_voice_returns_official_application_name(rig, language, utterance, answer):
    rig.store.record.language = language
    request = body(utterance)
    request["messages"].insert(0, {"role": "assistant", "content": "Copp Adresd."})
    response = await post(rig, request)

    assert response.status_code == 200
    deltas = [
        json.loads(line.removeprefix("data: "))["choices"][0]["delta"]
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    assert "".join(delta.get("content", "") for delta in deltas) == answer
    assert not rig.model.prompts


async def test_voice_corrects_legacy_name_before_answering_the_question(rig):
    rig.model.responses = [AIMessage(content="Puedes abrir tu plan desde Inicio.")]
    response = await post(rig, body("En Copp Adresd, ¿dónde veo mi plan?"))

    assert response.status_code == 200
    deltas = [
        json.loads(line.removeprefix("data: "))["choices"][0]["delta"]
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    assert "".join(delta.get("content", "") for delta in deltas) == (
        "La aplicación se llama Agentic Wellness IA. Puedes abrir tu plan desde Inicio."
    )
    assert rig.model.prompts


async def test_initial_sdk_language_does_not_trigger_redundant_preset_reset(rig):
    rig.store.record.language = "en"
    response = await post(rig, body("What is my assigned lunch?", tools=LANGUAGE_TOOLS))
    assert response.status_code == 200 and "language_detection" not in response.text
    assert rig.store.receipts[1].status == "completed"
    rig.resolver.assert_awaited_once()


async def test_language_switch_cannot_emit_after_concurrent_revocation(rig):
    async def revoked_during_catalog(_):
        rig.store.record.revoked = True
        return []

    rig.monkeypatch.setattr(vita_voice, "read_catalog", revoked_during_catalog)
    response = await post(rig, body("What is my assigned lunch?", tools=LANGUAGE_TOOLS))
    assert response.status_code == 403
    assert "language_detection" not in response.text
    assert not rig.store.receipts
    rig.resolver.assert_not_awaited()


async def test_history_cleanup_rejects_voice_before_generation(rig):
    rig.history.ensure_locked.side_effect = VoiceContextError("history_revision_changed", 409)
    result = await post(rig)
    assert result.status_code == 409
    rig.resolver.assert_not_awaited()
    assert rig.store.receipts[1].status == "interrupted"


async def test_replay_revalidates_history_before_sending_old_answer(rig):
    assert (await post(rig)).status_code == 200
    rig.history.before_turn_locked.side_effect = VoiceContextError("history_revision_changed", 409)
    replayed = await post(rig)
    assert replayed.status_code == 200
    assert "Respuesta uno" not in replayed.text
    assert "voice_turn_interrupted" in replayed.text
    assert len(rig.model.prompts) == 1


@pytest.mark.parametrize("quoted_asr", [False, True])
async def test_native_language_tool_roundtrip_is_not_another_patient_turn(rig, quoted_asr):
    text = "What does my nutrition plan include?"
    request = body(f'"{text}".' if quoted_asr else text, tools=LANGUAGE_TOOLS)
    switched = await post(rig, request)
    assert switched.status_code == 200
    chunks = [
        json.loads(line[6:]) for line in switched.text.splitlines() if line.startswith("data: {")
    ]
    call = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert call["function"]["name"] == "language_detection"
    assert json.loads(call["function"]["arguments"])["language"] == "en"
    assert chunks[1]["choices"][0]["finish_reason"] == "tool_calls"
    assert len(rig.store.receipts) == 1 and rig.store.receipts[1].status == "control"
    assert rig.model.prompts == []
    rig.resolver.assert_not_awaited()
    repeated_control = await post(rig, request)
    repeated_call = next(
        json.loads(line[6:])["choices"][0]["delta"]["tool_calls"][0]
        for line in repeated_control.text.splitlines()
        if '"tool_calls"' in line and '"delta"' in line
    )
    assert repeated_call["id"] == call["id"] and len(rig.store.receipts) == 1
    request["messages"].extend(
        [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "content": "Language changed", "tool_call_id": call["id"]},
        ]
    )
    answered = await post(rig, request)
    assert answered.status_code == 200
    assert "Respuesta uno" in answered.text
    assert len(rig.store.receipts) == 1
    assert rig.resolver.call_args.kwargs["text"] == request["messages"][0]["content"]
    assert len(rig.model.prompts) == 1
    assert (await post(rig, request)).status_code == 200
    assert rig.store.receipts[1].status == "completed" and len(rig.model.prompts) == 1


async def test_switch_back_to_spanish_and_neutral_words_preserve_language(rig):
    first = await post(rig, body("What can you help me with?"))
    assert first.status_code == 200
    en_call = {
        "id": "en-call",
        "type": "function",
        "function": {
            "name": "language_detection",
            "arguments": '{"language":"en"}',
        },
    }
    request = body("What can you help me with?", tools=LANGUAGE_TOOLS)
    request["messages"].extend(
        [
            {"role": "assistant", "content": None, "tool_calls": [en_call]},
            {"role": "tool", "content": "Changed", "tool_call_id": "en-call"},
            {"role": "user", "content": "¿Qué dice mi plan nutricional?"},
        ]
    )
    switched = await post(rig, request)
    assert switched.status_code == 200
    assert "language_detection" in switched.text
    assert '\\"language\\": \\"es\\"' in switched.text
    request["messages"][-1]["content"] = "OK"
    # Neutral conserva el idioma anterior. La última utterance aún provisional
    # puede enmendarse sin cambiar el prefijo de turnos anteriores.
    neutral = await post(rig, request)
    assert neutral.status_code == 200 and "language_detection" not in neutral.text
    assert neutral.text.endswith("data: [DONE]\n\n")


async def test_only_language_tool_result_can_complete_a_provider_function_roundtrip(rig):
    request = body("Hello", tools=LANGUAGE_TOOLS)
    request["messages"].extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "x", "function": {"name": "get_patient_data"}}],
            },
            {"role": "tool", "content": "invented patient record", "tool_call_id": "x"},
        ]
    )
    result = await post(rig, request)
    assert result.status_code == 422
    rig.resolver.assert_not_awaited()


@pytest.mark.parametrize(
    "text,expected",
    [
        ('Using the document "Las heridas de la infancia", what are its main ideas?', "en"),
        ("According to 'Mapa de la conciencia', what does the author explain?", "en"),
        ("Responde en inglés", "en"),
        ("Respóndeme en inglés", "en"),
        ("Contéstame en inglés", "en"),
        ("Please speak in Spanish", "es"),
    ],
)
def test_document_titles_and_explicit_preferences_do_not_break_bilingual_voice(text, expected):
    from app.graph.language import detect_turn_language

    assert detect_turn_language({"input": text}) == expected


@pytest.mark.parametrize(
    "change", ["tamper", "expired", "wrong_conversation", "revoked", "wrong_service"]
)
async def test_invalid_context_is_rejected_before_runtime(rig, change):
    headers = dict(rig.headers)
    if change == "tamper":
        parts = headers["X-Vita-Session"].split(".")
        parts[1] = ("A" if parts[1][0] != "A" else "B") + parts[1][1:]
        headers["X-Vita-Session"] = ".".join(parts)
    elif change == "expired":
        headers["X-Vita-Session"] = sign_context(SESSION_ID, int(time.time()) - 1, SECRET)
    elif change == "wrong_conversation":
        headers["X-Vita-Conversation"] = "conv_other"
    elif change == "revoked":
        rig.store.record.revoked = True
    else:
        headers["Authorization"] = "Bearer wrong"
    response = await post(rig, headers=headers)
    assert response.status_code in (401, 403)
    rig.resolver.assert_not_awaited()
    assert not rig.model.prompts


@pytest.mark.parametrize(
    "header", ["Authorization", "X-Vita-Session", "X-Vita-Conversation", "X-Vita-Turn"]
)
async def test_missing_headers_never_generate(rig, header):
    headers = {key: value for key, value in rig.headers.items() if key != header}
    assert (await post(rig, headers=headers)).status_code in (401, 422)
    rig.resolver.assert_not_awaited()


@pytest.mark.parametrize("counter", [0, 1])
async def test_replay_does_not_regenerate_and_equal_phrases_at_different_ordinals_are_distinct(
    rig, counter
):
    headers = {**rig.headers, "X-Vita-Turn": str(counter)}
    first = await post(rig, body("La misma frase"), headers=headers)
    replay = await post(rig, body("La misma frase"), headers=headers)
    assert first.status_code == replay.status_code == 200
    assert "Respuesta uno" in first.text and "Respuesta uno" in replay.text
    assert len(rig.model.prompts) == 1
    second = await post(rig, body("La misma frase", "La misma frase"), headers=headers)
    assert second.status_code == 200 and "Respuesta dos" in second.text
    assert len(rig.model.prompts) == 2
    snapshot = await rig.graph.aget_state({"configurable": {"thread_id": "owner::shared"}})
    users = [m for m in snapshot.values["messages"] if m.type == "human"]
    assert len(users) == 2 and users[0].content == users[1].content
    assert users[0].id != users[1].id
    assert first.text.endswith("data: [DONE]\n\n")


async def test_changed_earlier_utterance_is_not_a_new_question(rig):
    await post(rig)
    assert (await post(rig, body("Hola", "Otra pregunta"))).status_code == 200
    assert (await post(rig, body("Texto cambiado"))).status_code == 409
    assert len(rig.model.prompts) == 2


async def test_voice_persiste_una_traza_y_replay_no_agrega_otra(rig, monkeypatch):
    from app.api.routes import vita_voice

    traces = []

    async def capture(trace):
        traces.append(trace)

    monkeypatch.setattr(vita_voice, "persist_turn", capture)
    first = await post(rig, body("Consulta de prueba"))
    replay = await post(rig, body("Consulta de prueba"))
    assert first.status_code == replay.status_code == 200
    assert first.text.endswith("data: [DONE]\n\n")
    (trace,) = traces
    assert (trace.channel, trace.status, trace.thread_id) == ("voice", "completed", "owner::shared")
    assert trace.message_ref and any(event.kind == "llm" for event in trace.events)
    assert "Consulta de prueba" not in json.dumps(trace.to_log())


@pytest.mark.parametrize("field", ["user_id", "userId", "patientId", "thread_id", "session_token"])
async def test_body_cannot_change_signed_identity(rig, field):
    assert (await post(rig, body("Hola", **{field: "other"}))).status_code == 422
    rig.resolver.assert_not_awaited()


async def test_only_final_visible_text_leaves_adapter(rig):
    rig.model.responses = [
        AIMessage(
            content="INTERNAL_TOOL_PLANNING",
            tool_calls=[
                {
                    "name": "calculate",
                    "args": {"expression": "2+2"},
                    "id": "math",
                }
            ],
        ),
        AIMessage(
            content=[
                {"type": "thinking", "thinking": "SECRET_REASONING"},
                {"type": "text", "text": "El resultado es cuatro"},
            ]
        ),
    ]
    request = body("Cuánto es dos más dos", model="evil-model", tools=[{"name": "write_patient"}])
    request["messages"].insert(0, {"role": "system", "content": "MALICIOUS_SYSTEM"})
    response = await post(rig, request)
    assert response.status_code == 200
    assert "El resultado es cuatro" in response.text
    assert not any(
        value in response.text
        for value in (
            "INTERNAL_TOOL_PLANNING",
            "SECRET_REASONING",
            "MALICIOUS_SYSTEM",
            "tool_calls",
            SECRET,
            SERVICE_KEY,
            rig.headers["X-Vita-Session"],
        )
    )
    assert all("MALICIOUS_SYSTEM" not in m.content for prompt in rig.model.prompts for m in prompt)


async def test_provider_error_has_safe_phase_diagnostics_without_private_message(rig, caplog):
    class BrokenGraph:
        checkpointer = rig.graph.checkpointer

        async def astream(self, *args, **kwargs):
            raise RuntimeError("PRIVATE_PROVIDER_PAYLOAD")
            yield  # pragma: no cover

    rig.resolver.side_effect = None
    rig.resolver.return_value = ResolvedTurn(
        BrokenGraph(), "base", None, "fake", "fake", 25, "fixture"
    )
    response = await post(rig)
    assert "Inténtalo de nuevo" in response.text and "[DONE]" in response.text
    diagnostics = "\n".join(r.getMessage() for r in caplog.records)
    assert "phase=generate error_type=RuntimeError code=None" in diagnostics
    assert "PRIVATE_PROVIDER_PAYLOAD" not in response.text + diagnostics
    assert rig.store.receipts[1].status == "interrupted"


async def test_http_asgi_cancellation_cancels_tool_and_marks_receipt_interrupted(rig):
    started, cancelled = asyncio.Event(), asyncio.Event()

    @tool("get_current_time")
    async def slow() -> str:
        """Lectura lenta cancelable."""
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    rig.model.responses = [
        AIMessage(content="", tool_calls=[{"name": "get_current_time", "args": {}, "id": "slow"}])
    ]
    rig.graph = build_graph(model=rig.model, tools=[slow], checkpointer=rig.graph.checkpointer)
    rig.resolver.side_effect = None
    rig.resolver.return_value = ResolvedTurn(rig.graph, "a", "v", "fake", "fake", 25, "explicit")
    task = asyncio.create_task(post(rig, body("Consulta lenta")))
    await asyncio.wait_for(started.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(cancelled.wait(), 2)
    assert rig.store.receipts[1].status == "interrupted"

    # La transcripción final puede llegar DESPUÉS de cancelar el runner anterior.
    # Misma utterance/prefijo previo, texto revisado: nueva época y respuesta útil.
    previous_id = rig.store.receipts[1].response_message_id
    rig.model.responses = [AIMessage(content="Respuesta recuperada")]
    revised = await post(rig, body("Consulta lenta con transcripción final"))
    assert revised.status_code == 200 and "Respuesta recuperada" in revised.text
    assert "voice_turn_conflict" not in revised.text
    assert rig.store.receipts[1].response_message_id != previous_id
    assert rig.store.receipts[1].status == "completed"
    replay = await post(rig, body("Consulta lenta con transcripción final"))
    assert replay.status_code == 200 and "Respuesta recuperada" in replay.text
    assert len(rig.model.prompts) == 2


async def test_revoke_requires_owner_and_invalidates_token(rig):
    headers = {"X-Internal-Key": get_settings().internal_api_key}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=rig.app), base_url="http://test"
    ) as client:
        assert (
            await client.delete(
                f"/internal/voice/session/{SESSION_ID}?user_id=other", headers=headers
            )
        ).status_code == 404
        assert (
            await client.delete(
                f"/internal/voice/session/{SESSION_ID}?user_id=owner", headers=headers
            )
        ).status_code == 204
        assert (
            await client.delete(
                f"/internal/voice/session/{SESSION_ID}?user_id=owner", headers=headers
            )
        ).status_code == 204
    assert (await post(rig)).status_code == 403


def test_token_has_no_patient_identity_and_binds_audience():
    expiry = int(time.time()) + 60
    token = sign_context(SESSION_ID, expiry, SECRET)
    assert verify_context(token, SECRET) == (SESSION_ID, expiry)
    import base64

    claims = json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "=="))
    assert claims == {"sid": SESSION_ID, "exp": expiry, "aud": AUDIENCE}


@pytest.mark.parametrize(
    "flag",
    [
        "voice_enabled",
        "vita_runtime_enabled",
        "vita_voice_enabled",
        "vita_voice_transport_verified",
    ],
)
async def test_each_disabled_flag_closes_adapter(rig, flag):
    rig.monkeypatch.setattr(get_settings(), flag, False)
    assert (await post(rig)).status_code == 503
    rig.resolver.assert_not_awaited()


async def test_vita_session_emits_only_reduced_context_and_bound_thread(rig):
    from app.voice import vita_session

    settings = get_settings()
    rig.monkeypatch.setattr(settings, "vita_elevenlabs_agent_id", "agent_vita_fixture")
    rig.monkeypatch.setattr(settings, "vita_elevenlabs_version_id", "agtvrsn_vita_fixture")
    rig.monkeypatch.setattr(vita_session, "read_catalog", AsyncMock(return_value=[]))
    upstream = AsyncMock(
        return_value={"signed_url": "wss://fixture.test/session", "conversation_id": "conv_bound"}
    )
    saved = AsyncMock()
    rig.monkeypatch.setattr(vita_session, "create_signed_url", upstream)
    rig.monkeypatch.setattr(vita_session, "voice_store", SimpleNamespace(create=saved))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=rig.app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/internal/voice/session",
            json={
                "user_id": "owner",
                "patient_id": "patient",
                "thread_id": "existing",
                "language": "en",
                "supports_vita_voice": True,
            },
            headers={"X-Internal-Key": settings.internal_api_key},
        )
    assert response.status_code == 200
    result = response.json()
    assert result["runtime_mode"] == "vita" and result["thread_id"] == "existing"
    assert result["conversation_id"] == "conv_bound"
    sid, expiry = verify_context(result["session_token"], SECRET)
    record = saved.call_args.args[0]
    assert sid == record.id and record.user_id == "owner"
    assert record.thread_id == "existing" and int(record.expires_at.timestamp()) == expiry
    assert "owner" not in result["session_token"] and "patient" not in result["session_token"]
    assert len(upstream.call_args.args) == 3  # No JWT/identidad enviados al proveedor.


async def test_revocation_during_http_asgi_generation_cancels_tool(rig, caplog):
    started, cancelled = asyncio.Event(), asyncio.Event()

    @tool("get_current_time")
    async def slow() -> str:
        """Lectura aún en vuelo al revocar desde otra réplica."""
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    rig.model.responses = [
        AIMessage(content="", tool_calls=[{"name": "get_current_time", "args": {}, "id": "slow"}])
    ]
    graph = build_graph(model=rig.model, tools=[slow], checkpointer=MemorySaver())
    rig.resolver.side_effect = None
    rig.resolver.return_value = ResolvedTurn(graph, "a", "v", "fake", "fake", 25, "explicit")
    task = asyncio.create_task(post(rig, body("Consulta lenta")))
    await asyncio.wait_for(started.wait(), 3)
    rig.store.record.revoked = True
    response = await asyncio.wait_for(task, 3)
    assert "voice_turn_interrupted" in response.text
    assert cancelled.is_set()
    # La finalización también exige sesión activa: revocar no permite completar
    # ni reclamar el recibo pendiente aunque la limpieza ya no pueda modificarlo.
    assert rig.store.receipts[1].status != "completed"
    assert (await post(rig, body("Consulta lenta"))).status_code == 403
    await asyncio.sleep(0)
    assert not any("Task exception was never retrieved" in r.getMessage() for r in caplog.records)


class StreamingFinalModel(RecordingFakeModel):
    release: Any = Field(default_factory=asyncio.Event)
    waiting_for_release: Any = Field(default_factory=asyncio.Event)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(list(messages))
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=[
                    {
                        "type": "thinking",
                        "thinking": "SECRET_REASONING",
                    }
                ]
            )
        )
        yield ChatGenerationChunk(message=AIMessageChunk(content="**Tu resultado fue 9.**"))
        yield ChatGenerationChunk(message=AIMessageChunk(content=" "))
        self.waiting_for_release.set()
        await self.release.wait()
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="¿Te lo explico? Tercera frase. **DETALLE_RECORTADO.**")
        )


@pytest.mark.parametrize("outcome", ["complete", "revoke"])
async def test_inflight_http_duplicate_waits_and_replays_without_another_llm(rig, outcome):
    entered, release = asyncio.Event(), asyncio.Event()

    @tool("get_current_time")
    async def hold() -> str:
        """Lectura controlada para observar un duplicado durante ejecución."""
        entered.set()
        await release.wait()
        return "fixture read"

    graph = build_graph(
        model=rig.model,
        tools=[hold],
        prepare_tools=["get_current_time"],
        checkpointer=rig.graph.checkpointer,
    )
    rig.resolver.side_effect = None
    rig.resolver.return_value = ResolvedTurn(graph, "base", None, "fake", "fake", 25, "fixture")
    first = asyncio.create_task(post(rig))
    duplicate = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        duplicate = asyncio.create_task(post(rig))
        await asyncio.wait_for(rig.store.wait_started.wait(), 3)
        assert not duplicate.done() and not rig.model.prompts
        if outcome == "complete":
            release.set()
            responses = await asyncio.wait_for(asyncio.gather(first, duplicate), 3)
            assert all(r.status_code == 200 and "Respuesta uno" in r.text for r in responses)
            assert len(rig.model.prompts) == 1 and rig.resolver.await_count == 1
            assert rig.store.receipts[1].status == "completed"
        else:
            rig.store.record.revoked = True
            responses = await asyncio.wait_for(asyncio.gather(first, duplicate), 3)
            assert responses[0].status_code == 200 and "voice_turn_interrupted" in responses[0].text
            assert responses[1].status_code == 403
            assert rig.model.prompts == [] and rig.resolver.await_count == 1
    finally:
        release.set()
        for task in (first, duplicate):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*[t for t in (first, duplicate) if t], return_exceptions=True)


@pytest.mark.parametrize("stage", ["prepare", "resolve"])
async def test_cancellation_before_generation_marks_only_admitted_run_interrupted(rig, stage):
    entered = asyncio.Event()
    original_resolve = rig.resolver.side_effect

    async def hold(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    if stage == "prepare":
        rig.history.ensure_locked.side_effect = hold
    else:
        rig.resolver.side_effect = hold
    pending = asyncio.create_task(post(rig))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert rig.store.receipts[1].status == "interrupted"
        assert rig.model.prompts == []
        # Un reintento de la misma locución tras cancelar se admite como run
        # nuevo (epoch nuevo); no se devuelve un SSE vacío ni se pierde la
        # pregunta del paciente.
        if stage == "prepare":
            rig.history.ensure_locked.side_effect = None
        else:
            rig.resolver.side_effect = original_resolve
        rig.model.responses = [AIMessage(content="Respuesta recuperada")]
        repeated = await post(rig)
        assert repeated.status_code == 200 and "Respuesta recuperada" in repeated.text
        assert len(rig.model.prompts) == 1
        assert rig.store.receipts[1].status == "completed"
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize("asr_amendment", [False, True])
async def test_split_audio_supersedes_control_roundtrip_runner_with_same_counter(
    rig, asr_amendment
):
    entered, cancelled = asyncio.Event(), asyncio.Event()
    reads = 0

    @tool("get_current_time")
    async def read_once() -> str:
        """La primera lectura se cancela; la sucesora continúa."""
        nonlocal reads
        reads += 1
        if reads == 1:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return "fixture read"

    graph = build_graph(
        model=rig.model,
        tools=[read_once],
        prepare_tools=["get_current_time"],
        checkpointer=rig.graph.checkpointer,
    )
    rig.resolver.side_effect = None
    rig.resolver.return_value = ResolvedTurn(graph, "base", None, "fake", "fake", 25, "fixture")
    first_text = "What does my nutrition plan say for lunch today?"
    second_text = "Please answer briefly."
    request = body(first_text, tools=LANGUAGE_TOOLS)
    switched = await post(rig, request)
    call = next(
        json.loads(line[6:])["choices"][0]["delta"]["tool_calls"][0]
        for line in switched.text.splitlines()
        if '"tool_calls"' in line and '"delta"' in line
    )
    request["messages"].extend(
        [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "content": "Changed", "tool_call_id": call["id"]},
        ]
    )
    first = asyncio.create_task(post(rig, request))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        followup = (
            body(first_text + " " + second_text, tools=LANGUAGE_TOOLS)
            if asr_amendment
            else {
                **request,
                "messages": [*request["messages"], {"role": "user", "content": second_text}],
            }
        )
        if asr_amendment:
            # El SDK ya cambió de idioma: conserva el retorno nativo actual.
            followup["messages"].extend(request["messages"][1:])
        result = await asyncio.wait_for(post(rig, followup), 4)
        old = await asyncio.wait_for(first, 3)
        assert result.status_code == old.status_code == 200
        assert "Respuesta uno" in result.text and "Respuesta" not in old.text
        assert "error" not in old.text and old.text.endswith("data: [DONE]\n\n")
        assert cancelled.is_set() and len(rig.model.prompts) == 1
        if asr_amendment:
            assert len(rig.store.receipts) == 1 and rig.store.receipts[1].status == "completed"
        else:
            assert rig.store.receipts[1].status == "superseded"
            assert rig.store.receipts[2].status == "completed"
            assert [r.provider_turn for r in rig.store.receipts.values()] == [1, 1]
        routed = rig.resolver.call_args.kwargs["text"]
        assert first_text in routed and second_text in routed
        state = await rig.graph.aget_state({"configurable": {"thread_id": "owner::shared"}})
        users = [m.content for m in state.values["messages"] if m.type == "human"]
        assert users == (
            [first_text + " " + second_text] if asr_amendment else [first_text, second_text]
        )
        assert (await post(rig, followup)).status_code == 200 and len(rig.model.prompts) == 1
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)


async def test_old_completed_ordinal_is_obsolete_after_new_utterance(rig):
    assert (await post(rig, body("Uno"))).status_code == 200
    assert (await post(rig, body("Uno", "Dos"))).status_code == 200
    old = await post(rig, body("Uno"))
    assert old.status_code == 200 and old.text.endswith("data: [DONE]\n\n")
    assert "Respuesta" not in old.text and "error" not in old.text
    assert len(rig.model.prompts) == 2


async def test_late_latest_revision_after_generation_completes_keeps_voice_chain(rig):
    first_text = "What is my lunch?"
    final_text = "What is my lunch today? Please include the portions."
    first = await post(rig, body(first_text))
    assert first.status_code == 200 and rig.store.receipts[1].status == "completed"
    old_response_id = rig.store.receipts[1].response_message_id
    revised = await post(rig, body(final_text))
    assert revised.status_code == 200 and "voice_turn_conflict" not in revised.text
    assert len(rig.model.prompts) == 2
    assert rig.store.receipts[1].response_message_id != old_response_id
    replay = await post(rig, body(final_text))
    assert replay.status_code == 200 and len(rig.model.prompts) == 2
    snapshot = await rig.graph.aget_state({"configurable": {"thread_id": "owner::shared"}})
    assert [m.content for m in snapshot.values["messages"] if m.type == "human"] == [final_text]
    assert len([m for m in snapshot.values["messages"] if m.type == "ai"]) == 1
    next_question = await post(rig, body(final_text, "What is my assigned podcast?"))
    assert next_question.status_code == 200 and len(rig.model.prompts) == 3
    changed_previous = await post(rig, body(first_text, "What is my assigned podcast?"))
    assert changed_previous.status_code == 409 and len(rig.model.prompts) == 3


async def test_changed_previous_prefix_is_rejected_with_safe_diagnostics(rig, caplog):
    assert (await post(rig, body("Original fixture"))).status_code == 200
    response = await post(rig, body("Modified fixture", "New fixture"))
    assert response.status_code == 409 and response.json()["detail"] == "voice_history_changed"
    assert len(rig.model.prompts) == 1
    rejection = next(
        r.getMessage() for r in caplog.records if "voice_history_changed" in r.getMessage()
    )
    assert "previous_hash_matches" in rejection and "counter_reused" in rejection
    assert "fixture" not in rejection and SECRET not in rejection


async def test_http_asgi_withholds_audio_until_generation_completes(rig):
    @tool("get_current_time")
    def internal_read() -> str:
        """Lectura interna preparada en código, nunca narrada al proveedor."""
        return "INTERNAL_TOOL_PLANNING"

    model = StreamingFinalModel(responses=[AIMessage(content="unused")])
    graph = build_graph(
        model=model,
        tools=[internal_read],
        prepare_tools=["get_current_time"],
        checkpointer=rig.graph.checkpointer,
    )
    rig.resolver.side_effect = None
    rig.resolver.return_value = ResolvedTurn(graph, "a", "v", "fake", "fake", 25, "explicit")
    first_content = asyncio.Event()
    bodies = []
    request_sent = False
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "method": "POST",
        "scheme": "http",
        "path": "/internal/voice/v1/chat/completions",
        "raw_path": b"/internal/voice/v1/chat/completions",
        "query_string": b"",
        "http_version": "1.1",
        "server": ("test", 80),
        "client": ("test", 123),
        "headers": [
            (k.lower().encode(), v.encode())
            for k, v in {
                **rig.headers,
                "Content-Type": "application/json",
            }.items()
        ],
    }

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {
                "type": "http.request",
                "body": json.dumps(body("Consulta sintética")).encode(),
                "more_body": False,
            }
        await asyncio.Event().wait()

    async def send(message):
        if message["type"] == "http.response.body":
            bodies.append(message.get("body", b""))
            if b"Tu resultado fue 9." in bodies[-1]:
                first_content.set()

    task = asyncio.create_task(rig.app(scope, receive, send))
    try:
        await asyncio.wait_for(model.waiting_for_release.wait(), 3)
        assert not first_content.is_set()
        assert b"Tu resultado" not in b"".join(bodies)
        assert rig.store.receipts[1].status == "running"
        assert not task.done()
    finally:
        model.release.set()
        await asyncio.wait_for(task, 3)
    assert first_content.is_set()
    response = b"".join(bodies).decode()
    spoken = "".join(
        json.loads(line[6:])["choices"][0]["delta"].get("content", "")
        for line in response.splitlines()
        if line.startswith("data: {")
    )
    snapshot = await graph.aget_state({"configurable": {"thread_id": "owner::shared"}})
    assert spoken == "Tu resultado fue 9. ¿Te lo explico? Tercera frase."
    assert snapshot.values["messages"][-1].content == spoken
    assert "DETALLE_RECORTADO" not in response
    assert "INTERNAL_TOOL_PLANNING" not in response and "SECRET_REASONING" not in response
    assert len(model.prompts) == 1  # Sin generación de planificación ni reescritor.
    assert rig.store.receipts[1].status == "completed"


def set_safety_voice_model(rig, model):
    graph = build_graph(
        model=model,
        checkpointer=rig.graph.checkpointer,
        tools=[],
        prepare_tools=[],
    )
    rig.resolver.side_effect = None
    rig.resolver.return_value = ResolvedTurn(
        graph, "base", None, "fixture", "fixture", 25, "fixture"
    )
    return graph


def voice_fragments(response):
    assert response.status_code == 200
    assert response.text.endswith("data: [DONE]\n\n")
    assert HIDDEN_MARKER not in response.text
    frames = sse_payloads(response.text)
    assert not any("error" in frame for frame in frames)
    return [
        frame["choices"][0]["delta"]["content"]
        for frame in frames
        if frame["choices"][0]["delta"].get("content")
    ]


@pytest.mark.parametrize("case", SAFETY_CASES)
async def test_vita_safety_voice_matrix(rig, case):
    rig.store.record.language = case.language
    model = ChunkedSafetyModel(responses=[AIMessage(content=PRIOR_ANSWER[case.language])])
    graph = set_safety_voice_model(rig, model)
    if case.previous:
        first = await post(rig, body(case.previous))
        assert_context_delivery(model, case, voice_fragments(first))
        payload = body(case.previous, case.prompt)
        headers = {**rig.headers, "X-Vita-Turn": "2"}
    else:
        payload = body(case.prompt)
        headers = rig.headers

    prior_calls = len(model.prompts)
    model.responses = [AIMessage(content=case.answer)]
    response = await post(rig, payload, headers=headers)
    delivered = assert_safe_delivery(case, voice_fragments(response), model.prompts[prior_calls:])
    snapshot = await graph.aget_state({"configurable": {"thread_id": "owner::shared"}})
    assert snapshot.values["messages"][-1].content == delivered
