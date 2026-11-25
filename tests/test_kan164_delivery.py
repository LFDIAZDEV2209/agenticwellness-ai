"""Aceptación KAN-164 a través de HTTP y el checkpoint reales, sin claves externas."""

from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import AIMessage

from app.api.routes import chat, vita_voice
from tests.e2e.kan164_fixture import create
from tests.test_vita_voice import body, post, rig  # noqa: F401

FALLBACK = "Consulta a un profesional de la salud"
BLOCKED = "Haz una pausa consciente. La dosis indicada es 50 mg."


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("Diagnóstico", "clinical_topic"),
        ("Dosis", "clinical_topic"),
        ("¿Qué medicamento tomo?", "clinical_topic"),
        ("Dolor en el pecho", "clinical_topic"),
        ("Quiero saber más sobre bienestar", "clinical_output"),
        ("Objetivo de energía", "engine_decision"),
        ("Prueba de interrupción sintética", "provider_failure"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_protected_chat_delivers_fallback_and_audits(monkeypatch, message, reason, stream):
    with monkeypatch.context() as m:
        m.setattr(chat, "_rate_limiters", {})
        app = create(patch=m.setattr)
        audit = AsyncMock()
        m.setattr(chat, "persist_turn", audit)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/chat/stream" if stream else "/api/v1/chat",
                headers={"X-Internal-Key": "kan164-local-fixture"},
                json={"message": message, "supports_vita": True, "user_id": "synthetic"},
            )
    assert response.status_code == 200
    assert (FALLBACK if reason != "provider_failure" else "Inténtalo de nuevo") in response.text
    assert "50 mg" not in response.text
    assert "1500 kcal" not in response.text
    assert "SYNTHETIC PROVIDER DETAIL" not in response.text
    trace = audit.await_args.args[0]
    assert trace.error_code == reason
    assert "50 mg" not in str(trace.to_log())
    if reason in {"clinical_output", "engine_decision"}:
        assert any(event.name.startswith("sha256:") for event in trace.events)


async def test_voice_never_speaks_blocked_output_on_generation_or_replay(rig, monkeypatch):  # noqa: F811
    rig.model.responses = [AIMessage(content=BLOCKED)]
    trace = AsyncMock()
    monkeypatch.setattr(vita_voice, "persist_turn", trace)
    for _ in range(2):
        response = await post(rig, body("Quiero saber más sobre bienestar"))
        assert response.status_code == 200
        assert FALLBACK in response.text and "50 mg" not in response.text
        assert "Haz una pausa consciente" not in response.text
        assert "[DONE]" in response.text
    assert trace.await_args.args[0].error_code == "clinical_output"
    state = await rig.graph.aget_state({"configurable": {"thread_id": "owner::shared"}})
    assert BLOCKED not in str(state.values["messages"])


@pytest.mark.parametrize(
    "answer",
    [
        "Toma una pausa y respira con calma.",
        "Si te gusta, puedes salir a dar un paseo.",
        "Tu plan de hoy está en la app. Puedes revisarlo cuando quieras.",
        "Try setting aside time for a habit you enjoy.",
    ],
)
async def test_voice_keeps_general_wellness_guidance(rig, answer):  # noqa: F811
    rig.model.responses = [AIMessage(content=answer)]
    response = await post(rig, body("Hola Vita"))
    assert answer in response.text
    assert FALLBACK not in response.text


async def test_english_clinical_request_uses_localized_fallback_without_provider(rig):  # noqa: F811
    response = await post(rig, body("What medicine should I take?"))
    assert "Please consult a healthcare professional." in response.text
    assert not rig.model.prompts


async def test_voice_provider_error_returns_normal_termination(rig, monkeypatch):  # noqa: F811
    rig.model.responses = []
    audit = AsyncMock()
    monkeypatch.setattr(vita_voice, "persist_turn", audit)
    response = await post(rig)
    assert response.status_code == 200
    assert "Inténtalo de nuevo" in response.text and "[DONE]" in response.text
    assert audit.await_args.args[0].error_code == "provider_failure"


async def test_legacy_whatsapp_keeps_guard_with_routing_off(monkeypatch):
    monkeypatch.setattr(chat, "_rate_limiters", {})
    app = create(patch=monkeypatch.setattr)
    monkeypatch.setattr(chat.get_settings(), "vita_runtime_enabled", False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/chat",
            headers={"X-Internal-Key": "kan164-local-fixture"},
            json={"message": "Dosis", "agent": "whatsapp", "channel": "whatsapp"},
        )
    assert response.status_code == 200 and FALLBACK in response.json()["answer"]


async def test_new_engine_tool_not_bound_or_executed_on_protected_turn(rig):  # noqa: F811
    from langchain_core.tools import tool

    from app.graph.graph import build_graph

    edit_called = False

    @tool
    async def adjust_engine() -> str:
        """Simula una tool futura que muta el plan."""
        nonlocal edit_called
        edit_called = True
        return "MUTATED"

    rig.model.responses = [
        AIMessage(content="", tool_calls=[{"name": "adjust_engine", "args": {}, "id": "fake"}])
    ]
    rig.graph = build_graph(
        model=rig.model, checkpointer=rig.graph.checkpointer, tools=[adjust_engine]
    )
    response = await post(rig, body("¿Qué puedo hacer hoy?"))
    assert response.status_code == 200 and FALLBACK in response.text
    assert not edit_called and "MUTATED" not in response.text


async def test_active_vita_cannot_fall_back_to_unchecked_voice_transport(rig, monkeypatch):  # noqa: F811
    from app.api.routes import voice

    settings = chat.get_settings()
    monkeypatch.setattr(settings, "internal_api_key", "kan164-local-fixture")
    monkeypatch.setattr(settings, "vita_voice_enabled", False)
    provider = AsyncMock(side_effect=AssertionError("Audio directo fuera del validador"))
    monkeypatch.setattr(voice, "create_signed_url", provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=rig.app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/internal/voice/session",
            json={"user_id": "owner", "supports_vita_voice": True},
            headers={"X-Internal-Key": "kan164-local-fixture"},
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "vita_voice_safety_unavailable"
    provider.assert_not_awaited()


async def test_voice_rechecks_preexisting_checkpoint_before_replay(rig, monkeypatch):  # noqa: F811
    audit = AsyncMock()
    monkeypatch.setattr(vita_voice, "persist_turn", audit)
    await post(rig, body("Hola"))
    config = {"configurable": {"thread_id": "owner::shared"}}
    snapshot = await rig.graph.aget_state(config)
    last = snapshot.values["messages"][-1]
    await rig.graph.aupdate_state(
        config, {"messages": [AIMessage(content=BLOCKED, id=last.id)]}, as_node="agent"
    )
    replay = await post(rig, body("Hola"))
    assert FALLBACK in replay.text and "50 mg" not in replay.text
    trace = audit.await_args.args[0]
    assert trace.status == "rejected" and trace.error_code == "clinical_output"


async def test_clinical_preflight_precedes_provider_routing(rig, monkeypatch):  # noqa: F811
    from app.vita import runtime

    catalog = AsyncMock(return_value=[])
    classify = AsyncMock(
        side_effect=AssertionError("No enviar tema clínico al clasificador remoto")
    )
    monkeypatch.setattr(runtime, "read_catalog", catalog)
    monkeypatch.setattr(runtime, "classify", classify)
    turn = await runtime.resolve_turn(
        user_id="owner",
        text="Qué medicamento tomo",
        session=None,
        base_graph=rig.graph,
        config={"configurable": {"thread_id": "owner::shared", "vita_safety": True}},
    )
    catalog.assert_awaited_once_with("owner")
    classify.assert_not_awaited()
    assert turn.reason == "safety_preflight"
