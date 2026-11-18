"""Contrato Ola A en el trayecto modelo → grafo → tool → cliente HTTP."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from functools import partial
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agents.runtime_registry import AgentRuntimeRegistry
from app.core.config import get_settings
from app.graph.graph import build_graph
from app.observability.turn_trace import TurnTraceCollector, attach_collector
from app.tools.catalog import PROVIDER_CONVERTERS
from app.tools.registry import TOOLS_BY_NAME
from app.tools.vita import VITA_RECORD_TOOL_NAMES, VITA_TOOL_ARGUMENTS
from app.vita import client, runtime
from tests.fakes import RecordingFakeModel
from tests.test_runtime_registry import AGENT_ID, VERSION_ID, make_session_with_row

OWNER = "04b9df0a-b4bd-4779-89e6-72d05e001223"
OTHER = "872523df-2f71-4dbd-8cc5-792cd0a7943e"
KEY = "test-only-credential"
AS_OF = "2026-11-18T15:00:00Z"
POINT = {
    "metricCode": "heart_rate",
    "value": 68,
    "unitCode": "bpm",
    "unitSymbol": "bpm",
    "observedAt": "2026-11-18T12:00:00Z",
    "source": "device",
}
WEIGHT = {
    **POINT,
    "metricCode": "weight",
    "value": 74.5,
    "unitCode": "kg",
    "unitSymbol": "kg",
    "source": "patient",
}
HEIGHT = {
    **POINT,
    "metricCode": "height",
    "value": 168,
    "unitCode": "cm",
    "unitSymbol": "cm",
    "source": None,
}
PROFILE = {
    "asOf": AS_OF,
    "profile": {
        "firstName": "Paciente sintético",
        "ageYears": 38,
        "gender": "F",
        "timeZone": "America/Bogota",
    },
    "goals": {
        "source": "nutrition_prescription",
        "targetCondition": None,
        "dailyCalorieTarget": 1800,
        "dailyProteinTargetG": 100.5,
        "dailyCarbsTargetG": 200,
        "dailyFatTargetG": None,
        "dailyFiberTargetG": 25,
    },
    "allergies": ["Alérgeno sintético"],
    "restrictions": [],
    "anthropometry": {
        "weight": WEIGHT,
        "height": HEIGHT,
        "bmi": None,
        "history": [WEIGHT, HEIGHT],
        "hasMore": False,
    },
}
VITALS = {
    "asOf": AS_OF,
    "timeZone": "America/Bogota",
    "from": "2026-11-12",
    "to": "2026-11-18",
    "latest": [POINT],
    "history": [POINT],
    "hasMore": False,
}
MEDICATIONS = {
    "asOf": AS_OF,
    "timeZone": "America/Bogota",
    "localDate": "2026-11-18",
    "treatments": [
        {
            "id": OTHER,
            "name": "Suplemento sintético",
            "productType": "supplement",
            "dosage": 1.25,
            "unit": "cápsula",
            "route": "",
            "timeZone": "America/Bogota",
            "times": ["08:00", "20:00"],
            "startDate": "2026-11-12",
            "endDate": None,
            "doses": [
                {
                    "scheduledAt": "2026-11-18T13:00:00Z",
                    "status": "taken",
                    "occurredAt": "2026-11-18T13:05:00Z",
                },
                {"scheduledAt": "2026-11-19T01:00:00Z", "status": "pending", "occurredAt": None},
            ],
        }
    ],
}


@pytest.fixture
def backend(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "backend_base_url", "http://backend.invalid")
    monkeypatch.setattr(settings, "internal_api_key", KEY)
    original = httpx.AsyncClient
    requests = []

    def serve(handler):
        async def route(request):
            requests.append(request)
            return await handler(request)

        transport = httpx.MockTransport(route)
        monkeypatch.setattr(client.httpx, "AsyncClient", partial(original, transport=transport))

    return serve, requests


def scripted_tool(name, args):
    return RecordingFakeModel(
        responses=[
            AIMessage(
                content="plan interno", tool_calls=[{"name": name, "args": args, "id": "read-1"}]
            ),
            AIMessage(content="Respuesta final de prueba."),
        ]
    )


def graph_config(channel="text", user=OWNER):
    return {
        "configurable": {"thread_id": "fixture-thread", "user_id": user, "message_channel": channel}
    }


async def invoke(resource, args=None, *, config=None):
    return await invoke_tool(f"get_my_{resource}", args, config=config)


async def invoke_tool(name, args=None, *, config=None):
    graph = build_graph(
        model=scripted_tool(name, args or {}),
        tools=[TOOLS_BY_NAME[name]],
        checkpointer=MemorySaver(),
    )
    state = await graph.ainvoke(
        {"input": "Muestra mis datos registrados"}, config=config or graph_config()
    )
    assert state["messages"][-1].content == "Respuesta final de prueba."
    message = next(item for item in state["messages"] if isinstance(item, ToolMessage))
    return json.loads(message.content)


@pytest.mark.parametrize("provider", PROVIDER_CONVERTERS)
@pytest.mark.parametrize("name", sorted(VITA_RECORD_TOOL_NAMES))
def test_schema_y_descripcion_convertibles(name, provider):
    tool = TOOLS_BY_NAME[name]
    schema = PROVIDER_CONVERTERS[provider](tool)
    public = schema["function"] if provider == "openai" else schema
    args = public["parameters"] if provider == "openai" else public["input_schema"]
    assert public["description"] == tool.description and len(tool.description) > 50
    expected = VITA_TOOL_ARGUMENTS.get(name, set())
    assert set(args.get("properties", {})) == expected
    assert args.get("required", []) == (["id"] if name == "read_my_document" else [])
    native = (
        tool.args_schema
        if isinstance(tool.args_schema, dict)
        else tool.args_schema.model_json_schema()
    )
    assert native["additionalProperties"] is False


@pytest.mark.parametrize(
    "resource,payload,expected_status",
    [
        ("profile", PROFILE, "ok"),
        ("profile", {**PROFILE, "goals": None}, "ok"),
        ("vitals", VITALS, "ok"),
        ("vitals", {**VITALS, "latest": [], "history": []}, "empty"),
        ("medications", MEDICATIONS, "ok"),
        ("medications", {**MEDICATIONS, "treatments": []}, "empty"),
    ],
)
async def test_dto_del_contrato(resource, payload, expected_status, backend):
    serve, requests = backend

    async def reply(_):
        return httpx.Response(200, json={**payload, "extraBackendField": "omitido"})

    serve(reply)
    result = await invoke(resource)
    if resource == "medications" and expected_status == "empty":
        assert result.pop("message") == "No hay tratamientos vigentes."
    assert result == {"status": expected_status, "data": payload}
    (request,) = requests
    assert request.method == "GET"
    assert request.url.path == f"/api/v1/internal/vita/{resource}"
    assert dict(request.url.params) == {"userId": OWNER}
    assert request.headers["X-Internal-Key"] == KEY


@pytest.mark.parametrize(
    "args",
    [
        {"from": "2026-10-19", "to": "2026-11-18"},
        {"from": "2026-11-18"},
        {"to": "2026-11-18"},
    ],
)
async def test_fechas_locales_llegan_a_http_sin_inventar_el_otro_extremo(args, backend):
    serve, requests = backend

    async def reply(_):
        return httpx.Response(200, json=VITALS)

    serve(reply)
    assert (await invoke("vitals", args))["status"] == "ok"
    (request,) = requests
    assert dict(request.url.params) == {**args, "userId": OWNER}


@pytest.mark.parametrize(
    "args",
    [
        {"from": "2026-11-19", "to": "2026-11-18"},
        {"from": "2026-10-18", "to": "2026-11-18"},
        {"from": "2026-02-30"},
        {"to": 1795000000},
    ],
)
async def test_fechas_invalidas_no_salen_a_http(args, backend):
    _, requests = backend
    result = await invoke("vitals", args)
    assert result["code"] == "invalid_date_range"
    assert "31 días" in result["message"]
    assert requests == []


@pytest.mark.parametrize("resource", ["profile", "vitals", "medications"])
async def test_argumento_de_identidad_inventado_no_llega_al_backend(resource, backend):
    serve, requests = backend

    async def reply(_):
        return httpx.Response(403, json={"detail": "VITA_PATIENT_UNAVAILABLE"})

    serve(reply)
    assert (await invoke(resource, {"userId": OTHER}))["code"] == "tool_not_authorized"
    assert requests == []
    assert (await invoke(resource))["code"] == "vita_patient_unavailable"
    assert [r.url.params["userId"] for r in requests] == [OWNER]


@pytest.mark.parametrize("user", [None, "not-a-uuid", "00000000-0000-0000-0000-000000000000"])
async def test_identidad_invalida_no_sale_a_http(user, backend):
    _, requests = backend
    assert (await invoke("profile", config=graph_config(user=user)))[
        "code"
    ] == "vita_identity_required"
    assert requests == []


@pytest.mark.parametrize(
    "http_status,body,code,safe_text",
    [
        (403, {"detail": "VITA_PATIENT_UNAVAILABLE"}, "vita_patient_unavailable", "perfil activo"),
        (
            403,
            {"detail": "VITA_MEDICATIONS_FORBIDDEN"},
            "vita_medications_forbidden",
            "deshabilitado",
        ),
        (403, {"code": "CONSENT_REQUIRED", "scope": "chat"}, "consent_required", "consentimiento"),
        (404, {"detail": "Módulo no disponible."}, "vita_module_unavailable", "no está disponible"),
        (
            503,
            {"code": "CONSENT_DOCUMENT_UNAVAILABLE", "scope": "chat"},
            "consent_document_unavailable",
            "verificar el consentimiento",
        ),
        (503, {"code": "OTHER"}, "vita_record_unavailable", "no están disponibles"),
        (400, {}, "invalid_date_range", "31 días"),
    ],
)
async def test_error_http_se_traduce_sin_exponer_el_body(
    http_status, body, code, safe_text, backend
):
    serve, requests = backend

    async def reply(_):
        return httpx.Response(http_status, json={**body, "private": "datos privados"})

    serve(reply)
    resource = "vitals" if http_status == 400 else "medications"
    result = await invoke(resource)
    assert len(requests) == 1
    assert result["code"] == code and safe_text in result["message"]
    assert "datos privados" not in json.dumps(result)
    assert "data" not in result


async def test_timeout_y_cancelacion_del_turno(backend, monkeypatch):
    serve, requests = backend
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow(_):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    serve(slow)
    monkeypatch.setattr(get_settings(), "vita_http_timeout", 0.04)
    assert (await invoke("vitals"))["code"] == "tool_timeout"
    assert cancelled.is_set()
    entered.clear()
    cancelled.clear()
    monkeypatch.setattr(get_settings(), "vita_http_timeout", 3.0)
    pending = asyncio.create_task(client.read_record("medications", OWNER))
    await asyncio.wait_for(entered.wait(), 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert cancelled.is_set() and len(requests) == 2


@pytest.mark.parametrize("channel", ["text", "voice"])
@pytest.mark.parametrize("specialist", [False, True])
async def test_runtime_deja_elegir_la_tool_y_voz_responde_desde_answer(
    channel, specialist, backend, monkeypatch
):
    serve, requests = backend

    async def reply(request):
        if request.url.path.endswith("/agents"):
            agents = [
                {
                    "id": AGENT_ID,
                    "runtimeKey": AGENT_ID,
                    "name": "Especialista sintético",
                    "versionId": VERSION_ID,
                    "versionNumber": 1,
                    "routing": {"channels": ["text", "voice"]},
                }
            ]
            return httpx.Response(200, json=agents if specialist else [])
        return httpx.Response(200, json=VITALS)

    serve(reply)
    model = scripted_tool("get_my_vitals", {"from": "2026-11-17", "to": "2026-11-17"})
    monkeypatch.setattr(runtime, "get_chat_model", lambda **_: model)
    checkpoint = MemorySaver()
    monkeypatch.setattr(
        runtime,
        "runtime_registry",
        AgentRuntimeRegistry(
            model_factory=lambda **_: model,
            checkpointer_factory=lambda: checkpoint,
        ),
    )
    lookup_kbs = AsyncMock(return_value=["fixture-kb"])
    monkeypatch.setattr(runtime, "resolve_knowledge_base_ids", lookup_kbs)

    @asynccontextmanager
    async def unlocked(_):
        yield

    monkeypatch.setattr("app.vita.locking.thread_guard", unlocked)
    base = build_graph(model=model, checkpointer=checkpoint)
    config = graph_config(channel)
    text = "¿Cómo salió mi presión de ayer?"
    turn = await runtime.resolve_turn(
        user_id=OWNER,
        text=text,
        session=make_session_with_row({"provider": "openai", "model": "effective", "tools": []})
        if specialist
        else None,
        base_graph=base,
        config=config,
        channel=channel,
        explicit=AGENT_ID if specialist else None,
    )
    updates = [event async for event in turn.graph.astream({"input": text}, config=config)]
    assert [r.url.path.rsplit("/", 1)[-1] for r in requests] == ["agents", "vitals"]
    assert [r.url.params.get("from") for r in requests] == [None, "2026-11-17"]
    lookup_kbs.assert_not_called()
    node = "answer" if channel == "voice" else "agent"
    assert updates[-1][node]["messages"][-1].content == "Respuesta final de prueba."
    assert len(model.prompts) == 2
    assert not any(isinstance(m, ToolMessage) for m in model.prompts[0])


@pytest.mark.parametrize("channel", ["text", "voice"])
async def test_saludo_no_precarga_lecturas_de_ficha(channel, backend, monkeypatch):
    serve, requests = backend

    async def reply(_):
        return httpx.Response(200, json=[])

    serve(reply)
    model = RecordingFakeModel(responses=[AIMessage(content="Hola.")])
    monkeypatch.setattr(runtime, "get_chat_model", lambda **_: model)

    @asynccontextmanager
    async def unlocked(_):
        yield

    monkeypatch.setattr("app.vita.locking.thread_guard", unlocked)
    config = graph_config(channel)
    turn = await runtime.resolve_turn(
        user_id=OWNER,
        text="Hola Vita",
        session=None,
        channel=channel,
        config=config,
        base_graph=build_graph(model=model, checkpointer=MemorySaver()),
    )
    result = await turn.graph.ainvoke({"input": "Hola Vita"}, config=config)
    assert result["messages"][-1].content == "Hola."
    assert len(model.prompts) == 1
    assert [r.url.path.rsplit("/", 1)[-1] for r in requests] == ["agents"]


async def test_whatsapp_bloquea_la_lectura_y_la_traza_no_copia_datos(backend, caplog):
    serve, http_requests = backend

    async def reply(_):
        return httpx.Response(200, json=PROFILE)

    serve(reply)
    rejected = await invoke("profile", config=graph_config("whatsapp"))
    assert rejected["code"] == "tool_not_authorized" and http_requests == []
    trace = TurnTraceCollector(thread_id="fixture-thread", channel="text")
    caplog.set_level(logging.INFO)
    await invoke("profile", config=attach_collector(graph_config(), trace))
    observed = json.dumps(trace.finish(status="completed").to_log())
    assert any(e.kind == "tool" and e.name == "get_my_profile" for e in trace.events)
    for sensitive in ("Paciente sintético", "Alérgeno sintético", KEY):
        assert sensitive not in observed + caplog.text
