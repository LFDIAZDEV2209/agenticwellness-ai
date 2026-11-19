"""Captura confirmada de Vita (KAN-70): propuesta → confirmación explícita → recibo idempotente.

Trayecto real modelo (fake) → grafo → nodo de tools → cliente HTTP con transporte simulado
del backend. Sin API keys ni red.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from functools import partial

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agents.runtime_config import AgentRuntimeConfig
from app.agents.runtime_registry import AgentRuntimeRegistry
from app.core.config import get_settings
from app.graph.graph import build_graph
from app.tools.catalog import PROVIDER_CONVERTERS
from app.tools.policy import spec_for
from app.tools.registry import TOOLS_BY_NAME
from app.vita import client
from app.vita.capture import (
    CAPTURE_PROPOSE_TOOLS,
    PENDING_ARTIFACT,
    VITA_CAPTURE_TOOL_NAMES,
    capture_turn_id,
    explicit_confirmation,
    has_pending_proposals,
    previous_turn_proposals,
)
from app.vita.routing import capture_query
from tests.fakes import RecordingFakeModel

KEY = "capture-internal-key"
OWNER = "04b9df0a-b4bd-4779-89e6-72d05e001223"
WAMID = "wamid.HBgMNTczMDAwMDAwMDAwFQIAERgSQ0FQVFVSRVRFU1QA"
TOKEN = "cifrado-aes-gcm-sintetico"
WEIGHT = {"metric_code": "weight", "value": 80.5, "unit_code": "kg"}


def _expires(minutes: int = 5) -> str:
    return (datetime.now(UTC) + timedelta(minutes=minutes)).isoformat()


@pytest.fixture
def backend(monkeypatch):
    """BACK simulado: registra cada request y responde con el handler del test."""
    settings = get_settings()
    monkeypatch.setattr(settings, "backend_base_url", "http://backend.invalid")
    monkeypatch.setattr(settings, "internal_api_key", KEY)
    original = httpx.AsyncClient
    requests: list[httpx.Request] = []
    state = {"unusual": False, "confirm": None, "propose": None}

    async def route(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        if request.url.path.endswith("/capture/propose"):
            if state["propose"] is not None:
                return await state["propose"](request)
            return httpx.Response(
                200,
                json={
                    "token": TOKEN,
                    "expiresAt": _expires(),
                    "requiresAdditionalConfirmation": state["unusual"],
                    "data": {"tool": body["tool"], "metricCode": body.get("metricCode")},
                },
            )
        if state["confirm"] is not None:
            return await state["confirm"](request)
        return httpx.Response(
            200,
            json={
                "receiptId": "5c2b1f1e-0000-4000-8000-000000000001",
                "entityType": "clinical_measurement",
                "entityId": "5c2b1f1e-0000-4000-8000-000000000002",
            },
        )

    transport = httpx.MockTransport(route)
    monkeypatch.setattr(client.httpx, "AsyncClient", partial(original, transport=transport))
    return requests, state


def bodies(requests, action):
    return [json.loads(r.content) for r in requests if r.url.path.endswith(f"/capture/{action}")]


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def capture_graph(*responses):
    model = RecordingFakeModel(responses=list(responses))
    graph = build_graph(
        model=model,
        tools=[TOOLS_BY_NAME[name] for name in sorted(VITA_CAPTURE_TOOL_NAMES)],
        checkpointer=MemorySaver(),
    )
    return graph, model


def config(channel="whatsapp", *, execution="exec-1", user: str | None = OWNER, **extra):
    configurable = {
        "thread_id": f"{OWNER}::capture-{channel}",
        "user_id": user,
        "message_channel": channel,
        "execution_id": execution,
        **extra,
    }
    if channel == "whatsapp":
        configurable.setdefault("channel_message_id", WAMID)
    return {"configurable": configurable}


def tool_results(state, name):
    return [
        json.loads(m.content)
        for m in state["messages"][state.get("turn_start_index", 0) :]
        if isinstance(m, ToolMessage) and m.name == name
    ]


# --- Política y contrato --------------------------------------------------------------


def test_captura_tiene_la_misma_politica_en_los_tres_canales():
    for name in VITA_CAPTURE_TOOL_NAMES:
        assert all(spec_for(name).allows(c) for c in ("text", "voice", "whatsapp"))
        assert spec_for(name).retries == 0
    confirm = spec_for("confirm_capture")
    assert confirm.max_calls_per_turn == 1 and not confirm.idempotent
    assert spec_for("record_measurement").max_calls_per_turn == 2  # sistólica + diastólica


@pytest.mark.parametrize("provider", PROVIDER_CONVERTERS)
@pytest.mark.parametrize("name", sorted(VITA_CAPTURE_TOOL_NAMES))
def test_esquemas_convertibles_sin_identidad_ni_estado(name, provider):
    schema = json.dumps(PROVIDER_CONVERTERS[provider](TOOLS_BY_NAME[name]))
    for forbidden in (
        "user_id",
        "userId",
        "patient_id",
        "patientId",
        "turn_id",
        "channel",
        "state",
    ):
        assert f'"{forbidden}"' not in schema


def test_turn_id_es_estable_distinto_por_contenido_y_acotado():
    first = capture_turn_id(
        WAMID, "record_measurement", {"metricCode": "systolic_bp", "value": 120}
    )
    again = capture_turn_id(
        WAMID, "record_measurement", {"value": 120, "metricCode": "systolic_bp"}
    )
    other = capture_turn_id(
        WAMID, "record_measurement", {"metricCode": "diastolic_bp", "value": 80}
    )
    assert first == again and first != other and first.startswith(f"{WAMID}:")
    long_id = capture_turn_id("w" * 400, "record_meal", {"meal": {"mealCode": "alm"}})
    assert len(long_id) <= 256 and long_id.startswith("sha256:")
    assert capture_turn_id("a\nb", "record_meal", {}).startswith("sha256:")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Sí", True),
        ("sí, por favor", True),
        ("Yes please", True),
        ("Confirmo", True),
        ("ok, regístralo", True),
        ("De acuerdo.", True),
        ("no", False),
        ("Sí pero eran 82", False),
        ("sí, espera", False),
        ("mejor no", False),
        ("cuéntame sobre el sueño", False),
        ("", False),
        ("si " + "x" * 80, False),
    ],
)
def test_confirmacion_explicita_es_determinista(text, expected):
    assert explicit_confirmation(text) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Registra mi peso de 80 kg", True),
        ("Hoy pesé 79,5 kg", True),
        ("Mi presión es 120/80", True),
        ("Ya me tomé la pastilla de la mañana", True),
        ("I took my meds", True),
        ("Log my mood as 4", True),
        ("¿Cuál es mi peso?", False),
        ("¿Qué es la presión arterial?", False),
        ("¿Qué desayuno hoy?", False),
        ("Hola Vita", False),
    ],
)
def test_intencion_de_captura(text, expected):
    assert capture_query(text) is expected


def test_el_artifact_con_token_no_se_serializa_hacia_el_proveedor():
    from langchain_anthropic.chat_models import _format_messages
    from langchain_openai.chat_models.base import _convert_message_to_dict

    message = ToolMessage(
        content='{"status": "proposed"}',
        tool_call_id="p-1",
        name="record_measurement",
        artifact={"kind": PENDING_ARTIFACT, "proposals": [{"token": TOKEN}]},
    )
    assert TOKEN not in str(_format_messages([message]))
    assert TOKEN not in str(_convert_message_to_dict(message))


# --- Flujo en dos fases ---------------------------------------------------------------


async def test_whatsapp_propone_con_identidad_del_turno_y_sin_exponer_el_token(backend):
    requests, _ = backend
    graph, model = capture_graph(
        call("record_measurement", WEIGHT, "p-1"), AIMessage(content="¿Confirmas 80,5 kg?")
    )
    state = await graph.ainvoke({"input": "Registra mi peso: 80,5 kg"}, config=config())
    [proposal] = tool_results(state, "record_measurement")
    assert proposal["status"] == "proposed" and proposal["saved"] is False
    # El token vive solo en el artifact: ni el contenido que lee el modelo ni el payload
    # que se enviaría al proveedor lo contienen.
    assert TOKEN not in json.dumps(proposal)
    assert all(TOKEN not in str(m.content) for prompt in model.prompts for m in prompt)
    [body] = bodies(requests, "propose")
    assert body["channel"] == "vita_whatsapp" and body["tool"] == "record_measurement"
    assert body["turnId"].startswith(f"{WAMID}:")
    assert (body["metricCode"], body["value"], body["unitCode"]) == ("weight", 80.5, "kg")
    assert "userId" not in body and all(r.url.params["userId"] == OWNER for r in requests)
    assert requests[0].headers["X-Internal-Key"] == KEY
    assert bodies(requests, "confirm") == []
    message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert message.artifact["kind"] == PENDING_ARTIFACT
    assert message.artifact["proposals"][0]["token"] == TOKEN


async def test_confirmar_en_el_mismo_turno_de_la_propuesta_no_escribe(backend):
    requests, _ = backend
    graph, _ = capture_graph(
        call("record_measurement", WEIGHT, "p-1"),
        call("confirm_capture", {"confirmed": True}, "c-1"),
        AIMessage(content="Listo"),
    )
    state = await graph.ainvoke({"input": "Sí, registra 80,5 kg"}, config=config())
    [result] = tool_results(state, "confirm_capture")
    assert result["code"] == "vita_capture_no_pending_proposal" and result["saved"] is False
    assert bodies(requests, "confirm") == []


@pytest.mark.parametrize(
    "channel,extra,source",
    [
        ("text", {}, "vita_chat"),
        ("voice", {"voice_user_text": "sí"}, "vita_voice"),
        ("whatsapp", {}, "vita_whatsapp"),
    ],
)
async def test_confirmacion_del_paciente_en_el_turno_siguiente_escribe_una_vez(
    backend, channel, extra, source
):
    requests, _ = backend
    graph, _ = capture_graph(
        call("record_measurement", WEIGHT, "p-1"),
        AIMessage(content="¿Confirmas 80,5 kg?"),
        call("confirm_capture", {"confirmed": True}, "c-1"),
        AIMessage(content="Guardado."),
    )
    # En voz cada frase del paciente tiene su propio ID (repetir el ID es una enmienda ASR).
    first = {"user_message_id": "voice:s1:3:user"} if channel == "voice" else {}
    second = {"user_message_id": "voice:s1:4:user"} if channel == "voice" else {}
    await graph.ainvoke(
        {"input": "Registra mi peso 80,5 kg"}, config=config(channel, **first, **extra)
    )
    state = await graph.ainvoke(
        {"input": "sí"}, config=config(channel, execution="exec-2", **second, **extra)
    )
    [result] = tool_results(state, "confirm_capture")
    assert result["status"] == "recorded" and result["saved"] is True
    [proposed], [confirmed] = bodies(requests, "propose"), bodies(requests, "confirm")
    assert proposed["channel"] == confirmed["channel"] == source
    # El recibo se ancla al turno de ORIGEN, no al turno de la confirmación.
    assert confirmed["turnId"] == proposed["turnId"]
    assert confirmed == {
        "token": TOKEN,
        "channel": source,
        "turnId": proposed["turnId"],
        "confirmed": True,
        "confirmedUnusual": False,
    }


@pytest.mark.parametrize("reply", ["Sí pero eran 82 kg", "cuéntame sobre el sueño"])
async def test_el_modelo_no_puede_sustituir_el_si_del_paciente(backend, reply):
    requests, _ = backend
    graph, _ = capture_graph(
        call("record_measurement", WEIGHT, "p-1"),
        AIMessage(content="¿Confirmas?"),
        call("confirm_capture", {"confirmed": True}, "c-1"),
        AIMessage(content="Necesito un sí o no."),
    )
    await graph.ainvoke({"input": "Registra mi peso 80,5 kg"}, config=config())
    state = await graph.ainvoke({"input": reply}, config=config(execution="exec-2"))
    [result] = tool_results(state, "confirm_capture")
    assert result["code"] == "vita_capture_explicit_confirmation_required"
    assert bodies(requests, "confirm") == []


async def test_rechazo_descarta_sin_llamar_al_backend(backend):
    requests, _ = backend
    graph, _ = capture_graph(
        call("record_checkin", {"mood_score": 4}, "p-1"),
        AIMessage(content="¿Confirmas ánimo 4?"),
        call("confirm_capture", {"confirmed": False}, "c-1"),
        AIMessage(content="No guardé nada."),
    )
    await graph.ainvoke({"input": "Registra mi ánimo en 4"}, config=config())
    state = await graph.ainvoke({"input": "no"}, config=config(execution="exec-2"))
    assert tool_results(state, "confirm_capture")[0]["status"] == "cancelled"
    assert bodies(requests, "confirm") == []
    assert not has_pending_proposals(state)


async def test_fallo_transitorio_conserva_la_propuesta_y_reintenta_con_el_mismo_turn_id(backend):
    requests, state_backend = backend

    async def timeout(_):
        raise httpx.ReadTimeout("simulado")

    state_backend["confirm"] = timeout
    graph, _ = capture_graph(
        call("record_measurement", WEIGHT, "p-1"),
        AIMessage(content="¿Confirmas?"),
        call("confirm_capture", {"confirmed": True}, "c-1"),
        AIMessage(content="No pude guardarlo, ¿reintento?"),
        call("confirm_capture", {"confirmed": True}, "c-2"),
        AIMessage(content="Guardado."),
    )
    await graph.ainvoke({"input": "Registra mi peso 80,5 kg"}, config=config())
    failed = await graph.ainvoke({"input": "sí"}, config=config(execution="exec-2"))
    assert tool_results(failed, "confirm_capture")[0]["saved"] is False
    assert has_pending_proposals(failed)
    state_backend["confirm"] = None
    done = await graph.ainvoke({"input": "sí"}, config=config(execution="exec-3"))
    assert tool_results(done, "confirm_capture")[0]["status"] == "recorded"
    first, second = bodies(requests, "confirm")
    assert first == second  # mismo token y turnId → BACK devuelve el mismo recibo


async def test_valor_inusual_exige_confirmacion_adicional(backend):
    requests, state_backend = backend
    state_backend["unusual"] = True
    graph, _ = capture_graph(
        call(
            "record_measurement",
            {"metric_code": "temperature_c", "value": 45, "unit_code": "celsius"},
            "p-1",
        ),
        AIMessage(content="Es un valor inusual, ¿lo confirmas?"),
        call("confirm_capture", {"confirmed": True}, "c-1"),
        AIMessage(content="Confírmalo otra vez."),
        call("confirm_capture", {"confirmed": True, "confirmed_unusual": True}, "c-2"),
        AIMessage(content="Guardado."),
    )
    await graph.ainvoke({"input": "Registra temperatura 45 grados"}, config=config())
    pending = await graph.ainvoke({"input": "sí"}, config=config(execution="exec-2"))
    assert (
        tool_results(pending, "confirm_capture")[0]["code"]
        == "vita_capture_additional_confirmation_required"
    )
    assert bodies(requests, "confirm") == []
    await graph.ainvoke({"input": "sí, confirmo"}, config=config(execution="exec-3"))
    [confirmed] = bodies(requests, "confirm")
    assert confirmed["confirmedUnusual"] is True


async def test_confirmar_por_otro_canal_no_escribe(backend):
    requests, _ = backend
    graph, _ = capture_graph(
        call("record_measurement", WEIGHT, "p-1"),
        AIMessage(content="¿Confirmas?"),
        call("confirm_capture", {"confirmed": True}, "c-1"),
        AIMessage(content="No."),
    )
    await graph.ainvoke({"input": "Registra mi peso 80,5 kg"}, config=config("text"))
    state = await graph.ainvoke(
        {"input": "sí"},
        config={
            "configurable": {
                **config("text")["configurable"],
                "message_channel": "whatsapp",
                "channel_message_id": WAMID,
            }
        },
    )
    assert tool_results(state, "confirm_capture")[0]["code"] == "vita_capture_turn_mismatch"
    assert bodies(requests, "confirm") == []


async def test_sin_identidad_o_con_selector_de_paciente_no_llama_al_backend(backend):
    requests, _ = backend
    graph, _ = capture_graph(call("record_measurement", WEIGHT, "p-1"), AIMessage(content="x"))
    anonymous = await graph.ainvoke({"input": "Registra mi peso 80,5 kg"}, config=config(user=None))
    assert tool_results(anonymous, "record_measurement")[0]["code"] == "vita_identity_required"
    graph, _ = capture_graph(
        call("record_measurement", {**WEIGHT, "patient_id": OWNER}, "p-1"),
        AIMessage(content="x"),
    )
    injected = await graph.ainvoke({"input": "Registra mi peso 80,5 kg"}, config=config())
    assert tool_results(injected, "record_measurement")[0]["code"] == "tool_not_authorized"
    assert requests == []


async def test_error_de_validacion_del_backend_devuelve_solo_el_codigo(backend):
    _, state_backend = backend

    async def out_of_range(_):
        return httpx.Response(
            400,
            json={"title": "Bad Request", "errors": {"value": ["VITA_CAPTURE_VALUE_OUT_OF_RANGE"]}},
        )

    state_backend["propose"] = out_of_range
    graph, _ = capture_graph(
        call("record_measurement", {**WEIGHT, "value": 900}, "p-1"), AIMessage(content="x")
    )
    state = await graph.ainvoke({"input": "Registra mi peso 900 kg"}, config=config())
    [result] = tool_results(state, "record_measurement")
    assert result["code"] == "vita_capture_value_out_of_range" and result["saved"] is False


# --- Concesión de tools ---------------------------------------------------------------


async def test_config_erp_no_concede_captura_sin_el_overlay_vita():
    registry = AgentRuntimeRegistry(
        model_factory=lambda **_: RecordingFakeModel(responses=[AIMessage(content="ok")]),
        checkpointer_factory=MemorySaver,
    )
    names = ["calculate", "record_measurement", "confirm_capture"]
    assert [t.name for t in registry._select_tools(names, "erp")] == ["calculate"]
    granted = registry._select_tools(names, "erp", allow_capture=True)
    assert [t.name for t in granted] == names
    compiled = await registry._compile("erp", "v1", AgentRuntimeConfig(tools=names))
    overlay = await registry.with_vita_reads(compiled, allow_capture=True)
    assert set(overlay.runtime_config.tools) >= VITA_CAPTURE_TOOL_NAMES
    closed = await registry.with_vita_reads(compiled, allow_capture=False)
    assert not VITA_CAPTURE_TOOL_NAMES & set(closed.runtime_config.tools)


def test_solo_cuentan_propuestas_vigentes_del_turno_anterior():
    def proposal(turn, minutes=5):
        return ToolMessage(
            content="{}",
            tool_call_id=turn,
            name=CAPTURE_PROPOSE_TOOLS[0],
            artifact={
                "kind": PENDING_ARTIFACT,
                "proposals": [
                    {
                        "turn_id": turn,
                        "token": "t",
                        "channel": "vita_chat",
                        "expires_at": _expires(minutes),
                        "tool": CAPTURE_PROPOSE_TOOLS[0],
                    }
                ],
            },
        )

    history = [
        HumanMessage(content="turno 1"),
        proposal("viejo"),
        HumanMessage(content="turno 2"),
        proposal("vigente"),
        proposal("expirado", minutes=-1),
        HumanMessage(content="turno 3 (actual)"),
    ]
    assert [p["turn_id"] for p in previous_turn_proposals(history, 5)] == ["vigente"]
    assert previous_turn_proposals(history, 0) == []


# --- Runtime Vita: la captura usa el planificador en los tres canales ------------------


@pytest.mark.parametrize(
    "channel,explicit",
    [("text", False), ("voice", False), ("whatsapp", False), ("whatsapp", True)],
)
async def test_runtime_enruta_la_captura_al_planificador_con_tools(monkeypatch, channel, explicit):
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock

    from app.agents.prompts import WHATSAPP_FORMAT_POLICY, WHATSAPP_SCOPE_POLICY
    from app.vita import runtime
    from tests.test_runtime_registry import AGENT_ID, VERSION_ID, make_session_with_row

    checkpoint = MemorySaver()
    responses = [
        call("record_measurement", WEIGHT, "p-1"),
        AIMessage(content="¿Confirmas 80,5 kg?"),
    ]
    model = RecordingFakeModel(responses=responses)
    registry = AgentRuntimeRegistry(
        model_factory=lambda **_: model, checkpointer_factory=lambda: checkpoint
    )
    monkeypatch.setattr(runtime, "runtime_registry", registry)
    monkeypatch.setattr(runtime, "get_chat_model", lambda **_: model)
    monkeypatch.setattr(
        runtime,
        "read_catalog",
        AsyncMock(
            return_value=[
                {
                    "id": AGENT_ID,
                    "runtimeKey": AGENT_ID,
                    "slug": "nutricion",
                    "name": "Nutrición",
                    "versionId": VERSION_ID,
                    "versionNumber": 1,
                    # El backend solo publica texto/voz; WhatsApp hereda texto.
                    "routing": {"keywords": [], "topics": [], "channels": ["text", "voice"]},
                }
            ]
        ),
    )
    propose = AsyncMock(return_value=(True, {"token": TOKEN, "expiresAt": _expires(), "data": {}}))
    monkeypatch.setattr("app.tools.vita_capture.post_capture", propose)

    @asynccontextmanager
    async def unlocked(_):
        yield

    monkeypatch.setattr("app.vita.locking.thread_guard", unlocked)
    cfg = config(channel)
    if channel == "voice":
        cfg["configurable"]["user_message_id"] = "voice:s1:1:user"
    text = "Registra mi peso de 80,5 kg"
    turn = await runtime.resolve_turn(
        user_id=OWNER,
        text=text,
        session=make_session_with_row({"provider": "openai", "model": "m", "tools": []}),
        base_graph=build_graph(model=model, checkpointer=checkpoint),
        config=cfg,
        explicit=AGENT_ID if explicit else None,
        channel=channel,
    )
    assert turn.agent_id == (AGENT_ID if explicit else "base")
    result = await turn.graph.ainvoke({"input": text, "agent": turn.agent_id}, config=cfg)
    assert result["tools_used"] == ["record_measurement"]
    (_, user, body, _), _ = propose.await_args
    assert (
        user == OWNER
        and body["channel"]
        == {
            "text": "vita_chat",
            "voice": "vita_voice",
            "whatsapp": "vita_whatsapp",
        }[channel]
    )
    planner = [m.content for m in model.prompts[0] if m.type == "system"]
    assert (WHATSAPP_FORMAT_POLICY in planner) is (channel == "whatsapp")
    assert (planner[0] == WHATSAPP_SCOPE_POLICY) or channel != "whatsapp"
    assert not has_pending_proposals({}) and has_pending_proposals(result)
