"""Harness Vita Ola A: registro tipado de tools (2.6), context builder (2.5),
traza por turno (2.7) y escritura del event log (2.4, sin BD).

La migración y la persistencia real en PostgreSQL viven en
`tests/test_vita_turn_log_postgres.py` (requiere VITA_TEST_DATABASE_URL).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from app.api.deps import get_db_session, get_graph
from app.api.main import create_app
from app.graph.graph import build_graph
from app.observability import turn_trace
from app.observability.turn_trace import TurnTraceCollector, attach_collector, persist_turn
from app.tools.policy import (
    TOOL_SPECS,
    ToolSpec,
    filter_for_channel,
    resolve_channel,
    spec_for,
)
from app.tools.registry import ALL_TOOLS, TOOLS_BY_NAME
from app.tools.vita import VITA_RECORD_TOOL_NAMES
from app.vita.context_builder import (
    SUMMARY_PREFIX,
    ContextBudget,
    build_context,
    estimate_tokens,
    message_tokens,
)
from tests.fakes import FakeToolAwareModel, RecordingFakeModel
from tests.test_api import HEADERS, FakeDbSession

CLINICAL = "mi glucosa en ayunas salió en 187 mg/dL y tomo metformina"
SECRET = "sk-ant-api03-SECRETO-NO-LOGUEAR"


def _call(name: str, call_id: str, args: dict | None = None) -> dict:
    return {"name": name, "args": args or {}, "id": call_id}


def _config(thread: str, **configurable) -> dict:
    return {"configurable": {"thread_id": thread, **configurable}}


# --- 2.6 Registro tipado de tools ---------------------------------------------


def test_toda_tool_registrada_tiene_politica_explicita():
    missing = sorted(t.name for t in ALL_TOOLS if t.name not in TOOL_SPECS)
    assert not missing, f"Tools registradas sin ToolSpec en app/tools/policy.py: {missing}"


def test_tools_de_ficha_tienen_paridad_en_chat_voz_y_whatsapp():
    # KAN-74 8.1: misma matriz de lecturas en los tres canales.
    assert VITA_RECORD_TOOL_NAMES.issubset(TOOLS_BY_NAME)
    for name in VITA_RECORD_TOOL_NAMES:
        for channel in ("text", "voice", "whatsapp"):
            assert spec_for(name).allows(channel)
    # El perfil legado y el flujo de controles siguen fuera de WhatsApp; las citas, fuera de todos.
    assert not spec_for("get_patient_data").allows("whatsapp")
    assert not spec_for("mark_control_declined").allows("whatsapp")
    assert not any(spec_for("suggest_appointment").allows(c) for c in ("text", "voice", "whatsapp"))


def test_politica_por_canal():
    tools = [TOOLS_BY_NAME[n] for n in ("calculate", "mark_control_declined", "get_my_day")]
    assert [t.name for t in filter_for_channel(tools, "text")] == [
        "calculate",
        "mark_control_declined",
        "get_my_day",
    ]
    assert [t.name for t in filter_for_channel(tools, "voice")] == ["calculate", "get_my_day"]
    assert "suggest_appointment" not in [t.name for t in filter_for_channel(ALL_TOOLS, "text")]
    assert resolve_channel({"message_channel": "voice"}) == "voice"
    assert resolve_channel({}, "whatsapp") == "whatsapp"
    assert resolve_channel({"message_channel": "otro"}) == "text"


def test_spec_rechaza_reintentos_en_tool_no_idempotente():
    with pytest.raises(ValueError, match="idempotente"):
        ToolSpec("escritura", retries=1, idempotent=False)
    with pytest.raises(ValueError, match="Canal"):
        ToolSpec("x", channels=frozenset({"sms"}))


async def test_canal_no_puede_invocar_tool_no_permitida():
    """Voz pide `mark_control_declined` (solo app texto): se rechaza sin ejecutarla."""
    executed: list[str] = []

    @tool
    def mark_control_declined() -> str:
        """Tool espía con el mismo nombre que la real."""
        executed.append("x")
        return "ok"

    model = FakeToolAwareModel(
        responses=[
            AIMessage(content="", tool_calls=[_call("mark_control_declined", "c1")]),
            AIMessage(content="Entendido, seguimos."),
        ]
    )
    graph = build_graph(model=model, checkpointer=MemorySaver(), tools=[mark_control_declined])
    result = await graph.ainvoke(
        {"input": "no quiero el control", "messages": []},
        config=_config("policy-voice", message_channel="voice"),
    )
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert json.loads(tool_message.content)["code"] == "tool_not_authorized"
    assert executed == []
    assert result["messages"][-1].content == "Entendido, seguimos."


async def test_presupuesto_por_tool_en_el_turno():
    model = FakeToolAwareModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_call("calculate", f"k{i}", {"expression": "1+1"}) for i in range(4)],
            ),
            AIMessage(content="Listo."),
        ]
    )
    graph = build_graph(model=model, checkpointer=MemorySaver(), tools=[TOOLS_BY_NAME["calculate"]])
    result = await graph.ainvoke({"input": "suma", "messages": []}, config=_config("budget"))
    codes = [
        json.loads(m.content).get("code") if m.status == "error" else "ok"
        for m in result["messages"]
        if isinstance(m, ToolMessage)
    ]
    assert codes == ["ok", "ok", "ok", "tool_budget_exceeded"]
    assert result["tools_used"] == ["calculate"] * 3


def test_presupuesto_global_agotado_responde_util_sin_500():
    looping = [
        AIMessage(
            content="",
            tool_calls=[_call("calculate", f"a{i}", {"expression": "2*2"}) for i in range(2)],
        ),
        AIMessage(content="", tool_calls=[_call("calculate", "b1", {"expression": "3*3"})]),
    ]
    app = create_app()
    app.dependency_overrides[get_graph] = lambda: build_graph(
        model=FakeToolAwareModel(responses=looping),
        checkpointer=MemorySaver(),
        tools=[TOOLS_BY_NAME["calculate"]],
        max_tool_calls=1,
    )
    app.dependency_overrides[get_db_session] = FakeDbSession
    with TestClient(app) as client:
        response = client.post("/api/v1/chat", json={"message": "calcula"}, headers=HEADERS)
    assert response.status_code == 200, response.text
    assert "No pude completar la consulta" in response.json()["answer"]


async def test_deadline_con_reintento_en_tool_idempotente(monkeypatch):
    attempts = {"n": 0}

    @tool
    async def slow_read() -> str:
        """Lectura lenta solo en el primer intento."""
        attempts["n"] += 1
        if attempts["n"] == 1:
            await asyncio.sleep(1)
        return '{"status":"ok"}'

    monkeypatch.setitem(
        TOOL_SPECS, "slow_read", ToolSpec("slow_read", deadline_seconds=0.05, retries=1)
    )
    model = FakeToolAwareModel(
        responses=[
            AIMessage(content="", tool_calls=[_call("slow_read", "s1")]),
            AIMessage(content="Hecho."),
        ]
    )
    collector = TurnTraceCollector(thread_id="t", channel="text")
    graph = build_graph(model=model, checkpointer=MemorySaver(), tools=[slow_read])
    result = await graph.ainvoke(
        {"input": "lee", "messages": []}, config=attach_collector(_config("retry"), collector)
    )
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_message.status != "error" and attempts["n"] == 2
    trace = collector.finish(status="completed")
    assert any(e.kind == "policy" and e.status == "retry" for e in trace.events)


async def test_deadline_agotado_cierra_presupuesto_con_respuesta(monkeypatch):
    @tool
    async def stuck_read() -> str:
        """Nunca responde a tiempo."""
        await asyncio.sleep(1)
        return "x"

    monkeypatch.setitem(
        TOOL_SPECS, "stuck_read", ToolSpec("stuck_read", deadline_seconds=0.02, retries=1)
    )
    model = FakeToolAwareModel(
        responses=[
            AIMessage(content="", tool_calls=[_call("stuck_read", "s1")]),
            AIMessage(content="No pude leerlo ahora."),
        ]
    )
    graph = build_graph(model=model, checkpointer=MemorySaver(), tools=[stuck_read])
    result = await graph.ainvoke({"input": "lee", "messages": []}, config=_config("stuck"))
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert json.loads(tool_message.content)["code"] == "tool_timeout"
    assert result["tool_budget_exhausted"] is True
    assert result["messages"][-1].content == "No pude leerlo ahora."


# --- 2.5 Context builder ------------------------------------------------------


def _simulated_thread(turns: int) -> list:
    messages: list = []
    for i in range(turns):
        messages.append(HumanMessage(content=f"Turno {i}: ¿qué desayuno hoy? prefiero avena {i}"))
        if i % 3 == 0:
            messages.append(AIMessage(content="", tool_calls=[_call("get_my_nutrition", f"t{i}")]))
            messages.append(
                ToolMessage(
                    content=json.dumps({"status": "ok", "plan": "x" * 400}),
                    name="get_my_nutrition",
                    tool_call_id=f"t{i}",
                )
            )
        messages.append(AIMessage(content=f"Respuesta {i}: avena con fruta y agua. " * 3))
    return messages


def test_500_turnos_no_exceden_presupuesto_y_conservan_continuidad():
    budget = ContextBudget(total=8_000)
    thread = _simulated_thread(500)
    built = build_context(
        identity="Eres Vita. " * 50,
        snapshot='{"plan":"vigente"}',
        memory="Prefiere respuestas breves.",
        history=thread,
        budget=budget,
    )
    assert built.total_tokens <= budget.total
    assert estimate_tokens(built.summary or "") <= budget.summary
    # El turno vigente entra completo y al final.
    assert built.history[-1].content.startswith("Respuesta 499")
    assert any(
        isinstance(m, HumanMessage) and m.content.startswith("Turno 499") for m in built.history
    )
    # Continuidad: turnos recientes verbatim + resumen del último compactado.
    kept_turns = sum(isinstance(m, HumanMessage) for m in built.history)
    assert kept_turns >= 5 and built.dropped_turns == 500 - kept_turns
    first_kept = int(
        next(m for m in built.history if isinstance(m, HumanMessage)).content.split()[1][:-1]
    )
    assert f"Turno {first_kept - 1}:" in built.summary
    # Nunca se parte un tool_use de su tool_result.
    ids = [c["id"] for m in built.history if isinstance(m, AIMessage) for c in m.tool_calls]
    results = {m.tool_call_id for m in built.history if isinstance(m, ToolMessage)}
    assert set(ids) == results


def test_compactacion_incremental_arrastra_resumen_previo():
    built = build_context(
        identity="Vita",
        history=_simulated_thread(60),
        previous_summary=f"{SUMMARY_PREFIX} objetivo declarado: caminar 30 minutos",
        budget=ContextBudget(total=8_000),
    )
    assert "caminar 30 minutos" in built.summary


def test_tool_results_del_turno_vigente_se_acotan():
    current = [
        HumanMessage(content="dame todo"),
        AIMessage(content="", tool_calls=[_call("get_my_day", "d1")]),
        ToolMessage(content="y" * 50_000, name="get_my_day", tool_call_id="d1"),
    ]
    built = build_context(identity="Vita", history=current, budget=ContextBudget(total=8_000))
    assert built.layer_tokens["tool_results"] <= 2_500 + 50
    assert built.history[-1].tool_call_id == "d1"


def test_contexto_reciente_es_noop_si_cabe():
    small = _simulated_thread(3)
    built = build_context(identity="Vita", history=small)
    assert built.history == small
    assert built.summary is None
    assert built.total_tokens == sum(message_tokens(m) for m in built.prompt)


async def test_el_presupuesto_incluye_perfil_memoria_y_lectura_preparada(monkeypatch):
    from app.core.config import get_settings

    @tool
    def get_my_day() -> str:
        """Snapshot de prueba con contenido muy grande."""
        return json.dumps({"status": "ok", "data": "snapshot " * 8_000})

    budget = 4_000
    monkeypatch.setattr(get_settings(), "vita_context_token_budget", budget)
    model = RecordingFakeModel(responses=[AIMessage(content="Hola.")])
    graph = build_graph(
        model=model,
        checkpointer=MemorySaver(),
        system_prompt='"perfil\\' * 10_000,
        tools=[get_my_day],
        prepare_tools=["get_my_day"],
    )
    await graph.ainvoke(
        {
            "input": "¿Qué tengo para hoy?",
            "messages": _simulated_thread(500),
            "memory_context": "memoria " * 10_000,
        },
        config=_config("all-layers"),
    )
    prompt = model.prompts[-1]
    assert sum(message_tokens(m) for m in prompt) <= budget
    assert any(m.content == "¿Qué tengo para hoy?" for m in prompt)
    assert any(SUMMARY_PREFIX in str(m.content) for m in prompt)
    result = prompt[-1]
    assert isinstance(result, ToolMessage) and "snapshot" in result.content
    assert result.tool_call_id == prompt[-2].tool_calls[0]["id"]


def test_contexto_extremo_retira_el_lote_completo_para_respetar_el_limite():
    current = [
        HumanMessage(content="Última pregunta"),
        AIMessage(content="", tool_calls=[_call("calculate", "huge", {"x": "z" * 100_000})]),
        ToolMessage(content="resultado", name="calculate", tool_call_id="huge"),
    ]
    built = build_context(identity="Vita", history=current, budget=ContextBudget(total=4_000))
    assert built.total_tokens <= 4_000
    assert built.history == current[:1]


async def test_nodo_agente_no_envia_hilo_completo_de_500_turnos():
    model = RecordingFakeModel(responses=[AIMessage(content="Hola de nuevo.")])
    graph = build_graph(model=model, checkpointer=MemorySaver())
    await graph.ainvoke(
        {"input": "¿y hoy qué como?", "messages": _simulated_thread(500)},
        config=_config("long-thread"),
    )
    prompt = model.prompts[-1]
    from app.core.config import get_settings

    assert sum(message_tokens(m) for m in prompt) <= get_settings().vita_context_token_budget
    assert any(SUMMARY_PREFIX in str(m.content) for m in prompt)
    assert prompt[-1].content == "¿y hoy qué como?"


# --- 2.7 Traza por turno sin contenido clínico ni secretos ---------------------


async def test_traza_registra_nodo_tool_tokens_y_latencia_sin_contenido(caplog):
    @tool
    def get_my_progress() -> str:
        """Lectura que devuelve datos clínicos y un secreto."""
        return json.dumps(
            {"status": "ok", "nota": CLINICAL, "token": SECRET, "code": "glucosa_token_privado"}
        )

    model = FakeToolAwareModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_call("get_my_progress", "p1")],
                usage_metadata={"input_tokens": 120, "output_tokens": 8, "total_tokens": 128},
            ),
            AIMessage(
                content=f"Tu valor fue {CLINICAL}",
                usage_metadata={"input_tokens": 200, "output_tokens": 30, "total_tokens": 230},
            ),
        ]
    )
    collector = TurnTraceCollector(
        thread_id="owner::thread", channel="text", patient_id="patient-1", agent_id="base"
    )
    graph = build_graph(model=model, checkpointer=MemorySaver(), tools=[get_my_progress])
    with caplog.at_level(logging.DEBUG):
        state = await graph.ainvoke(
            {"input": f"{CLINICAL} {SECRET}", "messages": []},
            config=attach_collector(_config("trace"), collector),
        )
        trace = collector.finish(status="completed", model="claude-sonnet-4-5")
        await persist_turn(trace, session_factory=None)  # log técnico (BD desactivada)

    kinds = {(e.kind, e.name) for e in trace.events}
    assert ("node", "guardrails") in kinds and ("node", "agent") in kinds
    assert ("node", "tools") in kinds and ("tool", "get_my_progress") in kinds
    assert any(e.kind == "llm" for e in trace.events)
    assert (trace.tokens_in, trace.tokens_out) == (320, 38)
    assert all(e.latency_ms is not None and e.latency_ms >= 0 for e in trace.events)
    assert [e.seq for e in trace.events] == list(range(1, len(trace.events) + 1))
    assert state["messages"][-1].content.startswith("Tu valor fue")

    serialized = json.dumps(asdict(trace), default=str) + json.dumps(trace.to_log())
    logs = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("app."))
    # Los timestamps contienen dígitos al azar (p. ej. «…416187»), que pueden
    # coincidir con un valor clínico sin que haya fuga. La comprobación anti-PHI
    # solo debe mirar contenido y metadatos estables.
    timestamp = r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?"
    serialized = re.sub(timestamp, "", serialized)
    logs = re.sub(timestamp, "", logs)
    for leaked in ("glucosa", "187", "metformina", "SECRETO", "sk-ant"):
        assert leaked not in serialized, f"La traza contiene '{leaked}'"
        assert leaked not in logs, f"Los logs contienen '{leaked}'"
    assert "patient-1" not in json.dumps(trace.to_log())


async def test_error_code_nunca_incluye_el_texto_de_la_excepcion():
    collector = TurnTraceCollector(thread_id="t", channel="text")
    run_id = __import__("uuid").uuid4()
    await collector.on_tool_start({"name": "get_my_day"}, "", run_id=run_id)
    await collector.on_tool_error(ValueError(f"fallo con {CLINICAL}"), run_id=run_id)
    event = collector.finish(status="failed", error_code="Texto libre no permitido").events[0]
    assert event.error_code == "exc_valueerror"
    assert collector.finish(status="failed", error_code="Texto libre").error_code is None


async def test_la_traza_no_guarda_nombres_de_tools_inventados():
    model = FakeToolAwareModel(
        responses=[
            AIMessage(content="", tool_calls=[_call("glucosa_token_privado", "fake")]),
            AIMessage(content="Entendido."),
        ]
    )
    collector = TurnTraceCollector(thread_id="trace-unknown", channel="text")
    graph = build_graph(model=model, checkpointer=MemorySaver(), tools=[])
    await graph.ainvoke(
        {"input": "hola"}, config=attach_collector(_config("trace-unknown"), collector)
    )
    trace = collector.finish(status="completed")
    assert any(event.kind == "policy" and event.name == "unknown" for event in trace.events)
    assert "glucosa_token_privado" not in json.dumps(trace.to_log())


# --- 2.4 Escritura del event log (sesión inyectada) ----------------------------


class _RecordingSession:
    def __init__(self, sink: list):
        self.sink = sink

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def add(self, obj):
        self.sink.append(obj)

    def add_all(self, objs):
        self.sink.extend(objs)

    async def flush(self):
        pass

    async def commit(self):
        self.sink.append("commit")


async def test_persist_turn_inserta_turno_y_eventos_en_una_transaccion():
    from app.db.models import VitaTurn, VitaTurnEvent

    collector = TurnTraceCollector(thread_id="o::t", channel="whatsapp", patient_id="p")
    collector._append("node", "agent", "ok", latency_ms=5)
    collector._append("tool", "calculate", "ok", latency_ms=1)
    sink: list = []
    turn_id = await persist_turn(
        collector.finish(status="completed", message_ref="m-1"),
        session_factory=lambda: _RecordingSession(sink),
    )
    turn, *events, commit = sink
    assert isinstance(turn, VitaTurn) and turn.id == turn_id and commit == "commit"
    assert (turn.channel, turn.status, turn.message_ref) == ("whatsapp", "completed", "m-1")
    assert [(e.seq, e.kind, e.name) for e in events] == [
        (1, "node", "agent"),
        (2, "tool", "calculate"),
    ]
    assert all(isinstance(e, VitaTurnEvent) and e.turn_id == turn_id for e in events)


async def test_persist_turn_es_best_effort(caplog):
    def broken():
        raise ConnectionError(f"host=db password={SECRET}")

    with caplog.at_level(logging.WARNING):
        result = await persist_turn(
            TurnTraceCollector(thread_id="t", channel="text").finish(status="completed"),
            session_factory=broken,
        )
    assert result is None
    assert "vita_turn_log_failed error=ConnectionError" in caplog.text
    assert SECRET not in caplog.text


def test_chat_http_registra_turno_whatsapp_sin_cambiar_contrato(monkeypatch):
    captured: list = []

    async def capture(trace, **_):
        captured.append(trace)

    monkeypatch.setattr("app.api.routes.chat.persist_turn", capture)
    app = create_app()
    app.dependency_overrides[get_graph] = lambda: build_graph(
        model=FakeToolAwareModel(responses=[AIMessage(content="Hola, te ayudo.")]),
        checkpointer=MemorySaver(),
    )
    app.dependency_overrides[get_db_session] = FakeDbSession
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={"message": CLINICAL, "agent": "whatsapp", "user_id": "u-1"},
            headers=HEADERS,
        )
    assert response.status_code == 200
    assert set(response.json()) >= {"thread_id", "answer", "execution_id", "tools_used"}
    (trace,) = captured
    assert (trace.channel, trace.status, trace.patient_id) == ("whatsapp", "completed", "u-1")
    assert trace.execution_id == response.json()["execution_id"]
    assert "glucosa" not in json.dumps(asdict(trace), default=str)


def test_chat_stream_registra_turno(monkeypatch):
    captured: list = []

    async def capture(trace, **_):
        captured.append(trace)

    monkeypatch.setattr("app.api.routes.chat.persist_turn", capture)
    app = create_app()
    app.dependency_overrides[get_graph] = lambda: build_graph(
        model=FakeToolAwareModel(responses=[AIMessage(content="Hola.")]),
        checkpointer=MemorySaver(),
    )
    app.dependency_overrides[get_db_session] = FakeDbSession
    with TestClient(app) as client:
        response = client.post("/api/v1/chat/stream", json={"message": "hola"}, headers=HEADERS)
    assert response.status_code == 200 and "event: done" in response.text
    (trace,) = captured
    assert trace.channel == "text" and trace.status == "completed"
    assert any(e.kind == "node" and e.name == "agent" for e in trace.events)


def test_turn_trace_module_no_guarda_entradas():
    """Contrato estructural: TraceEvent no tiene campos de contenido."""
    fields = set(turn_trace.TraceEvent.__dataclass_fields__)
    assert fields == {
        "seq",
        "kind",
        "name",
        "status",
        "latency_ms",
        "tokens_in",
        "tokens_out",
        "error_code",
    }
