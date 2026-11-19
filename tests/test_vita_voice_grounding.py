"""Replay sintético del incidente KAN-101, con el runtime y el checkpoint reales."""

import hashlib
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Overwrite

from app.agents.prompts import VOICE_DELIVERY_POLICY, build_patient_voice_prompt
from app.core.config import get_settings
from app.graph.graph import build_graph
from app.vita import runtime
from app.vita.contracts import RecordReadResult, VitalsRecord
from app.vita.routing import voice_record_followup
from app.voice.delivery import format_voice_answer
from tests.fakes import RecordingFakeModel
from tests.test_vita_extended_reads import ASSESSMENTS
from tests.test_vita_record_tools import OWNER, POINT, VITALS, graph_config, invoke
from tests.test_vita_record_tools import backend as backend

BOGOTA_POINTS = [
    {**POINT, "observedAt": "2026-11-19T04:40:00Z", "value": 72},
    {**POINT, "observedAt": "2026-11-19T01:00:00Z", "value": 70},
    {**POINT, "observedAt": "2026-11-18T01:00:00Z", "value": 68},
]


@pytest.fixture
def voice_runtime(monkeypatch):
    monkeypatch.setattr(runtime, "resolve_knowledge_base_ids", AsyncMock(return_value=["global"]))

    @asynccontextmanager
    async def unlocked(_):
        yield

    monkeypatch.setattr("app.vita.locking.thread_guard", unlocked)

    def setup(model, channel="voice", language="es"):
        monkeypatch.setattr(runtime, "get_chat_model", lambda **_: model)
        base = build_graph(model=model, checkpointer=MemorySaver())
        config = graph_config(channel)
        config["configurable"].update(language=language, patient_id=OWNER)

        async def turn(text):
            resolved = await runtime.resolve_turn(
                user_id=OWNER,
                text=text,
                session=None,
                base_graph=base,
                config=config,
                channel=channel,
            )
            return await resolved.graph.ainvoke({"input": text}, config=config)

        return base, config, turn

    return setup


@pytest.mark.parametrize(
    "language,queries,first_plan,grounded",
    [
        (
            "es",
            [
                "¿Puedes revisar mis signos vitales?",
                "Muéstrame los últimos días",
                "Los del reloj, la manilla, el wearable",
                "¿Cómo así 17 de nov? ¿De qué hablas?",
            ],
            "Sí, puedo revisar tus signos vitales.",
            "El 18 de noviembre registraste 72 bpm. El 17 de noviembre registraste 68 bpm.",
        ),
        (
            "en",
            [
                "Can you check my vitals?",
                "What about last week?",
                "The ones from my watch",
                "What do you mean by November 17?",
            ],
            "Yes, I can check your vitals.",
            "On November 18 you recorded 72 bpm. On November 17 you recorded 68 bpm.",
        ),
    ],
)
async def test_voice_replays_vitals_and_keeps_local_dates(
    language, queries, first_plan, grounded, voice_runtime, backend
):
    serve, requests = backend

    async def reply(request):
        if request.url.path.endswith("/agents"):
            return httpx.Response(200, json=[])
        assert request.url.params["userId"] == OWNER
        return httpx.Response(
            200, json={**VITALS, "latest": BOGOTA_POINTS[:1], "history": BOGOTA_POINTS}
        )

    class GroundedFake(RecordingFakeModel):
        def _generate(self, messages, *args, **kwargs):
            if read := next((m for m in messages if isinstance(m, ToolMessage)), None):
                data = json.loads(read.content)["data"]
                assert [(p["localDate"], p["value"]) for p in data["history"]] == [
                    ("2026-11-18", 72),
                    ("2026-11-17", 68),
                ]
                assert any(
                    VOICE_DELIVERY_POLICY in m.content
                    for m in messages
                    if isinstance(m, SystemMessage)
                )
            return super()._generate(messages, *args, **kwargs)

    serve(reply)
    model = GroundedFake(
        responses=[
            AIMessage(content=first_plan),
            AIMessage(content=grounded),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "retrieve_knowledge",
                        "args": {"query": "nutrition"},
                        "id": "wrong-plan",
                    }
                ],
            ),
            AIMessage(content=grounded),
            AIMessage(content=first_plan),
            AIMessage(content=grounded),
            AIMessage(content=first_plan),
            AIMessage(content=grounded),
        ]
    )
    base, config, run = voice_runtime(model, language=language)
    for index, query in enumerate(queries):
        state = await run(query)
        assert state["record_topic"] == "vitals"
        assert state["tools_used"] == ["get_my_vitals"]
        assert state["messages"][-1].content == grounded
        # Simula el checkpoint del incidente, con tema planificado pero sin tools usadas.
        if index == 0:
            await base.aupdate_state(config, {"tools_used": Overwrite([])}, as_node="agent")

    assert [request.url.path.rsplit("/", 1)[-1] for request in requests] == ["agents", "vitals"] * 4


@pytest.mark.parametrize(
    "query", ["y ayer", "y los de esta semana", "entonces cuáles", "dime el resto"]
)
def test_voice_recognizes_elliptical_record_followups(query):
    assert voice_record_followup(query)


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Hola Vita", []),
        ("¿Qué puedes hacer?", []),
        ("¿Qué recomienda la guía sobre mi presión?", ["retrieve_knowledge:program"]),
        ("¿Qué dice mi informe de ayer?", ["retrieve_knowledge:patient"]),
        ("Registra mi peso de hoy, 72 kg", ["record_measurement"]),
    ],
)
async def test_voice_record_topic_does_not_override_other_routes(
    query, expected, voice_runtime, monkeypatch
):
    monkeypatch.setattr(runtime, "read_catalog", AsyncMock(return_value=[]))
    monkeypatch.setattr(get_settings(), "vita_capture_enabled", True)
    calls = []

    @tool
    async def retrieve_knowledge(query: str, scope: str = "program") -> str:
        """Devuelve evidencia sintética de la fuente solicitada."""
        calls.append(f"retrieve_knowledge:{scope}")
        return "Evidencia sintética."

    @tool
    async def record_measurement() -> str:
        """Devuelve una propuesta sintética sin guardar nada."""
        calls.append("record_measurement")
        return "Propuesta sintética."

    monkeypatch.setattr(runtime, "make_retrieve_tool", lambda *args, **kwargs: retrieve_knowledge)
    monkeypatch.setitem(runtime.TOOLS_BY_NAME, "record_measurement", record_measurement)
    final = AIMessage(content="Respuesta sintética.")
    responses = (
        [
            AIMessage(
                content="", tool_calls=[{"id": "capture", "name": "record_measurement", "args": {}}]
            ),
            final,
        ]
        if expected == ["record_measurement"]
        else [final]
    )
    base, config, run = voice_runtime(RecordingFakeModel(responses=responses))
    await base.aupdate_state(
        config,
        {"record_topic": "vitals", "messages": [AIMessage(content="Lectura anterior.")]},
        as_node="agent",
    )

    state = await run(query)
    assert calls == expected
    assert state["record_topic"] is None
    assert "get_my_vitals" not in state["tools_used"]


@pytest.mark.parametrize("channel", ["voice", "text", "whatsapp"])
async def test_vitals_history_is_bounded_only_for_voice(channel, backend):
    serve, _ = backend
    history = [
        {
            **POINT,
            "value": index,
            "observedAt": (datetime(2026, 11, 19, 4, 30, tzinfo=UTC) - timedelta(hours=index * 12))
            .isoformat()
            .replace("+00:00", "Z"),
        }
        for index in range(20)
    ]

    async def reply(_):
        return httpx.Response(
            200,
            json={
                **VITALS,
                "from": "2026-11-01",
                "latest": history[:1],
                "history": history,
            },
        )

    serve(reply)
    result = await invoke("vitals", config=graph_config(channel))
    data = result["data"]
    assert data["history"][0]["observedAt"] == "2026-11-19T04:30:00Z"
    assert data["history"][0]["localDate"] == "2026-11-18"
    assert data["history"][0]["localTime"] == "23:30"
    assert data["latest"][0]["localDate"] == "2026-11-18"
    if channel == "voice":
        assert data["truncated"] and data["daysReturned"] == 7
        assert [point["value"] for point in data["history"]] == list(range(0, 14, 2))
    else:
        assert len(data["history"]) == 20
        assert "truncated" not in data and "daysReturned" not in data


async def test_voice_vitals_empty_result_has_explicit_status_and_no_data(backend):
    serve, _ = backend

    async def reply(_):
        return httpx.Response(200, json={**VITALS, "latest": [], "history": []})

    serve(reply)
    assert await invoke("vitals", config=graph_config("voice")) == {"status": "empty"}


@pytest.mark.parametrize("status", ["forbidden", "unavailable", "stale"])
async def test_voice_vitals_non_ok_results_do_not_expose_values(status, monkeypatch):
    read = AsyncMock(
        return_value=RecordReadResult(status=status, data=VitalsRecord.model_validate(VITALS))
    )
    monkeypatch.setattr("app.tools.vita.read_record", read)
    assert await invoke("vitals", config=graph_config("voice")) == {"status": status}
    read.assert_awaited_once()


@pytest.mark.parametrize("channel", ["voice", "text", "whatsapp"])
@pytest.mark.parametrize("language", ["es", "en"])
async def test_nine_assessment_results_are_summarized_in_voice(
    channel, language, backend, voice_runtime
):
    serve, _ = backend
    battery = ASSESSMENTS["batteries"][1]
    title = "Evaluación" if language == "es" else "Assessment"
    items = [
        {**battery["tests"][0], "testName": f"{title} {index}", "testCode": f"test-{index}"}
        for index in range(1, 10)
    ]

    async def reply(request):
        return httpx.Response(
            200,
            json=[]
            if request.url.path.endswith("/agents")
            else {"batteries": [{**battery, "tests": items}]},
        )

    serve(reply)
    lengthy = f"## {title}\n" + "\n".join(
        f"- **{item['testName']}** recibió 7 puntos en total."
        if language == "es"
        else f"- **{item['testName']}** scored 7 points overall."
        for item in items
    )
    model = RecordingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_my_assessment_results", "args": {}, "id": "assessment-results"}
                ],
            ),
            AIMessage(content=lengthy),
        ]
    )
    _, _, run = voice_runtime(model, channel=channel, language=language)
    state = await run("Mis resultados registrados" if language == "es" else "My recorded scores")
    answer = state["messages"][-1].content
    assert state["tools_used"] == ["get_my_assessment_results"]
    read = next(message for message in model.prompts[-1] if isinstance(message, ToolMessage))
    assert len(json.loads(read.content)["data"]["batteries"][0]["tests"]) == 9
    if channel == "voice":
        assert len(answer.split()) <= 40
        assert 3 <= answer.count(title) <= 5
        assert answer.count(".") == 2 and "chat" in answer
        assert not any(marker in answer for marker in ("##", "**", "\n-"))
    else:
        assert answer == lengthy


@pytest.mark.parametrize(
    "language,action", [("es", "Llama al SOS ahora."), ("en", "Call 911 now.")]
)
def test_urgent_safety_guidance_keeps_all_instructions(language, action):
    guidance = action + " Stay in a safe place until help arrives." * 9
    assert format_voice_answer(guidance, language=language, has_results=True) == guidance


def test_voice_general_budget_and_decimal_unit_are_respected():
    essential = "Tu medición fue de 74.5 kg el 18 de noviembre."
    long = " ".join(
        ["Esta explicación extensa incluye muchos detalles repetidos por cada caso."] * 6
    )
    answer = format_voice_answer(f"{essential} {long}", language="es", has_results=False)
    assert answer.startswith(essential)
    assert len(answer.split()) <= 50


@pytest.mark.parametrize(
    "language,digest",
    [
        ("es", "3ce472b6d2b006d0d444eea8fcfc82d3275da33b2b8bd72edecb7b761f3bd03f"),
        ("en", "ebf2478174ae4bd5d9801b884fa4a9345e7af36886558278ff639a2435f0fc12"),
    ],
)
def test_custom_llm_policy_does_not_change_pinned_elevenlabs_prompt(language, digest):
    assert hashlib.sha256(build_patient_voice_prompt(language).encode()).hexdigest() == digest
