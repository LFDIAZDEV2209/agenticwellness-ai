"""Dos pacientes con el mismo ID de hilo, memoria y trazas en PostgreSQL real."""

import json
import logging
from copy import deepcopy
from dataclasses import asdict
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.routes.chat import _build_config, _turn_collector
from app.api.schemas import ChatRequest
from app.db.models import AgentExperience, AgentMemory, VitaTurn, VitaTurnEvent
from app.graph.graph import build_graph
from app.memory.adaptive import AdaptiveMemoryService
from app.memory.postgres import UserMemoryStore
from app.memory.service import UserMemoryService
from app.observability.turn_trace import attach_collector, persist_turn
from app.tools.registry import TOOLS_BY_NAME
from app.tools.vita import VITA_READ_TOOL_NAMES
from tests import test_vita_extended_reads, test_vita_record_tools
from tests.fakes import RecordingFakeModel
from tests.test_vita_extended_reads import (
    ASSESSMENTS,
    DOC_ID,
    DOCUMENT,
    DOCUMENTS,
    NUTRITION,
    SIGNED_PATH,
)
from tests.test_vita_history_postgres import isolated_test_url
from tests.test_vita_record_tools import MEDICATIONS, OTHER, OWNER, PROFILE, VITALS, invoke_tool

backend = test_vita_record_tools.backend
storage = test_vita_extended_reads.storage

USERS = {OWNER: "ficha-sintetica-alfa", OTHER: "ficha-sintetica-beta"}
DOCUMENT_IDS = {OWNER: DOC_ID, OTHER: "7dfd163f-711b-4b15-8e4b-de9ee0324812"}
TABLES = [
    AgentMemory.__table__,
    AgentExperience.__table__,
    VitaTurn.__table__,
    VitaTurnEvent.__table__,
]


@pytest.fixture
async def sessions():
    schema = f"vita_isolation_{uuid4().hex}"
    engine = create_async_engine(
        isolated_test_url(), execution_options={"schema_translate_map": {"ai": schema}}
    )
    try:
        async with engine.begin() as sql:
            await sql.execute(text(f'CREATE SCHEMA "{schema}"'))
            await sql.run_sync(lambda conn: AgentMemory.metadata.create_all(conn, tables=TABLES))
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        async with engine.begin() as sql:
            await sql.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()


def documents_for(user):
    return {**DOCUMENT, "id": DOCUMENT_IDS[user], "name": USERS[user]}


def fixture_dtos(user):
    marker = USERS[user]
    profile, vitals, meds, nutrition, assessments = deepcopy(
        (PROFILE, VITALS, MEDICATIONS, NUTRITION, ASSESSMENTS)
    )
    profile["profile"]["firstName"] = marker
    vitals["latest"][0]["metricCode"] = marker
    meds["treatments"][0]["name"] = marker
    nutrition["recommendations"][0]["patientMessage"] = marker
    assessments["batteries"][0]["name"] = marker
    return {
        "profile": profile,
        "vitals": vitals,
        "medications": meds,
        "nutrition": nutrition,
        "assessments": assessments,
        "documents": {**DOCUMENTS, "items": [documents_for(user)]},
        "context": {"status": "ok", "data": {"fixture": marker}},
    }


def assert_isolated(own_marker, other_marker, state, prompts, trace):
    assert own_marker in state["memory_context"]
    assert other_marker not in state["memory_context"], "memoria de otro paciente"
    for message in state["messages"]:
        if isinstance(message, ToolMessage):
            assert own_marker in message.content, message.name
            assert other_marker not in message.content, f"{message.name}: dato ajeno"
    assert all(other_marker not in str(prompt) for prompt in prompts), "contexto de otro paciente"
    assert all(marker not in trace for marker in USERS.values()), "dato clínico en traza"


@pytest.mark.parametrize("drop_owner_filter", [False, True])
async def test_aislamiento_de_tools_memoria_y_traza(
    drop_owner_filter, sessions, backend, storage, monkeypatch, caplog
):
    serve, requests = backend
    response, downloads = storage
    response["body"] = lambda path: next(
        marker.encode() for marker in USERS.values() if marker in path
    )

    async def handler(request):
        user = request.url.params["userId"]
        assert user in USERS
        resource = request.url.path.rsplit("/", 1)[-1]
        if "/documents/" in request.url.path:
            if resource != DOCUMENT_IDS[user]:
                return httpx.Response(404)
            return httpx.Response(
                200,
                json={
                    "document": documents_for(user),
                    "expiresInSeconds": 120,
                    "url": SIGNED_PATH.replace("record.txt", f"{USERS[user]}.txt"),
                },
            )
        return httpx.Response(200, json=fixture_dtos(user)[resource])

    serve(handler)
    async with sessions() as seed:
        seed.add_all(
            AgentMemory(user_id=owner, agent_type_id="base", category="preferencia", content=marker)
            for owner, marker in USERS.items()
        )
        await seed.commit()

    if drop_owner_filter:

        async def without_owner(self, **_):
            # Mutación deliberada: mantiene BD y servicio reales, omite solo el scope.
            return list((await self._session.scalars(select(AgentMemory))).all())

        monkeypatch.setattr(UserMemoryStore, "list_memories", without_owner)

    names = sorted(VITA_READ_TOOL_NAMES)
    model = RecordingFakeModel(
        responses=[
            message
            for user in USERS
            for message in (
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": name,
                            "id": f"fixture-{i}",
                            "args": {"id": DOCUMENT_IDS[user]}
                            if name == "read_my_document"
                            else {},
                        }
                        for i, name in enumerate(names)
                    ],
                ),
                AIMessage(content=f"Respuesta {USERS[user]}"),
            )
        ]
    )
    graph = build_graph(
        model=model,
        checkpointer=MemorySaver(),
        max_tool_calls=len(names),
        tools=[TOOLS_BY_NAME[name] for name in names],
        enable_memory=True,
    )
    thread = "mismo-id-visible"
    caplog.set_level(logging.INFO, logger="app.observability.turn_trace")
    completed = []
    for user, other in ((OWNER, OTHER), (OTHER, OWNER)):
        request = ChatRequest(message=f"Consulta {USERS[user]}", thread_id=thread, user_id=user)
        collector = _turn_collector(request, thread, str(uuid4()))
        config = attach_collector(_build_config(thread, request), collector)
        first_prompt = len(model.prompts)
        async with sessions() as session:
            monkeypatch.setattr(
                "app.graph.nodes._default_service", lambda: UserMemoryService(session)
            )
            monkeypatch.setattr(
                "app.graph.nodes._default_adaptive_service", lambda: AdaptiveMemoryService(session)
            )
            state = await graph.ainvoke({"input": request.message, "agent": "base"}, config=config)
        trace = collector.finish(status="completed")
        serialized = json.dumps(asdict(trace), default=str)
        turn_id = await persist_turn(trace, session_factory=sessions)
        assert turn_id and trace.patient_id == user and other not in serialized
        assert {event.name for event in trace.events if event.kind == "tool"} == set(names)
        assert set(state["tools_used"]) == set(names)

        if drop_owner_filter:
            with pytest.raises(AssertionError, match="memoria de otro paciente"):
                assert_isolated(
                    USERS[user], USERS[other], state, model.prompts[first_prompt:], serialized
                )
        else:
            assert_isolated(
                USERS[user], USERS[other], state, model.prompts[first_prompt:], serialized
            )
        completed.append((user, turn_id))

    async with sessions() as session:
        for user, turn_id in completed:
            record = await session.get(VitaTurn, turn_id)
            assert record.thread_id == f"{user}::{thread}" and record.patient_id == user
    assert all(marker not in caplog.text for marker in USERS.values())
    assert len(requests) == len(names) * 2 and len(downloads) == 2

    denied = await invoke_tool(
        "read_my_document",
        {"id": DOCUMENT_IDS[OTHER]},
        config=_build_config(thread, ChatRequest(message="Consulta", user_id=OWNER)),
    )
    assert denied["code"] == "vita_document_unavailable" and len(downloads) == 2
    assert requests[-1].url.params["userId"] == OWNER
