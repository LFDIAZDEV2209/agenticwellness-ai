"""Smoke tests de la API (con el grafo sobreescrito con un LLM falso)."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.api.deps import get_db_session, get_graph
from app.api.main import create_app
from app.core.config import get_settings
from app.graph.graph import build_graph
from app.history import service as history_service
from app.vita.runtime import ResolvedTurn
from tests.fakes import FakeToolAwareModel
from tests.vita_safety_cases import (
    HIDDEN_MARKER,
    PRIOR_ANSWER,
    SAFETY_CASES,
    ChunkedSafetyModel,
    assert_context_delivery,
    assert_safe_delivery,
    sse_payloads,
)

FAKE_ANSWER = "Respuesta de prueba"

# Los endpoints funcionales exigen X-Internal-Key (canal backend → AI).
INTERNAL_KEY = get_settings().internal_api_key
HEADERS = {"X-Internal-Key": INTERNAL_KEY}


class FakeDbSession:
    """Sesión de BD fake: los smoke tests no tocan Postgres.

    El tracker de observabilidad y los nodos de memoria usan la sesión; en
    los tests solo se verifica que el chat responde, no la persistencia.
    """

    async def add(self, *args, **kwargs):
        pass

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def get(self, *args, **kwargs):
        return None

    async def execute(self, *args, **kwargs):
        return None


@pytest.fixture
def client():
    app = create_app()
    fake = FakeToolAwareModel(responses=[AIMessage(content=FAKE_ANSWER) for _ in range(5)])

    def override():
        return build_graph(model=fake, checkpointer=MemorySaver())

    app.dependency_overrides[get_graph] = override
    app.dependency_overrides[get_db_session] = FakeDbSession
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["provider"] in {"anthropic", "openai"}
    assert "calculate" in body["tools"]


def test_chat_requires_internal_key(client):
    res = client.post("/api/v1/chat", json={"message": "hola"})
    assert res.status_code == 401


def test_chat_returns_answer(client):
    res = client.post("/api/v1/chat", json={"message": "hola"}, headers=HEADERS)
    assert res.status_code == 200
    body = res.json()
    assert body["thread_id"]
    assert body["answer"] == FAKE_ANSWER
    assert body["agent"] == "base"


def test_chat_stream_sse(client):
    with client.stream(
        "POST", "/api/v1/chat/stream", json={"message": "hola"}, headers=HEADERS
    ) as res:
        assert res.status_code == 200
        text = "".join(res.iter_text())
        assert "event:" in text
        assert "done" in text


class SafetyDbSession(FakeDbSession):
    def add(self, *args, **kwargs):
        pass


@pytest.fixture
def safety_chat(monkeypatch):
    model = ChunkedSafetyModel(responses=[AIMessage(content="Unused fixture response.")])
    graph = build_graph(model=model, checkpointer=MemorySaver())
    runtime = AsyncMock(
        return_value=ResolvedTurn(graph, "base", None, "fixture", "fixture", 25, "fixture")
    )
    monkeypatch.setattr(get_settings(), "vita_runtime_enabled", True)
    monkeypatch.setattr("app.api.routes.chat._rate_limiters", {})
    monkeypatch.setattr("app.vita.runtime.resolve_chat_turn", runtime)
    monkeypatch.setattr("app.vita.locking.thread_guard", history_service.thread_guard)
    app = create_app()
    app.dependency_overrides[get_graph] = lambda: graph
    app.dependency_overrides[get_db_session] = SafetyDbSession
    with TestClient(app) as test_client:
        yield test_client, model, runtime


def post_safety_chat(client, channel, language, message):
    return client.post(
        "/api/v1/chat/stream" if channel == "stream" else "/api/v1/chat",
        json={
            "message": message,
            "thread_id": "safety-thread",
            "user_id": "synthetic-safety-owner",
            "language": language,
            "supports_vita": True,
        },
        headers=HEADERS,
    )


def chat_fragments(response, channel):
    assert response.status_code == 200
    assert HIDDEN_MARKER not in response.text
    if channel == "json":
        return [response.json()["answer"]]
    assert "event: done\n" in response.text
    frames = sse_payloads(response.text)
    assert not any(frame.get("type") == "error" for frame in frames)
    fragments = [frame["content"] for frame in frames if frame.get("type") == "token"]
    assert frames[-1]["answer"] == "".join(fragments)
    return fragments


@pytest.mark.parametrize("channel", ["json", "stream"])
@pytest.mark.parametrize("case", SAFETY_CASES)
def test_vita_safety_chat_matrix(safety_chat, channel, case):
    client, model, runtime = safety_chat
    if case.previous:
        model.responses = [AIMessage(content=PRIOR_ANSWER[case.language])]
        prior = post_safety_chat(client, channel, case.language, case.previous)
        assert_context_delivery(model, case, chat_fragments(prior, channel))

    prior_calls = len(model.prompts)
    model.responses = [AIMessage(content=case.answer)]
    response = post_safety_chat(client, channel, case.language, case.prompt)
    fragments = chat_fragments(response, channel)
    assert_safe_delivery(case, fragments, model.prompts[prior_calls:])
    assert runtime.await_count == (2 if case.previous else 1)
