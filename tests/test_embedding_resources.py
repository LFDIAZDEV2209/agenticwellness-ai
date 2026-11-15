"""Recursos embeddings locales: ningún test usa red, BD ni credenciales reales."""

import asyncio
import socket
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import openai
import pytest
import tiktoken
import tiktoken.load
from langchain_openai import OpenAIEmbeddings

from app.api import main
from app.core.config import get_settings
from app.rag import embeddings


@pytest.fixture(autouse=True)
async def isolated_resources(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("La preparación/prueba de recursos no debe acceder a la red")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(tiktoken.load, "read_file", no_network)
    monkeypatch.setattr(embeddings, "_openai_resources", OrderedDict())
    settings = get_settings()
    monkeypatch.setattr(settings, "openai_api_key", "synthetic-resource-key")
    monkeypatch.setattr(settings, "openai_embedding_model", "text-embedding-3-small")
    # Encoding real de bytes, pequeño y local: ejercita el fraccionamiento real
    # de LangChain sin descargar vocabularios ni desactivar sus salvaguardas.
    encoding = tiktoken.Encoding(
        name="embedding-resource-test",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={},
    )
    monkeypatch.setattr(tiktoken, "encoding_for_model", Mock(return_value=encoding))
    monkeypatch.setattr(tiktoken, "get_encoding", Mock(return_value=encoding))
    return settings


@pytest.fixture
def sdk(monkeypatch):
    def response(*, input, **kwargs):
        count = 1 if isinstance(input, str) else len(input)
        return {"data": [{"embedding": [1.0, 2.0, 3.0]} for _ in range(count)]}

    sync = SimpleNamespace(create=Mock(side_effect=response))
    async_ = SimpleNamespace(create=AsyncMock(side_effect=response))
    sync_factory = Mock(return_value=SimpleNamespace(embeddings=sync))
    async_factory = Mock(return_value=SimpleNamespace(embeddings=async_))
    monkeypatch.setattr(openai, "OpenAI", sync_factory)
    monkeypatch.setattr(openai, "AsyncOpenAI", async_factory)
    return SimpleNamespace(
        sync=sync, async_=async_, sync_factory=sync_factory, async_factory=async_factory
    )


def test_reuses_resource_and_initializes_both_clients_and_tokenizer_once(sdk):
    first = embeddings.get_embeddings()
    assert embeddings.get_embeddings("openai") is first
    assert embeddings.get_embeddings() is first
    assert first.dimensions == 1536
    assert first._embeddings.check_embedding_ctx_length is True
    assert first._embeddings.tiktoken_enabled is True
    assert first._embeddings.embedding_ctx_length == 8191
    assert first._embeddings.async_client is sdk.async_
    assert sdk.sync_factory.call_count == sdk.async_factory.call_count == 1
    tiktoken.encoding_for_model.assert_called_once_with("text-embedding-3-small")
    sdk.sync.create.assert_not_called()
    sdk.async_.create.assert_not_called()


def test_concurrent_misses_construct_only_one_resource(sdk):
    barrier = Barrier(8)

    def get_resource(_):
        barrier.wait(timeout=5)
        return embeddings.get_embeddings()

    with ThreadPoolExecutor(max_workers=8) as pool:
        resources = list(pool.map(get_resource, range(8)))
    assert all(resource is resources[0] for resource in resources)
    assert sdk.sync_factory.call_count == sdk.async_factory.call_count == 1


def test_model_and_credential_changes_never_reuse_wrong_client(isolated_resources, sdk):
    first = embeddings.get_embeddings()
    isolated_resources.openai_embedding_model = "text-embedding-3-large"
    second = embeddings.get_embeddings()
    isolated_resources.openai_api_key = "synthetic-rotated-key"
    third = embeddings.get_embeddings()
    assert first is not second and second is not third and first is not third
    assert first._embeddings.model == "text-embedding-3-small"
    assert second._embeddings.model == third._embeddings.model == "text-embedding-3-large"
    assert third._embeddings.openai_api_key.get_secret_value() == "synthetic-rotated-key"
    assert second._embeddings.openai_api_key.get_secret_value() == "synthetic-resource-key"
    assert embeddings.get_embeddings() is third
    assert sdk.async_factory.call_count == 3
    # Ni la clave del índice ni la telemetría contienen la credencial original.
    assert "synthetic" not in repr(list(embeddings._openai_resources))


def test_resource_cache_is_bounded_and_evicts_least_recent(isolated_resources, sdk, monkeypatch):
    monkeypatch.setattr(embeddings, "_MAX_OPENAI_RESOURCES", 2)
    first = embeddings.get_embeddings()
    isolated_resources.openai_embedding_model = "second-model"
    second = embeddings.get_embeddings()
    isolated_resources.openai_embedding_model = "text-embedding-3-small"
    assert embeddings.get_embeddings() is first
    isolated_resources.openai_embedding_model = "third-model"
    embeddings.get_embeddings()
    assert len(embeddings._openai_resources) == 2
    isolated_resources.openai_embedding_model = "text-embedding-3-small"
    assert embeddings.get_embeddings() is first
    isolated_resources.openai_embedding_model = "second-model"
    assert embeddings.get_embeddings() is not second
    assert len(embeddings._openai_resources) == 2
    assert sdk.sync_factory.call_count == 4


@pytest.mark.parametrize("provider", ["hash", "dev", "auto"])
def test_hash_provider_is_always_fresh(isolated_resources, provider):
    isolated_resources.openai_api_key = None
    first = embeddings.get_embeddings(provider)
    second = embeddings.get_embeddings(provider)
    assert isinstance(first, embeddings.HashEmbeddingsProvider)
    assert first is not second
    assert not embeddings._openai_resources


def test_explicit_hash_does_not_initialize_openai_with_key(sdk):
    assert isinstance(embeddings.get_embeddings("hash"), embeddings.HashEmbeddingsProvider)
    sdk.sync_factory.assert_not_called()
    sdk.async_factory.assert_not_called()


def test_missing_key_still_rejects_explicit_openai(isolated_resources):
    isolated_resources.openai_api_key = None
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY no configurada"):
        embeddings.get_embeddings("openai")
    assert not embeddings._openai_resources


def test_unknown_model_prepares_same_fallback_as_langchain(sdk):
    tiktoken.encoding_for_model.side_effect = KeyError("synthetic model")
    embeddings.get_embeddings()
    tiktoken.get_encoding.assert_called_once_with("cl100k_base")


def test_failed_initialization_is_not_cached_and_does_not_log_secret(sdk, caplog):
    sdk.async_factory.side_effect = RuntimeError("synthetic initialization failure")
    with pytest.raises(RuntimeError, match="synthetic initialization failure"):
        embeddings.get_embeddings()
    assert not embeddings._openai_resources
    assert "synthetic-resource-key" not in caplog.text
    sdk.async_factory.side_effect = None
    assert embeddings.get_embeddings() is embeddings.get_embeddings()


def test_document_length_splitting_and_mapping_are_preserved(sdk):
    resource = embeddings.get_embeddings()
    resource._embeddings.embedding_ctx_length = 3
    resource._embeddings.chunk_size = 2
    documents = ["abcdefgh", "xy"]
    result = resource.embed_documents(documents)
    assert len(result) == len(documents)
    assert sdk.sync.create.call_count == 2
    chunks = [chunk for call in sdk.sync.create.call_args_list for chunk in call.kwargs["input"]]
    assert [len(chunk) for chunk in chunks] == [3, 3, 2, 2]
    assert result[0] == pytest.approx([1 / 14**0.5, 2 / 14**0.5, 3 / 14**0.5])
    assert result[1] == [1.0, 2.0, 3.0]
    assert resource._embeddings.check_embedding_ctx_length is True


async def test_async_query_preserves_length_safety_and_does_not_cache_results(sdk):
    resource = embeddings.get_embeddings()
    resource._embeddings.embedding_ctx_length = 3
    results = await asyncio.gather(
        *[embeddings.get_embeddings().aembed_query("abcdefgh") for _ in range(6)]
    )
    assert len(results) == 6 and sdk.async_.create.await_count == 6
    assert all(result == pytest.approx(results[0]) for result in results)
    assert results[0] is not results[1]
    assert all(
        [len(chunk) for chunk in call.kwargs["input"]] == [3, 3, 2]
        for call in sdk.async_.create.call_args_list
    )
    sdk.sync.create.assert_not_called()


def test_sync_queries_and_documents_are_not_memoized(sdk):
    resource = embeddings.get_embeddings()
    resource.embed_query("synthetic query")
    embeddings.get_embeddings().embed_query("synthetic query")
    resource.embed_documents(["synthetic document"])
    embeddings.get_embeddings().embed_documents(["synthetic document"])
    assert sdk.sync.create.call_count == 4


async def test_prewarm_initializes_resources_without_requests(sdk):
    await asyncio.gather(*[embeddings.prewarm_embeddings() for _ in range(4)])
    assert sdk.sync_factory.call_count == sdk.async_factory.call_count == 1
    tiktoken.encoding_for_model.assert_called_once()
    sdk.sync.create.assert_not_called()
    sdk.async_.create.assert_not_called()


async def test_real_sdk_clients_can_initialize_without_network():
    await embeddings.prewarm_embeddings()
    resource = embeddings.get_embeddings()
    try:
        assert isinstance(resource._embeddings, OpenAIEmbeddings)
        assert resource._embeddings.client is not None
        assert resource._embeddings.async_client is not None
    finally:
        resource._embeddings.client._client.close()
        await resource._embeddings.async_client._client.close()


async def test_failed_prewarm_stops_startup_without_exposing_sdk_message(monkeypatch, sdk):
    settings = SimpleNamespace(
        environment="production",
        vita_runtime_enabled=True,
        openai_api_key="synthetic-resource-key",
        auto_migrate=False,
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "setup_logging", lambda: None)
    sdk.async_factory.side_effect = RuntimeError(settings.openai_api_key)
    with pytest.raises(RuntimeError, match="No fue posible preparar") as error:
        async with main.lifespan(None):
            pytest.fail("No servir peticiones tras fallar el prewarm")
    diagnostic = "".join(traceback.format_exception(error.value))
    assert settings.openai_api_key not in diagnostic
    assert not embeddings._openai_resources
    sdk.sync.create.assert_not_called()
    sdk.async_.create.assert_not_called()


@pytest.mark.parametrize(
    "environment,enabled,key,expected",
    [
        ("production", True, "synthetic-resource-key", True),
        ("development", True, "synthetic-resource-key", True),
        ("test", True, "synthetic-resource-key", False),
        ("TEST", True, "synthetic-resource-key", False),
        ("production", False, "synthetic-resource-key", False),
        ("production", True, None, False),
    ],
)
async def test_lifespan_prepares_before_serving_only_for_vita(
    monkeypatch, sdk, environment, enabled, key, expected, caplog
):
    settings = SimpleNamespace(
        environment=environment,
        vita_runtime_enabled=enabled,
        openai_api_key=key,
        auto_migrate=False,
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "setup_logging", lambda: None)
    stages = []

    @asynccontextmanager
    async def worker():
        stages.append("worker")
        assert bool(embeddings._openai_resources) is expected
        yield
        stages.append("stopped")

    monkeypatch.setattr("app.history.service.retention_worker", worker)
    async with main.lifespan(None):
        stages.append("serving")
        assert bool(embeddings._openai_resources) is expected
        assert sdk.async_factory.call_count == int(expected)
    assert stages == ["worker", "serving", "stopped"]
    sdk.sync.create.assert_not_called()
    sdk.async_.create.assert_not_called()
    assert "synthetic-resource-key" not in caplog.text
