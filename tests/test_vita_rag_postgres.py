"""Batería de recuperación ES/EN sobre pgvector asíncrono y datos sintéticos."""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from statistics import median, quantiles
from time import perf_counter
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agents.runtime_registry import AgentRuntimeRegistry
from app.db.models import KnowledgeChunk
from app.graph.graph import build_graph
from app.observability.executions import extract_rag_sources
from app.rag.chunker import Chunk
from app.rag.pgvector_store import PgVectorStore
from app.rag.retriever import PgVectorRetriever
from app.tools.retrieval import make_retrieve_tool
from app.vita import runtime
from tests.fakes import RecordingFakeModel
from tests.test_runtime_registry import AGENT_ID, VERSION_ID, make_session_with_row
from tests.test_vita_history_postgres import isolated_test_url


def entity(name):
    return str(uuid5(NAMESPACE_URL, f"fixture-ola-b/{name}"))


GLOBAL, SPECIALIST, OUTSIDE = (entity(name) for name in ("global", "specialist", "outside"))
PATIENT_A, PATIENT_B = (entity(name) for name in ("patient-a", "patient-b"))
DOC_A, DOC_B, DOC_GLOBAL, DOC_SPECIALIST, DOC_OUTSIDE = (
    entity(name) for name in ("doc-a", "doc-b", "doc-global", "doc-specialist", "doc-outside")
)
RECORDED = datetime(2026, 11, 13, tzinfo=UTC)
PATIENT_DATE = "2026-11-17"
GLOBAL_TITLE = "Guía global sintética"
SPECIALIST_TITLE = "Protocolo del especialista"
A_TITLE = "Informe privado alfa"
B_TITLE = "Informe privado beta"
QUESTIONS = [
    ("Según la guía general, ¿cómo practico una pausa?", "program", None, 0, GLOBAL_TITLE),
    ("What does the general guide say about breathing?", "program", None, 0, GLOBAL_TITLE),
    ("¿Qué indica la guía sobre el descanso?", "program", None, 0, GLOBAL_TITLE),
    ("How does the guide describe a routine?", "program", None, 0, GLOBAL_TITLE),
    (
        "Según el protocolo de psicología, ¿cómo se describe el ejercicio?",
        "program",
        None,
        1,
        SPECIALIST_TITLE,
    ),
    (
        "What does the specialist protocol say about grounding?",
        "program",
        None,
        1,
        SPECIALIST_TITLE,
    ),
    ("¿Qué dice el protocolo de psicología sobre el estrés?", "program", None, 1, SPECIALIST_TITLE),
    ("How does the specialist guide describe a pause?", "program", None, 1, SPECIALIST_TITLE),
    ("¿Qué dice mi informe sobre mi examen?", "patient", PATIENT_A, 2, A_TITLE),
    ("What does the document I uploaded mention?", "patient", PATIENT_A, 2, A_TITLE),
    ("¿Qué dice el examen que subí sobre la medición?", "patient", PATIENT_B, 2, B_TITLE),
    ("What does my uploaded report say about the test?", "patient", PATIENT_B, 2, B_TITLE),
    ("¿Hay una guía general sobre una prueba no cubierta?", "program", None, 3, None),
    ("What does the specialist say on the uncovered topic?", "program", None, 3, None),
    ("¿Qué dice mi expediente sobre un dato ausente?", "patient", PATIENT_A, 3, None),
]


def vector(index, *, secondary=None):
    value = [0.0] * 1536
    value[index] = 1.0
    if secondary is not None:
        value[secondary] = 0.65
    return value


async def _seed(sessions, agents):
    async with sessions.begin() as sql:
        for kb_id in (GLOBAL, SPECIALIST, OUTSIDE):
            await sql.execute(
                text(f'INSERT INTO "{agents}".knowledge_bases VALUES (:id, :status)'),
                {"id": kb_id, "status": "Activo"},
            )
        store = PgVectorStore(sql)
        for doc, kb, title, topic in (
            (DOC_GLOBAL, GLOBAL, GLOBAL_TITLE, 0),
            (DOC_SPECIALIST, SPECIALIST, SPECIALIST_TITLE, 1),
            (DOC_OUTSIDE, OUTSIDE, "Fuente de otro especialista", 1),
        ):
            await sql.execute(
                text(f'INSERT INTO "{agents}".documents VALUES (:id, :kb, :status, :title, :date)'),
                {"id": doc, "kb": kb, "status": "Listo", "title": title, "date": RECORDED},
            )
            await store.add(
                [Chunk(doc, "Extracto sintético del programa.", {"heading": "Programa"})],
                [vector(topic, secondary=0 if kb == SPECIALIST else None)],
                document_id=doc,
                knowledge_base_id=kb,
            )
        for doc, patient, title, embedding in (
            (DOC_A, PATIENT_A, A_TITLE, vector(2, secondary=0)),
            (DOC_B, PATIENT_B, B_TITLE, vector(2)),
        ):
            await store.add(
                [
                    Chunk(
                        doc,
                        "Extracto sintético privado.",
                        {"title": title, "source_date": PATIENT_DATE},
                    )
                ],
                [embedding],
                document_id=doc,
                patient_id=patient,
            )
        # Los resultados irrelevantes de cada dueño se filtran por similitud.
        for patient in (PATIENT_A, PATIENT_B):
            await store.add(
                [Chunk(entity(f"other-{patient}"), "Otro tema privado.", {"title": "Otro tema"})],
                [vector(4)],
                document_id=entity(f"other-{patient}"),
                patient_id=patient,
            )


@pytest.fixture
async def rag(monkeypatch):
    suffix = uuid4().hex
    ai, agents = f"rag_ai_{suffix}", f"rag_agents_{suffix}"
    engine = create_async_engine(
        isolated_test_url(),
        execution_options={"schema_translate_map": {"ai": ai, "agents": agents}},
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as sql:
            await sql.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await sql.execute(text(f'CREATE SCHEMA "{ai}"'))
            await sql.execute(text(f'CREATE SCHEMA "{agents}"'))
            await sql.execute(
                text(f'CREATE TABLE "{agents}".knowledge_bases (id uuid PRIMARY KEY, status text)')
            )
            await sql.execute(
                text(
                    f'CREATE TABLE "{agents}".documents (id uuid PRIMARY KEY, '
                    "knowledge_base_id uuid, status text, file_name text, created_at timestamptz)"
                )
            )
            await sql.run_sync(
                lambda conn: KnowledgeChunk.metadata.create_all(
                    conn, tables=[KnowledgeChunk.__table__]
                )
            )
        await _seed(sessions, agents)
        monkeypatch.setattr("app.tools.retrieval.async_session", sessions)
        yield sessions, agents
    finally:
        async with engine.begin() as sql:
            await sql.execute(text(f'DROP SCHEMA IF EXISTS "{ai}" CASCADE'))
            await sql.execute(text(f'DROP SCHEMA IF EXISTS "{agents}" CASCADE'))
        await engine.dispose()


class SampleEmbeddings:
    async def aembed_query(self, query):
        topics = {message: topic for message, _, _, topic, _ in QUESTIONS}
        return vector(topics[query])


def cfg(patient=None, *, channel="text"):
    return {
        "configurable": {
            "thread_id": "fixture::read",
            "user_id": entity("auth-user"),
            "patient_id": patient,
            "message_channel": channel,
        }
    }


def references(content):
    message = ToolMessage(content=content, name="retrieve_knowledge", tool_call_id="reference-test")
    return extract_rag_sources([message])


async def test_bateria_15_preguntas_es_en_y_latencia(rag, monkeypatch):
    monkeypatch.setattr("app.rag.retriever.get_embeddings", SampleEmbeddings)
    tool = make_retrieve_tool([GLOBAL, SPECIALIST], top_k=1, allow_patient=True)
    measurements, count = [], 0
    for _ in range(3):
        for question, scope, patient, _, expected in QUESTIONS:
            started = perf_counter()
            content = await tool.ainvoke({"query": question, "scope": scope}, config=cfg(patient))
            measurements.append((perf_counter() - started) * 1_000)
            sources = references(content)
            if expected is None:
                assert not sources and "no relevant excerpts" in content, question
            else:
                (source,) = sources
                assert (source["title"], source["origin"]) == (expected, scope), question
                expected_date = PATIENT_DATE if scope == "patient" else RECORDED.date().isoformat()
                assert source["date"] == expected_date and source["date_kind"] == "document"
                assert expected in content and expected_date in content
                assert "Fuente de otro especialista" not in content
                foreign = B_TITLE if patient == PATIENT_A else A_TITLE
                assert foreign not in content
            count += 1
    p50 = median(measurements)
    p95 = quantiles(measurements, n=20, method="inclusive")[-1]
    print(
        f"VITA_RAG_SAMPLE questions={len(QUESTIONS)} checks={count} "
        f"p50_ms={p50:.2f} p95_ms={p95:.2f}"
    )


async def test_invalida_documentos_y_kb_en_la_siguiente_consulta(rag):
    sessions, agents = rag
    async with sessions.begin() as sql:
        store = PgVectorStore(sql)
        assert await store.search(vector(2), 1, patient_id=PATIENT_A)
        assert (
            await store.set_patient_document_status(
                patient_id=PATIENT_A, document_id=DOC_A, status="archived"
            )
            == 1
        )
    async with sessions.begin() as sql:
        store = PgVectorStore(sql)
        assert await store.search(vector(2), 1, patient_id=PATIENT_A) == []
        assert await store.search(vector(2), 1, patient_id=PATIENT_B)
        assert (
            await store.set_patient_document_status(
                patient_id=PATIENT_B, document_id=DOC_B, status="deleted"
            )
            == 1
        )
        assert await store.search(vector(2), 1, patient_id=PATIENT_B) == []
        assert await store.search(vector(0), 1, knowledge_base_ids=[GLOBAL])
        await sql.execute(
            text(f'UPDATE "{agents}".knowledge_bases SET status = :status WHERE id = :kb'),
            {"status": "Inactivo", "kb": GLOBAL},
        )
    async with sessions.begin() as sql:
        store = PgVectorStore(sql)
        assert await store.search(vector(0), 1, knowledge_base_ids=[GLOBAL]) == []
        await sql.execute(
            text(f'UPDATE "{agents}".documents SET status = :status WHERE id = :doc'),
            {"status": "Archivado", "doc": DOC_SPECIALIST},
        )
        assert await store.search(vector(1), 1, knowledge_base_ids=[SPECIALIST]) == []
        assert (
            await store.search(vector(2), 1, knowledge_base_ids=[GLOBAL], patient_id=PATIENT_A)
            == []
        )


@pytest.mark.parametrize("channel", ["text", "voice"])
@pytest.mark.parametrize("specialist", [False, True])
async def test_runtime_privado_responde_con_fuentes_sin_inyectar_programa(
    rag, monkeypatch, channel, specialist
):
    monkeypatch.setattr("app.rag.retriever.get_embeddings", SampleEmbeddings)

    async def catalog(_):
        return (
            [
                {
                    "id": AGENT_ID,
                    "runtimeKey": AGENT_ID,
                    "name": "Especialista sintético",
                    "versionId": VERSION_ID,
                    "versionNumber": 1,
                    "routing": {"channels": ["text", "voice"]},
                }
            ]
            if specialist
            else []
        )

    async def unexpected_program(*_):
        pytest.fail("El runtime no debe usar KBs de programa para esta pregunta privada")

    @asynccontextmanager
    async def unlocked(_):
        yield

    monkeypatch.setattr("app.vita.locking.thread_guard", unlocked)
    monkeypatch.setattr(runtime, "read_catalog", catalog)
    monkeypatch.setattr(runtime, "resolve_knowledge_base_ids", unexpected_program)
    checkpoint = MemorySaver()
    model = RecordingFakeModel(responses=[AIMessage(content="Respuesta sintética con referencia.")])
    monkeypatch.setattr(
        runtime,
        "runtime_registry",
        AgentRuntimeRegistry(
            model_factory=lambda **_: model,
            checkpointer_factory=lambda: checkpoint,
        ),
    )
    monkeypatch.setattr(runtime, "get_chat_model", lambda **_: model)
    question = QUESTIONS[8][0]
    config = cfg(PATIENT_A, channel=channel)
    resolved = await runtime.resolve_turn(
        user_id=config["configurable"]["user_id"],
        text=question,
        session=make_session_with_row({"tools": [], "retrieval_config": {"enabled": False}}),
        base_graph=build_graph(model=model, checkpointer=checkpoint),
        config=config,
        channel=channel,
        explicit=AGENT_ID if specialist else None,
    )
    output = await resolved.graph.ainvoke({"input": question}, config=config)
    assert len(model.prompts) == 1
    assert [(s["title"], s["origin"]) for s in output["data_sources"]] == [(A_TITLE, "patient")]
    assert A_TITLE in str(model.prompts) and B_TITLE not in str(model.prompts)


async def test_paciente_sin_canal_autorizado_no_consulta_indice(rag, monkeypatch):
    def forbidden():
        pytest.fail("No debe crear embeddings antes de verificar el alcance")

    monkeypatch.setattr("app.rag.retriever.get_embeddings", forbidden)
    tool = make_retrieve_tool([GLOBAL], allow_patient=True)
    for config in (cfg(), cfg(PATIENT_A, channel="whatsapp")):
        result = await tool.ainvoke({"scope": "patient", "query": "privado"}, config=config)
        assert "unavailable" in result and not references(result)
    with pytest.raises(ValueError):
        await tool.ainvoke(
            {"scope": "patient", "query": "privado", "patient_id": PATIENT_B}, config=cfg(PATIENT_A)
        )


async def test_embedding_sincrono_no_bloquea_ni_serializa_recuperacion(rag):
    sessions, _ = rag
    first, second, release = threading.Event(), threading.Event(), threading.Event()

    class SyncEmbedding:
        def embed_query(self, query):
            (first if query == "a" else second).set()
            assert release.wait(2), "El event loop no pudo avanzar mientras se calculaba el vector"
            return vector(2)

    async def lookup(patient, query):
        async with sessions() as session:
            return await PgVectorRetriever(SyncEmbedding()).retrieve(
                session, query, patient_id=patient
            )

    tasks = [
        asyncio.create_task(lookup(PATIENT_A, "a")),
        asyncio.create_task(lookup(PATIENT_B, "b")),
    ]
    try:
        async with asyncio.timeout(1):
            while not (first.is_set() and second.is_set()):
                await asyncio.sleep(0.005)
        assert all(not task.done() for task in tasks)
    finally:
        release.set()
        results = await asyncio.gather(*tasks)
    assert [[item.reference()["title"] for item in result] for result in results] == [
        [A_TITLE],
        [B_TITLE],
    ]


async def test_fallo_de_rag_registra_solo_codigo_y_tipo(monkeypatch, caplog):
    def broken():
        raise RuntimeError("contenido privado que no debe aparecer")

    monkeypatch.setattr("app.rag.retriever.get_embeddings", broken)
    with caplog.at_level(logging.WARNING):
        output = await make_retrieve_tool([GLOBAL]).ainvoke({"query": "consulta privada"})
    assert "unavailable" in output and not references(output)
    assert "rag_retrieval_failed scope=program error=RuntimeError" in caplog.text
    assert "contenido privado" not in caplog.text and "consulta privada" not in caplog.text
