"""Contrato HTTP y persistencia de documentos de Vita sobre pgvector desechable."""

import asyncio
from base64 import b64encode
from datetime import UTC, datetime
from threading import Event
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.deps import get_db_session
from app.api.main import create_app
from app.core.config import get_settings
from app.db.models import KnowledgeChunk
from app.rag import patient_ingest
from app.rag.chunker import Chunk
from app.rag.embeddings import HashEmbeddingsProvider
from app.rag.pgvector_store import PgVectorStore
from tests.test_extract import _synthetic_pdf
from tests.test_vita_history_postgres import isolated_test_url

INDEX = "/internal/vita/documents/index"
PATIENT_A, PATIENT_B, DOC_A, DOC_B, KB = (str(uuid4()) for _ in range(5))
DATE = "2026-11-18T12:00:00Z"
CONTENTS = "Marcador sintético alfa de respiración"


def payload(**changes):
    return {
        "patientId": PATIENT_A,
        "documentId": DOC_A,
        "title": "Informe sintético",
        "documentDate": DATE,
        "status": "active",
        "content": CONTENTS,
        "contentType": "text/plain",
        **changes,
    }


def delete_url(document_id=DOC_A, patient_id=PATIENT_A):
    return f"/internal/vita/documents/{document_id}?patientId={patient_id}"


@pytest.fixture
async def index_api(monkeypatch):
    suffix = uuid4().hex
    ai, clinical, agents = (f"index_{scope}_{suffix}" for scope in ("ai", "app", "agents"))
    engine = create_async_engine(
        isolated_test_url(),
        execution_options={"schema_translate_map": {"ai": ai, "app": clinical, "agents": agents}},
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    embeddings = HashEmbeddingsProvider()
    monkeypatch.setattr(patient_ingest, "get_embeddings", lambda: embeddings)

    async def get_test_session():
        async with sessions() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_db_session] = get_test_session
    try:
        async with engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            for schema in (ai, clinical, agents):
                await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.execute(
                text(
                    f'CREATE TABLE "{clinical}".documents (id uuid PRIMARY KEY, patient_id uuid, '
                    "status text NOT NULL, deleted_at timestamptz)"
                )
            )
            await connection.execute(
                text(f'CREATE TABLE "{agents}".knowledge_bases (id uuid PRIMARY KEY, status text)')
            )
            await connection.execute(
                text(
                    f'CREATE TABLE "{agents}".documents (id uuid PRIMARY KEY, '
                    "knowledge_base_id uuid, status text, file_name text, created_at timestamptz)"
                )
            )
            await connection.run_sync(
                lambda conn: KnowledgeChunk.metadata.create_all(
                    conn, tables=[KnowledgeChunk.__table__]
                )
            )
        async with sessions.begin() as sql:
            for document_id, patient_id in ((DOC_A, PATIENT_A), (DOC_B, PATIENT_B)):
                await sql.execute(
                    text(
                        f"INSERT INTO \"{clinical}\".documents VALUES (:id, :owner, 'Ready', NULL)"
                    ),
                    {"id": document_id, "owner": patient_id},
                )
            await sql.execute(
                text(f"INSERT INTO \"{agents}\".knowledge_bases VALUES (:id, 'Activo')"), {"id": KB}
            )
            # Mismo UUID en otro ámbito para detectar borrados no acotados.
            await sql.execute(
                text(
                    f'INSERT INTO "{agents}".documents VALUES '
                    "(:id, :kb, 'Listo', 'Guía sintética', now())"
                ),
                {"id": DOC_A, "kb": KB},
            )
            await PgVectorStore(sql).add(
                [Chunk(DOC_A, "Guía pública del programa", {})],
                [embeddings.embed_query(CONTENTS)],
                document_id=DOC_A,
                knowledge_base_id=KB,
            )
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"X-Internal-Key": get_settings().internal_api_key},
        ) as api:
            yield api, sessions, clinical, embeddings
    finally:
        async with engine.begin() as connection:
            for schema in (ai, clinical, agents):
                await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()


async def _patient_hits(sessions, embeddings, owner, query=CONTENTS):
    async with sessions() as sql:
        return await PgVectorStore(sql).search(embeddings.embed_query(query), 5, patient_id=owner)


async def test_index_reindex_and_recover_without_crossing_patients_or_program(index_api):
    api, sessions, _, embeddings = index_api
    response = await api.post(INDEX, json=payload())
    assert response.status_code == 200, response.json()
    assert response.json() == {
        "documentId": DOC_A,
        "status": "active",
        "chunksCreated": 1,
        "replacedChunks": 0,
    }
    hits = await _patient_hits(sessions, embeddings, PATIENT_A)
    assert [chunk.content for chunk, _ in hits] == [CONTENTS]
    assert hits[0][0].metadata["title"] == "Informe sintético"
    assert hits[0][0].metadata["date"] == "2026-11-18"
    assert hits[0][0].metadata["date_kind"] == "document"
    assert hits[0][0].metadata["document_date"] == "2026-11-18T12:00:00+00:00"
    assert await _patient_hits(sessions, embeddings, PATIENT_B) == []

    content_b = "Marcador sintético beta de respiración"
    other = await api.post(
        INDEX, json=payload(patientId=PATIENT_B, documentId=DOC_B, content=content_b)
    )
    assert other.status_code == 200, other.json()
    assert [c.content for c, _ in await _patient_hits(sessions, embeddings, PATIENT_B)] == [
        content_b
    ]

    updated = "Versión privada actualizada sobre equilibrio"
    for _ in range(2):
        result = await api.post(INDEX, json=payload(content=updated))
        assert result.status_code == 200, result.json()
        assert result.json()["replacedChunks"] == result.json()["chunksCreated"] == 1
    assert [
        c.content for c, _ in await _patient_hits(sessions, embeddings, PATIENT_A, updated)
    ] == [updated]
    async with sessions() as sql:
        rows = (
            (
                await sql.execute(
                    select(KnowledgeChunk).where(KnowledgeChunk.patient_id == PATIENT_A)
                )
            )
            .scalars()
            .all()
        )
        assert [(row.document_id, row.content, row.retrieval_status) for row in rows] == [
            (DOC_A, updated, "active")
        ]
        program = await PgVectorStore(sql).search(
            embeddings.embed_query(CONTENTS), 5, knowledge_base_ids=[KB]
        )
        assert [chunk.content for chunk, _ in program] == ["Guía pública del programa"]


@pytest.mark.parametrize("status", ["archived", "deleted", "DELETE"])
async def test_invalidation_is_immediate_idempotent_and_keeps_program(index_api, status):
    api, sessions, _, embeddings = index_api
    assert (await api.post(INDEX, json=payload())).status_code == 200
    for expected_removals in (1, 0):
        if status == "DELETE":
            result = await api.delete(delete_url())
            assert result.status_code == 204 and not result.content
        else:
            result = await api.post(INDEX, json=payload(status=status, content=None))
            assert result.status_code == 200, result.json()
            assert result.json() == {
                "documentId": DOC_A,
                "status": status,
                "chunksCreated": 0,
                "replacedChunks": expected_removals,
            }
        assert await _patient_hits(sessions, embeddings, PATIENT_A) == []
    async with sessions() as sql:
        assert await PgVectorStore(sql).count_document(DOC_A, KB) == 1


async def test_rejects_another_patients_unindexed_and_indexed_document(index_api, monkeypatch):
    api, sessions, _, embeddings = index_api
    assert (await api.post(INDEX, json=payload())).status_code == 200

    def no_embedding():
        pytest.fail("No debe producir embeddings de un documento ajeno")

    monkeypatch.setattr(patient_ingest, "get_embeddings", no_embedding)
    for changes in (
        {"patientId": PATIENT_B},
        {"documentId": DOC_B},
        {"documentId": str(uuid4())},
        {"patientId": PATIENT_B, "status": "archived", "content": None},
    ):
        result = await api.post(INDEX, json=payload(**changes))
        assert result.status_code == 404
        assert result.json() == {"detail": "patient_document_not_found"}
    foreign = await api.delete(delete_url(patient_id=PATIENT_B))
    assert foreign.status_code == 404
    assert (await api.delete(delete_url(document_id=str(uuid4())))).status_code == 204
    assert [chunk.content for chunk, _ in await _patient_hits(sessions, embeddings, PATIENT_A)] == [
        CONTENTS
    ]


async def test_pdf_uses_existing_extractor(index_api):
    api, sessions, _, embeddings = index_api
    body = payload(
        content=None,
        contentBase64=b64encode(_synthetic_pdf()).decode(),
        contentType="application/pdf",
    )
    result = await api.post(INDEX, json=body)
    assert result.status_code == 200, result.json()
    hits = await _patient_hits(sessions, embeddings, PATIENT_A, "antes despues")
    assert [chunk.content for chunk, _ in hits] == ["antes despues"]


async def test_authentication_validation_and_error_payloads_do_not_echo_content(monkeypatch):
    async def no_database():
        pytest.fail("Una clave inválida no debe abrir el índice")

    app = create_app()
    app.dependency_overrides[get_db_session] = no_database
    secret = get_settings().internal_api_key
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as api:
        for headers in ({}, {"X-Internal-Key": "wrong"}):
            assert (await api.post(INDEX, json=payload(), headers=headers)).status_code == 401
            assert (await api.delete(delete_url(), headers=headers)).status_code == 401
        monkeypatch.setattr(get_settings(), "internal_api_key", "")
        assert (
            await api.post(INDEX, json=payload(), headers={"X-Internal-Key": secret})
        ).status_code == 503


async def test_malformed_payload_returns_sanitized_validation_or_extraction_error(index_api):
    api, _, _, _ = index_api
    cases = [
        (payload(patientId="inválido"), "invalid_patient_document_request"),
        (payload(documentDate="2026-11-18T12:00:00"), "invalid_patient_document_request"),
        (payload(status="disabled"), "invalid_patient_document_request"),
        (payload(contentBase64="private invalid"), "invalid_patient_document_request"),
        (payload(content=None), "invalid_patient_document_request"),
        (payload(contentType="application/pdf"), "invalid_patient_document_request"),
        (
            payload(content=None, contentBase64="private invalid", contentType="application/pdf"),
            "patient_document_not_extractable",
        ),
    ]
    for body, detail in cases:
        response = await api.post(INDEX, json=body)
        assert response.status_code == 422
        assert response.json() == {"detail": detail}
    bad_uuid = await api.delete(delete_url(patient_id="invalid"))
    assert bad_uuid.status_code == 422
    assert bad_uuid.json() == {"detail": "invalid_patient_document_request"}


async def test_failed_embedding_or_write_preserves_committed_chunks(index_api, monkeypatch, caplog):
    api, sessions, _, embeddings = index_api
    assert (await api.post(INDEX, json=payload())).status_code == 200

    class UnavailableEmbeddings:
        def embed_documents(self, texts):
            raise RuntimeError("contenido privado de proveedor simulado")

    class InvalidVectors:
        def embed_documents(self, texts):
            return [[1.0] for _ in texts]

    for provider in (UnavailableEmbeddings(), InvalidVectors()):
        monkeypatch.setattr(patient_ingest, "get_embeddings", lambda provider=provider: provider)
        response = await api.post(INDEX, json=payload(content="nuevo marcador privado sintético"))
        assert response.status_code == 503
        assert response.json() == {"detail": "patient_document_index_unavailable"}
        assert "contenido privado de proveedor simulado" not in caplog.text
        assert "nuevo marcador privado sintético" not in caplog.text
        assert [
            chunk.content for chunk, _ in await _patient_hits(sessions, embeddings, PATIENT_A)
        ] == [CONTENTS]


@pytest.mark.parametrize("archived", [True, False])
async def test_stale_active_retry_cannot_restore_withdrawn_document(
    index_api, monkeypatch, archived
):
    api, sessions, clinical, embeddings = index_api
    assert (await api.post(INDEX, json=payload())).status_code == 200
    started, finish = Event(), Event()

    class SlowEmbeddings:
        def embed_documents(self, texts):
            started.set()
            if not finish.wait(8):
                raise TimeoutError
            return embeddings.embed_documents(texts)

    monkeypatch.setattr(patient_ingest, "get_embeddings", SlowEmbeddings)
    retry = asyncio.create_task(api.post(INDEX, json=payload(content="contenido demorado")))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        async with sessions.begin() as sql:
            await sql.execute(
                text(
                    f'UPDATE "{clinical}".documents '
                    "SET status=:status, deleted_at=:date WHERE id=:id"
                ),
                {
                    "status": "Archived" if archived else "Ready",
                    "date": None if archived else datetime.now(UTC),
                    "id": DOC_A,
                },
            )
        assert (await api.delete(delete_url())).status_code == 204
    finally:
        finish.set()
        response = await retry
    assert response.status_code == 409
    assert response.json() == {"detail": "patient_document_inactive"}
    assert await _patient_hits(sessions, embeddings, PATIENT_A) == []


async def test_concurrent_reindexes_leave_one_complete_set(index_api):
    api, sessions, _, _ = index_api
    response_a, response_b = await asyncio.gather(
        api.post(INDEX, json=payload(content="Contenido A")),
        api.post(INDEX, json=payload(content="Contenido B")),
    )
    assert response_a.status_code == response_b.status_code == 200
    assert sorted(response.json()["replacedChunks"] for response in (response_a, response_b)) == [
        0,
        1,
    ]
    async with sessions() as sql:
        rows = await sql.scalars(
            select(KnowledgeChunk.content).where(KnowledgeChunk.patient_id == PATIENT_A)
        )
        assert rows.one() in {"Contenido A", "Contenido B"}


def test_openapi_describes_the_private_index_contract():
    spec = create_app().openapi()
    schemas = spec["components"]["schemas"]
    body = schemas["PatientDocumentIndexRequest"]
    assert set(body["required"]) == {"patientId", "documentId", "title", "documentDate", "status"}
    assert set(body["properties"]) == set(body["required"]) | {
        "content",
        "contentBase64",
        "contentType",
    }
    delete = spec["paths"]["/internal/vita/documents/{documentId}"]["delete"]
    assert {(param["name"], param["in"]) for param in delete["parameters"]} == {
        ("documentId", "path"),
        ("patientId", "query"),
        ("x-internal-key", "header"),
    }
    assert "content" not in delete["responses"]["204"]
