"""Escritura del índice privado a partir del catálogo documental del backend."""

import asyncio
from base64 import b64decode
from binascii import Error as Base64Error
from datetime import datetime

from sqlalchemy import Column, DateTime, MetaData, String, Table, Uuid, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.chunker import Chunk, chunk_markdown
from app.rag.embeddings import get_embeddings
from app.rag.extract import DocumentExtractionError, extract_text
from app.rag.pgvector_store import PgVectorStore

MAX_PDF_BYTES = 5 * 1024 * 1024
MAX_PDF_BASE64_CHARS = 4 * ((MAX_PDF_BYTES + 2) // 3)
_CHUNK_CHARS = 1_500
_EMBED_BATCH_SIZE = 16

# Catálogo clínico de solo lectura, fuera del metadata y Alembic de AI.
_DOCUMENTS = Table(
    "documents",
    MetaData(),
    Column("id", Uuid(as_uuid=False)),
    Column("patient_id", Uuid(as_uuid=False)),
    Column("status", String),
    Column("deleted_at", DateTime(timezone=True)),
    schema="app",
)


class PatientDocumentNotFoundError(Exception):
    pass


class PatientDocumentInactiveError(Exception):
    pass


class PatientEmbeddingUnavailableError(Exception):
    pass


async def _check_owner(
    session: AsyncSession,
    *,
    patient_id: str,
    document_id: str,
    active: bool,
    allow_missing: bool = False,
) -> None:
    columns = _DOCUMENTS.c
    document = (
        await session.execute(
            select(columns.patient_id, columns.status, columns.deleted_at).where(
                columns.id == document_id
            )
        )
    ).one_or_none()
    if document is None and allow_missing:
        return
    if document is None or document.patient_id != patient_id:
        raise PatientDocumentNotFoundError
    if active and (document.status != "Ready" or document.deleted_at is not None):
        raise PatientDocumentInactiveError


async def _lock_document(session: AsyncSession, document_id: str) -> None:
    await session.execute(
        select(func.pg_advisory_xact_lock(func.hashtextextended(f"vita-document:{document_id}", 0)))
    )


def _embed_document(
    *, title: str, document_date: datetime, content: str | None, pdf_base64: str | None
) -> tuple[list[Chunk], list[list[float]]]:
    if pdf_base64 is not None:
        try:
            pdf = b64decode(pdf_base64, validate=True)
        except (Base64Error, ValueError):
            raise DocumentExtractionError("PDF base64 inválido") from None
        if len(pdf) > MAX_PDF_BYTES:
            raise DocumentExtractionError("PDF demasiado grande")
        content = extract_text("documento.pdf", pdf)

    # Acota líneas extensas antes de reutilizar el chunker de RAG.
    lines = (content or "").replace("\x00", " ").splitlines()
    normalized = "\n".join(
        line[start : start + _CHUNK_CHARS]
        for line in lines
        for start in range(0, max(len(line), 1), _CHUNK_CHARS)
    )
    chunks = chunk_markdown(normalized, source=title, max_chars=_CHUNK_CHARS)
    if not chunks:
        raise DocumentExtractionError("Documento sin texto extraíble")

    for chunk in chunks:
        chunk.metadata.update(
            title=title,
            source_date=document_date.date().isoformat(),
            document_date=document_date.isoformat(),
        )

    vectors: list[list[float]] = []
    try:
        embeddings = get_embeddings()
        for start in range(0, len(chunks), _EMBED_BATCH_SIZE):
            batch = chunks[start : start + _EMBED_BATCH_SIZE]
            embedded = embeddings.embed_documents([chunk.content for chunk in batch])
            if len(embedded) != len(batch):
                raise ValueError("Cantidad de vectores inesperada")
            vectors.extend(embedded)
    except Exception as exc:
        raise PatientEmbeddingUnavailableError from exc
    return chunks, vectors


async def index_patient_document(
    session: AsyncSession,
    *,
    patient_id: str,
    document_id: str,
    title: str,
    document_date: datetime,
    content: str | None,
    pdf_base64: str | None,
) -> tuple[int, int]:
    async with session.begin():
        await _check_owner(session, patient_id=patient_id, document_id=document_id, active=True)
    chunks, vectors = await asyncio.to_thread(
        _embed_document,
        title=title,
        document_date=document_date,
        content=content,
        pdf_base64=pdf_base64,
    )
    async with session.begin():
        await _lock_document(session, document_id)
        # Un archivo o borrado confirmado mientras se incrustaba cancela el alta.
        await _check_owner(session, patient_id=patient_id, document_id=document_id, active=True)
        store = PgVectorStore(session)
        replaced = await store.delete_patient_document(
            patient_id=patient_id, document_id=document_id
        )
        created = await store.add(chunks, vectors, patient_id=patient_id, document_id=document_id)
    return created, replaced


async def deindex_patient_document(
    session: AsyncSession, *, patient_id: str, document_id: str, allow_missing: bool = False
) -> int:
    async with session.begin():
        await _lock_document(session, document_id)
        await _check_owner(
            session,
            patient_id=patient_id,
            document_id=document_id,
            active=False,
            allow_missing=allow_missing,
        )
        return await PgVectorStore(session).delete_patient_document(
            patient_id=patient_id, document_id=document_id
        )
