"""Pgvector con alcances mutuamente excluyentes de programa y paciente."""

from __future__ import annotations

from datetime import date
from typing import Literal

from sqlalchemy import (
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    Uuid,
    cast,
    delete,
    func,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import KnowledgeChunk
from app.rag.chunker import Chunk

RagScope = Literal["program", "patient"]
RetrievalStatus = Literal["active", "archived", "deleted", "disabled"]
MIN_COSINE_SCORE = 0.20
_CATALOG_READ_ONLY = MetaData()
_PROGRAM_DOCUMENTS = Table(
    "documents",
    _CATALOG_READ_ONLY,
    Column("id", Uuid(as_uuid=False)),
    Column("knowledge_base_id", Uuid(as_uuid=False)),
    Column("status", String),
    Column("file_name", String),
    Column("created_at", DateTime(timezone=True)),
    schema="agents",
).alias("d")
_PROGRAM_BASES = Table(
    "knowledge_bases",
    _CATALOG_READ_ONLY,
    Column("id", Uuid(as_uuid=False)),
    Column("status", String),
    schema="agents",
).alias("kb")


def _reference_metadata(row, origin: RagScope) -> dict:
    metadata = dict(row.extra or {})
    if origin == "program" and row.document_created_at is not None:
        source_date = row.document_created_at.date()
        metadata.setdefault("title", row.document_title)
        date_kind = "document"
    else:
        try:
            source_date = date.fromisoformat(metadata.get("source_date", ""))
            date_kind = "document"
        except (ValueError, TypeError):
            source_date, date_kind = row.created_at.date(), "indexed"
    metadata.update(origin=origin, date=source_date.isoformat(), date_kind=date_kind)
    return metadata


class PgVectorStore:
    """Almacén de chunks sobre pgvector con operaciones asíncronas.

    No gestiona sesiones: recibe una `AsyncSession` en cada operación para
    poder participar en transacciones del llamador (endpoints, nodos del grafo).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        *,
        document_id: str,
        knowledge_base_id: str | None = None,
        patient_id: str | None = None,
    ) -> int:
        """Inserta los chunks con sus embeddings.

        Args:
            chunks: chunks ya generados por el chunker.
            vectors: embeddings por chunk (mismo orden).
            document_id: ID del documento de origen en su ámbito.
            knowledge_base_id: KB del programa.
            patient_id: dueño canónico para el futuro indexador de expediente.

        Returns:
            Cantidad de chunks insertados.
        """
        if bool(knowledge_base_id) == bool(patient_id):
            raise ValueError("Se requiere exactamente un alcance de indexación")
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._session.add(
                KnowledgeChunk(
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id,
                    patient_id=patient_id,
                    content=chunk.content,
                    embedding=vector,
                    extra=chunk.metadata,
                    chunk_index=int(chunk.metadata.get("position", 0)),
                )
            )
        await self._session.flush()
        return len(chunks)

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        *,
        knowledge_base_ids: list[str] | None = None,
        document_id: str | None = None,
        patient_id: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Búsqueda semántica por similitud coseno (1 - distance = score).

        Args:
            query_vector: embedding de la consulta.
            top_k: máximo de resultados.
            knowledge_base_ids: si se indican, filtra SOLO esas KBs (aislamiento
                entre agentes: un agente nunca recupera chunks de KBs ajenas).
            document_id: filtra un documento concreto (útil para "dónde está esto").

        Returns:
            Lista de (Chunk, score) ordenada por relevancia descendente.
        """
        if bool(knowledge_base_ids) == bool(patient_id):
            return []
        distance = KnowledgeChunk.embedding.cosine_distance(query_vector)
        stmt = select(
            KnowledgeChunk.id,
            KnowledgeChunk.content,
            KnowledgeChunk.extra,
            KnowledgeChunk.created_at,
            distance.label("distance"),
        ).where(KnowledgeChunk.retrieval_status == "active", distance <= 1 - MIN_COSINE_SCORE)
        origin: RagScope = "patient" if patient_id else "program"
        if patient_id:
            # El índice HNSW de programa no participa. El índice parcial de
            # paciente delimita candidatos antes del ranking exacto y LIMIT.
            stmt = stmt.where(KnowledgeChunk.patient_id == patient_id)
        else:
            # Reevaluar el catálogo en cada búsqueda, incluso con grafo cacheado.
            doc, kb = _PROGRAM_DOCUMENTS.c, _PROGRAM_BASES.c
            stmt = (
                stmt.join(_PROGRAM_DOCUMENTS, cast(doc.id, String) == KnowledgeChunk.document_id)
                .join(_PROGRAM_BASES, kb.id == doc.knowledge_base_id)
                .add_columns(
                    doc.file_name.label("document_title"),
                    doc.created_at.label("document_created_at"),
                )
                .where(
                    KnowledgeChunk.patient_id.is_(None),
                    KnowledgeChunk.knowledge_base_id.in_(knowledge_base_ids),
                    KnowledgeChunk.knowledge_base_id == cast(kb.id, String),
                    doc.status == "Listo",
                    kb.status == "Activo",
                )
            )
        if document_id:
            stmt = stmt.where(KnowledgeChunk.document_id == document_id)

        rows = (await self._session.execute(stmt.order_by(distance).limit(top_k))).all()
        return [
            (
                Chunk(
                    id=str(row.id), content=row.content, metadata=_reference_metadata(row, origin)
                ),
                1 - row.distance,
            )
            for row in rows
        ]

    async def set_patient_document_status(
        self, *, patient_id: str, document_id: str, status: RetrievalStatus
    ) -> int:
        """Punto de entrada para invalidaciones 5.2 en la transacción del llamador."""
        result = await self._session.execute(
            update(KnowledgeChunk)
            .where(
                KnowledgeChunk.patient_id == patient_id,
                KnowledgeChunk.document_id == document_id,
            )
            .values(retrieval_status=status)
        )
        return result.rowcount or 0

    async def delete_document(self, document_id: str) -> int:
        """Elimina todos los chunks de un documento (re-ingest o borrado)."""
        result = await self._session.execute(
            delete(KnowledgeChunk).where(
                KnowledgeChunk.document_id == document_id, KnowledgeChunk.patient_id.is_(None)
            )
        )
        return result.rowcount or 0

    async def count_document(self, document_id: str, knowledge_base_id: str) -> int:
        """Cuenta solo los chunks retenidos del par documento/KB indicado."""
        stmt = select(func.count(KnowledgeChunk.id)).where(
            KnowledgeChunk.document_id == document_id,
            KnowledgeChunk.knowledge_base_id == knowledge_base_id,
        )
        return int((await self._session.execute(stmt)).scalar_one())

    async def count(self, knowledge_base_id: str | None = None) -> int:
        """Total de chunks (opcionalmente filtrado por KB)."""
        stmt = select(func.count(KnowledgeChunk.id))
        if knowledge_base_id:
            stmt = stmt.where(KnowledgeChunk.knowledge_base_id == knowledge_base_id)
        return int((await self._session.execute(stmt)).scalar_one())
