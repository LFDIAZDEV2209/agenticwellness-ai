"""Recuperación de fuentes de programa y del expediente autorizado."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from uuid import UUID

from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from pydantic import BaseModel, ConfigDict, Field

from app.db.engine import async_session
from app.rag.pgvector_store import RagScope
from app.rag.retriever import PgVectorRetriever

logger = logging.getLogger(__name__)

RETRIEVE_TOOL_NAME = "retrieve_knowledge"
_NO_RESULTS = (
    "DOCUMENTATION STATUS: no relevant excerpts were retrieved from the authorized "
    "active sources in this scope. Say that you did not find supporting sources in the "
    "user's language. Do not invent references or imply that a document was read."
)
_UNAVAILABLE = (
    "DOCUMENTATION STATUS: the requested scope is unavailable. Explain the limitation "
    "in the user's language without claiming that a document was read."
)
RETRIEVE_TOOL_DESCRIPTION = (
    "Busca extractos con fuentes autorizadas / Search authorized excerpts. "
    "Usa program para guías, protocolos y KBs activas del agente; usa patient solo para "
    "documentos propios del expediente. Cada referencia lleva título, origen y fecha."
)


class RetrieveArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=8_000)
    scope: RagScope = "program"


def authorized_patient_id(configurable: Mapping) -> str | None:
    """El ID proviene del backend (JWT o vínculo WhatsApp firmado) o de la sesión de voz."""
    if not configurable.get("user_id"):
        return None
    try:
        patient = UUID(str(configurable.get("patient_id")))
    except ValueError:
        return None
    return str(patient) if patient.int else None


def make_retrieve_tool(
    knowledge_base_ids: list[str] | None = None, top_k: int = 5, *, allow_patient: bool = False
) -> StructuredTool:
    kb_ids = list(knowledge_base_ids or [])

    async def retrieve_knowledge(query: str, scope: RagScope = "program") -> str:
        if scope == "patient":
            configurable = get_config().get("configurable", {})
            patient_id = authorized_patient_id(configurable) if allow_patient else None
            if patient_id is None:
                return _UNAVAILABLE
            filters = {"patient_id": patient_id}
        else:
            if not kb_ids:
                return _NO_RESULTS
            filters = {"knowledge_base_ids": kb_ids}
        try:
            retriever = PgVectorRetriever()
            async with async_session() as session:
                results = await retriever.retrieve(session, query, top_k=top_k, **filters)
        except Exception as exc:
            logger.warning("rag_retrieval_failed scope=%s error=%s", scope, type(exc).__name__)
            return _UNAVAILABLE

        parts, sources = [], []
        budget = 3_000
        for item in results:
            block = retriever.format_context([item], max_chars=budget)
            if block:
                parts.append(block)
                sources.append({**item.reference(), "origin": scope})
                budget -= len(block) + 2
        if not sources:
            return _NO_RESULTS
        context = "\n\n".join(parts)
        return f"{context}\n\n[FUENTES: {json.dumps(sources, ensure_ascii=False)}]"

    return StructuredTool.from_function(
        coroutine=retrieve_knowledge,
        name=RETRIEVE_TOOL_NAME,
        description=RETRIEVE_TOOL_DESCRIPTION,
        args_schema=RetrieveArguments,
    )
