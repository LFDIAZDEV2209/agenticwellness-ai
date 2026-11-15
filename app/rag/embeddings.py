"""Proveedores de embeddings.

Interfaz `EmbeddingsProvider` + dos implementaciones:
- `OpenAIEmbeddingsProvider`: producción (text-embedding-3-small por defecto).
- `HashEmbeddingsProvider`: desarrollo/pruebas sin API key (determinista).

Para escala real se puede conectar un servicio de embeddings dedicado o pgvector.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections import OrderedDict
from threading import Lock
from typing import Protocol

logger = logging.getLogger(__name__)

# Solo recursos de proceso; nunca consultas, tokens de pacientes ni vectores.
_MAX_OPENAI_RESOURCES = 4
_openai_resources: OrderedDict[tuple[str, bytes], OpenAIEmbeddingsProvider] = OrderedDict()
_resource_lock = Lock()


class EmbeddingsProvider(Protocol):
    """Contrato de un proveedor de embeddings."""

    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbeddingsProvider:
    """Embeddings de OpenAI (langchain-openai). Requiere OPENAI_API_KEY."""

    def __init__(self, model: str | None = None, *, api_key: str | None = None):
        from app.core.config import get_settings

        settings = get_settings()
        credential = api_key if api_key is not None else settings.openai_api_key
        if not credential:
            raise RuntimeError(
                "OPENAI_API_KEY no configurada. Usa HashEmbeddingsProvider para desarrollo."
            )
        from langchain_openai import OpenAIEmbeddings

        self._embeddings = OpenAIEmbeddings(
            model=model or settings.openai_embedding_model,
            api_key=credential,
        )
        # Mismo encoding y fallback que usa LangChain; no tokeniza ningún texto
        # ni desactiva check_embedding_ctx_length para consultas/documentos.
        import tiktoken

        tokenizer_model = self._embeddings.tiktoken_model_name or self._embeddings.model
        try:
            tiktoken.encoding_for_model(tokenizer_model)
        except KeyError:
            tiktoken.get_encoding("cl100k_base")
        self.dimensions: int = 1536

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    async def aembed_query(self, text: str) -> list[float]:
        """El retriever async debe poder cancelar el HTTP del proveedor."""
        return await self._embeddings.aembed_query(text)


class HashEmbeddingsProvider:
    """Embeddings deterministas sin API key — SOLO desarrollo y tests.

    Basados en hashing de tokens. No aptos para producción, pero permiten
    probar el pipeline RAG completo offline. La dimensión por defecto (1536)
    coincide con la columna `vector` de `ai.knowledge_chunks` para poder
    insertar en Postgres en entornos de desarrollo sin OpenAI.
    """

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions
        self._vocab: dict[str, int] = {}
        self._next_index = 0

    def _tokenize(self, text: str) -> list[str]:
        return [t for t in re.split(r"[^a-z0-9áéíóúñü]+", text.lower()) if t]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokenize(text):
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimensions
            vector[index] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    async def aembed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(provider: str = "auto") -> EmbeddingsProvider:
    """Fábrica de embeddings: 'auto' usa OpenAI si hay key, si no el hash provider."""
    from app.core.config import get_settings

    settings = get_settings()
    if provider in {"hash", "dev"} or (provider == "auto" and not settings.openai_api_key):
        logger.warning("Usando HashEmbeddingsProvider (solo desarrollo).")
        return HashEmbeddingsProvider()
    if not settings.openai_api_key:
        # Conserva el error explícito cuando se exige OpenAI sin credencial.
        return OpenAIEmbeddingsProvider()
    model, credential = settings.openai_embedding_model, settings.openai_api_key
    key = (model, hashlib.sha256(credential.encode()).digest())
    # lru_cache solo no evita construcciones duplicadas ante misses concurrentes.
    with _resource_lock:
        resource = _openai_resources.get(key)
        if resource is None:
            resource = OpenAIEmbeddingsProvider(model, api_key=credential)
            _openai_resources[key] = resource
            if len(_openai_resources) > _MAX_OPENAI_RESOURCES:
                # No cerrar clientes que todavía puede usar una petición activa.
                _openai_resources.popitem(last=False)
        _openai_resources.move_to_end(key)
        return resource


async def prewarm_embeddings() -> None:
    """Prepara cliente sync/async y encoding, sin llamadas de embeddings ni texto."""
    try:
        await asyncio.to_thread(get_embeddings, "openai")
    except Exception:
        # El traceback de startup no debe serializar errores del SDK/config con
        # credenciales. No servir Vita si sus recursos no quedaron preparados.
        raise RuntimeError("No fue posible preparar los recursos embeddings de Vita") from None
