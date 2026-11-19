"""Adjuntos efímeros del chat Vita (Ola B, 6.1 y 6.3).

- Validación por firma real (magic bytes), coherencia con el tipo declarado y la
  extensión, tamaño por tipo y estructura básica del PDF. Un rechazo nunca
  almacena el archivo.
- Persistencia efímera en `ai.vita_attachments` con `expires_at`. Toda lectura
  exige el par paciente/usuario autenticado y un TTL vigente; un adjunto ajeno,
  inexistente o expirado es indistinguible (no revela existencia).
- El adjunto nunca toca el expediente (`app.documents`) ni el RAG privado
  (`ai.knowledge_chunks`).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import PurePath
from typing import Literal
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import VitaAttachment

logger = logging.getLogger(__name__)

AttachmentKind = Literal["image", "pdf"]

# Tipos que aceptan los proveedores de visión configurados (Anthropic/OpenAI).
_EXTENSIONS: dict[str, frozenset[str]] = {
    "application/pdf": frozenset({".pdf"}),
    "image/jpeg": frozenset({".jpg", ".jpeg", ".jpe"}),
    "image/png": frozenset({".png"}),
    "image/webp": frozenset({".webp"}),
}
_DECLARED_ALIASES = {"image/jpg": "image/jpeg", "image/pjpeg": "image/jpeg"}
# Tipos declarados que no afirman nada: se decide solo por la firma.
_GENERIC_DECLARED = frozenset({"", "application/octet-stream", "binary/octet-stream"})


class AttachmentRejectedError(Exception):
    """Rechazo comprensible del archivo; `code` es estable para el backend."""

    def __init__(self, code: str, status: int, *, limit: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.status = status
        self.limit = limit


@dataclass(frozen=True)
class ValidatedFile:
    kind: AttachmentKind
    content_type: str
    content: bytes
    page_count: int | None
    sha256: str


def sniff_content_type(content: bytes) -> str | None:
    """MIME real a partir de la firma. None si no es un tipo admitido."""
    if content.startswith(b"%PDF-"):
        return "application/pdf"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return None


def max_bytes_for(content_type: str) -> int:
    settings = get_settings()
    if content_type == "application/pdf":
        return settings.vita_attachment_max_pdf_bytes
    return settings.vita_attachment_max_image_bytes


def max_upload_bytes() -> int:
    """Tope absoluto de lectura antes de conocer el tipo."""
    settings = get_settings()
    return max(settings.vita_attachment_max_pdf_bytes, settings.vita_attachment_max_image_bytes)


def _pdf_page_count(content: bytes) -> int:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
        # Muchos informes cifran solo permisos (contraseña de apertura vacía).
        if reader.is_encrypted and not reader.decrypt(""):
            raise AttachmentRejectedError("attachment_pdf_encrypted", 422)
        return len(reader.pages)
    except AttachmentRejectedError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError, OSError, RecursionError):
        raise AttachmentRejectedError("attachment_unreadable", 422) from None


async def validate_upload(
    content: bytes, *, declared_type: str | None, filename: str | None
) -> ValidatedFile:
    """Valida firma, coherencia, tamaño y estructura. No persiste nada."""
    if not content:
        raise AttachmentRejectedError("attachment_empty", 422)
    real = sniff_content_type(content)
    if real is None:
        raise AttachmentRejectedError("attachment_type_not_allowed", 415)
    declared = (declared_type or "").split(";")[0].strip().lower()
    declared = _DECLARED_ALIASES.get(declared, declared)
    if declared not in _GENERIC_DECLARED and declared != real:
        raise AttachmentRejectedError("attachment_mime_mismatch", 422)
    suffix = PurePath(filename or "").suffix.lower()
    if suffix and suffix not in _EXTENSIONS[real]:
        raise AttachmentRejectedError("attachment_mime_mismatch", 422)
    limit = max_bytes_for(real)
    if len(content) > limit:
        raise AttachmentRejectedError("attachment_too_large", 413, limit=limit)
    page_count = None
    if real == "application/pdf":
        # pypdf es síncrono y CPU-bound: fuera del event loop.
        page_count = await asyncio.to_thread(_pdf_page_count, content)
        if page_count == 0:
            raise AttachmentRejectedError("attachment_unreadable", 422)
        max_pages = get_settings().vita_attachment_max_pdf_pages
        if page_count > max_pages:
            raise AttachmentRejectedError("attachment_too_many_pages", 422, limit=max_pages)
    return ValidatedFile(
        kind="pdf" if real == "application/pdf" else "image",
        content_type=real,
        content=content,
        page_count=page_count,
        sha256=hashlib.sha256(content).hexdigest(),
    )


Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AttachmentStore:
    """Acceso a `ai.vita_attachments` siempre acotado por paciente y usuario."""

    def __init__(self, session: AsyncSession, clock: Clock | None = None) -> None:
        self.session = session
        self.clock = clock or _utcnow

    async def create(
        self,
        file: ValidatedFile,
        *,
        patient_id: UUID,
        user_id: str,
        thread_id: str | None,
    ) -> VitaAttachment:
        now = self.clock()
        row = VitaAttachment(
            patient_id=str(patient_id),
            user_id=user_id,
            thread_id=thread_id,
            kind=file.kind,
            content_type=file.content_type,
            size_bytes=len(file.content),
            page_count=file.page_count,
            sha256=file.sha256,
            content=file.content,
            created_at=now,
            expires_at=now + timedelta(seconds=get_settings().vita_attachment_ttl_seconds),
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def get(
        self,
        attachment_id: UUID,
        *,
        patient_id: UUID,
        user_id: str | None = None,
    ) -> VitaAttachment | None:
        rows = await self.get_many([attachment_id], patient_id=patient_id, user_id=user_id)
        return rows[0] if rows else None

    async def get_many(
        self,
        attachment_ids: Sequence[UUID],
        *,
        patient_id: UUID,
        user_id: str | None = None,
    ) -> list[VitaAttachment]:
        """Solo filas vigentes del paciente (y del usuario, si se indica)."""
        if not attachment_ids:
            return []
        query = select(VitaAttachment).where(
            VitaAttachment.id.in_([str(i) for i in attachment_ids]),
            VitaAttachment.patient_id == str(patient_id),
            VitaAttachment.expires_at > self.clock(),
        )
        if user_id is not None:
            query = query.where(VitaAttachment.user_id == user_id)
        rows = (await self.session.execute(query)).scalars().all()
        order = {str(i): n for n, i in enumerate(attachment_ids)}
        return sorted(rows, key=lambda row: order[str(row.id)])

    async def delete(self, attachment_id: UUID, *, patient_id: UUID) -> int:
        """Borra solo el par autorizado. Devuelve filas eliminadas (0 si ajeno)."""
        result = await self.session.execute(
            delete(VitaAttachment).where(
                VitaAttachment.id == str(attachment_id),
                VitaAttachment.patient_id == str(patient_id),
            )
        )
        return result.rowcount or 0

    async def purge_expired(self) -> int:
        result = await self.session.execute(
            delete(VitaAttachment).where(VitaAttachment.expires_at <= self.clock())
        )
        return result.rowcount or 0


@asynccontextmanager
async def attachment_purge_worker():
    """Purga periódica de adjuntos expirados (idempotente entre instancias)."""
    from app.db.engine import async_session

    async def run() -> None:
        while True:
            try:
                async with asyncio.timeout(30), async_session() as session:
                    purged = await AttachmentStore(session).purge_expired()
                    await session.commit()
                if purged:
                    logger.info("vita_attachments_purged count=%d", purged)
            except Exception as exc:
                logger.error("vita_attachments_purge_failed error=%s", type(exc).__name__)
            await asyncio.sleep(get_settings().vita_attachment_purge_interval)

    task = asyncio.create_task(run(), name="vita-attachments-purge")
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
