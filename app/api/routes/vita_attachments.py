"""Adjuntos efímeros del chat Vita (Ola B, KAN-71: 6.1 y 6.3).

Canal interno backend → AI Service (`X-Internal-Key`). El backend autentica al
paciente con su JWT y envía `userId`/`patientId` derivados de esa sesión; el
modelo nunca elige el paciente. Contrato: `docs/vita-attachments.md`.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db_session
from app.api.security import require_internal_key
from app.core.config import get_settings
from app.vita.attachments import (
    AttachmentRejectedError,
    AttachmentStore,
    max_upload_bytes,
    validate_upload,
)

logger = logging.getLogger(__name__)

# Mensajes comprensibles para el paciente (el backend puede mostrarlos tal cual).
_MESSAGES: dict[str, tuple[str, str]] = {
    "attachment_empty": ("El archivo está vacío.", "The file is empty."),
    "attachment_type_not_allowed": (
        "Solo se admiten imágenes JPG, PNG o WEBP y documentos PDF.",
        "Only JPG, PNG or WEBP images and PDF documents are supported.",
    ),
    "attachment_mime_mismatch": (
        "El contenido del archivo no coincide con su tipo o extensión.",
        "The file content does not match its type or extension.",
    ),
    "attachment_too_large": (
        "El archivo supera el tamaño máximo de {limit_mb} MB.",
        "The file exceeds the maximum size of {limit_mb} MB.",
    ),
    "attachment_too_many_pages": (
        "El PDF supera el máximo de {limit} páginas.",
        "The PDF exceeds the maximum of {limit} pages.",
    ),
    "attachment_pdf_encrypted": (
        "El PDF está protegido con contraseña y no se puede leer.",
        "The PDF is password-protected and cannot be read.",
    ),
    "attachment_unreadable": (
        "El archivo está dañado o no se puede abrir.",
        "The file is damaged or cannot be opened.",
    ),
}


def _rejection(exc: AttachmentRejectedError, language: str | None) -> JSONResponse:
    english = (language or "").strip().lower() == "en"
    template = _MESSAGES.get(exc.code, ("Archivo no válido.", "Invalid file."))[int(english)]
    limit = exc.limit or 0
    message = template.format(limit=limit, limit_mb=round(limit / (1024 * 1024), 1))
    return JSONResponse(status_code=exc.status, content={"detail": exc.code, "message": message})


class AttachmentRoute(APIRoute):
    """Errores sin eco del cuerpo recibido (puede contener PHI)."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        handle = super().get_route_handler()

        async def guarded(request: Request) -> Response:
            try:
                return await handle(request)
            except RequestValidationError:
                raise HTTPException(422, "invalid_attachment_request") from None
            except SQLAlchemyError as exc:
                logger.warning("vita_attachment_store_unavailable error=%s", type(exc).__name__)
                raise HTTPException(503, "attachment_store_unavailable") from None

        return guarded


class AttachmentError(BaseModel):
    detail: str
    message: str | None = None


def require_attachments_enabled() -> None:
    if not get_settings().vita_attachments_enabled:
        raise HTTPException(403, "vita_attachments_disabled")


router = APIRouter(
    prefix="/internal/vita/attachments",
    tags=["vita-attachments"],
    dependencies=[Depends(require_internal_key)],
    route_class=AttachmentRoute,
    responses={
        401: {"model": AttachmentError, "description": "Falta una clave interna válida"},
        503: {"model": AttachmentError, "description": "Almacén no disponible"},
    },
)


class AttachmentMetadata(BaseModel):
    attachment_id: UUID = Field(serialization_alias="attachmentId")
    kind: Literal["image", "pdf"]
    content_type: str = Field(serialization_alias="contentType")
    size_bytes: int = Field(serialization_alias="sizeBytes")
    page_count: int | None = Field(serialization_alias="pageCount")
    sha256: str
    thread_id: str | None = Field(serialization_alias="threadId")
    created_at: datetime = Field(serialization_alias="createdAt")
    expires_at: datetime = Field(serialization_alias="expiresAt")

    @classmethod
    def from_row(cls, row) -> AttachmentMetadata:
        return cls(
            attachment_id=row.id,
            kind=row.kind,
            content_type=row.content_type,
            size_bytes=row.size_bytes,
            page_count=row.page_count,
            sha256=row.sha256,
            thread_id=row.thread_id,
            created_at=row.created_at,
            expires_at=row.expires_at,
        )


class PurgeResponse(BaseModel):
    purged: int


# Margen para cabeceras multipart sobre el tope del archivo.
_MULTIPART_OVERHEAD = 64 * 1024


@router.post(
    "",
    status_code=201,
    dependencies=[Depends(require_attachments_enabled)],
    responses={
        403: {"model": AttachmentError, "description": "Flag desactivado"},
        413: {"model": AttachmentError, "description": "Archivo demasiado grande"},
        415: {"model": AttachmentError, "description": "Tipo real no admitido"},
        422: {"model": AttachmentError, "description": "Archivo o solicitud inválidos"},
    },
)
async def upload_attachment(
    request: Request,
    file: Annotated[UploadFile, File()],
    user_id: Annotated[str, Form(alias="userId", min_length=1, max_length=128)],
    patient_id: Annotated[UUID, Form(alias="patientId")],
    thread_id: Annotated[str | None, Form(alias="threadId", max_length=128)] = None,
    language: Annotated[str | None, Form(max_length=8)] = None,
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """Valida y guarda un adjunto efímero. Un rechazo no almacena nada."""
    limit = max_upload_bytes()
    declared_length = request.headers.get("content-length", "")
    if declared_length.isdigit() and int(declared_length) > limit + _MULTIPART_OVERHEAD:
        too_large = AttachmentRejectedError("attachment_too_large", 413, limit=limit)
        return _rejection(too_large, language)
    content = await file.read(limit + 1)
    try:
        validated = await validate_upload(
            content, declared_type=file.content_type, filename=file.filename
        )
    except AttachmentRejectedError as exc:
        logger.info("vita_attachment_rejected code=%s size=%d", exc.code, len(content))
        return _rejection(exc, language)
    row = await AttachmentStore(session).create(
        validated, patient_id=patient_id, user_id=user_id, thread_id=thread_id or None
    )
    metadata = AttachmentMetadata.from_row(row)
    await session.commit()
    logger.info(
        "vita_attachment_stored attachment=%s kind=%s size=%d",
        metadata.attachment_id,
        metadata.kind,
        metadata.size_bytes,
    )
    return JSONResponse(status_code=201, content=metadata.model_dump(mode="json", by_alias=True))


AttachmentId = Annotated[UUID, Path(alias="attachmentId")]
PatientId = Annotated[UUID, Query(alias="patientId")]


@router.get(
    "/{attachmentId}",
    responses={404: {"model": AttachmentError, "description": "Inexistente, ajeno o expirado"}},
)
async def get_attachment(
    attachment_id: AttachmentId,
    patient_id: PatientId,
    session: AsyncSession = Depends(get_db_session),
) -> AttachmentMetadata:
    row = await AttachmentStore(session).get(attachment_id, patient_id=patient_id)
    if row is None:
        raise HTTPException(404, "vita_attachment_not_found")
    return AttachmentMetadata.from_row(row)


@router.get(
    "/{attachmentId}/content",
    response_class=Response,
    responses={404: {"model": AttachmentError, "description": "Inexistente, ajeno o expirado"}},
)
async def get_attachment_content(
    attachment_id: AttachmentId,
    patient_id: PatientId,
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """Binario para que el backend lo archive tras confirmación (6.4)."""
    row = await AttachmentStore(session).get(attachment_id, patient_id=patient_id)
    if row is None:
        raise HTTPException(404, "vita_attachment_not_found")
    return Response(
        content=row.content,
        media_type=row.content_type,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": "attachment",
            "X-Attachment-Sha256": row.sha256,
        },
    )


@router.delete("/{attachmentId}", status_code=204)
async def delete_attachment(
    attachment_id: AttachmentId,
    patient_id: PatientId,
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """Idempotente. Un adjunto ajeno no se borra y la respuesta no lo delata."""
    await AttachmentStore(session).delete(attachment_id, patient_id=patient_id)
    await session.commit()
    return Response(status_code=204)


@router.post("/purge")
async def purge_attachments(session: AsyncSession = Depends(get_db_session)) -> PurgeResponse:
    """Borra todos los adjuntos expirados (también lo hace el worker periódico)."""
    purged = await AttachmentStore(session).purge_expired()
    await session.commit()
    if purged:
        logger.info("vita_attachments_purged count=%d", purged)
    return PurgeResponse(purged=purged)
