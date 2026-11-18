"""Indexación interna de documentos del expediente para Vita."""

import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal, Self
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db_session
from app.api.security import require_internal_key
from app.rag.extract import MAX_DOCUMENT_CHARS, DocumentExtractionError
from app.rag.patient_ingest import (
    MAX_PDF_BASE64_CHARS,
    PatientDocumentInactiveError,
    PatientDocumentNotFoundError,
    PatientEmbeddingUnavailableError,
    deindex_patient_document,
    index_patient_document,
)

logger = logging.getLogger(__name__)
DocumentStatus = Literal["active", "archived", "deleted"]


class PatientDocumentRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        handle = super().get_route_handler()

        async def guarded(request: Request) -> Response:
            try:
                return await handle(request)
            except RequestValidationError:
                # La respuesta por defecto puede repetir el texto o el PDF recibido.
                raise HTTPException(422, "invalid_patient_document_request") from None
            except DocumentExtractionError:
                raise HTTPException(422, "patient_document_not_extractable") from None
            except PatientDocumentNotFoundError:
                raise HTTPException(404, "patient_document_not_found") from None
            except PatientDocumentInactiveError:
                raise HTTPException(409, "patient_document_inactive") from None
            except (PatientEmbeddingUnavailableError, SQLAlchemyError) as exc:
                logger.warning("patient_document_index_unavailable error=%s", type(exc).__name__)
                raise HTTPException(503, "patient_document_index_unavailable") from None

        return guarded


class PatientDocumentError(BaseModel):
    detail: str


router = APIRouter(
    prefix="/internal/vita/documents",
    tags=["vita-documents"],
    dependencies=[Depends(require_internal_key)],
    route_class=PatientDocumentRoute,
    responses={
        401: {"model": PatientDocumentError, "description": "Falta una clave interna válida"},
        404: {"model": PatientDocumentError, "description": "Documento inexistente o ajeno"},
        422: {"model": PatientDocumentError, "description": "Solicitud inválida o sin texto"},
        503: {"model": PatientDocumentError, "description": "Servicio temporalmente no disponible"},
    },
)


class PatientDocumentIndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    patient_id: UUID = Field(alias="patientId")
    document_id: UUID = Field(alias="documentId")
    title: str = Field(min_length=1, max_length=200, pattern=r"^[^\x00]+$")
    document_date: AwareDatetime = Field(alias="documentDate")
    status: DocumentStatus
    content: str | None = Field(default=None, min_length=1, max_length=MAX_DOCUMENT_CHARS)
    pdf_base64: str | None = Field(
        default=None, alias="contentBase64", min_length=1, max_length=MAX_PDF_BASE64_CHARS
    )
    content_type: Literal["text/plain", "application/pdf"] | None = Field(
        default=None, alias="contentType"
    )

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        if self.status != "active":
            return self
        if (self.content is None) == (self.pdf_base64 is None):
            raise ValueError("Se requiere exactamente un formato de contenido")
        expected = "application/pdf" if self.pdf_base64 is not None else "text/plain"
        if self.content_type is not None and self.content_type != expected:
            raise ValueError("El tipo no coincide con el formato de contenido")
        return self


class PatientDocumentIndexResponse(BaseModel):
    document_id: UUID = Field(serialization_alias="documentId")
    status: DocumentStatus
    chunks_created: int = Field(serialization_alias="chunksCreated")
    replaced_chunks: int = Field(serialization_alias="replacedChunks")


@router.post(
    "/index",
    responses={409: {"model": PatientDocumentError, "description": "Documento clínico inactivo"}},
)
async def index_document(
    payload: PatientDocumentIndexRequest, session: AsyncSession = Depends(get_db_session)
) -> PatientDocumentIndexResponse:
    """Reemplaza los fragmentos privados o los retira si el evento indica baja."""
    document_id, patient_id = str(payload.document_id), str(payload.patient_id)
    if payload.status == "active":
        created, replaced = await index_patient_document(
            session,
            patient_id=patient_id,
            document_id=document_id,
            title=payload.title,
            document_date=payload.document_date,
            content=payload.content,
            pdf_base64=payload.pdf_base64,
        )
    else:
        created = 0
        replaced = await deindex_patient_document(
            session, patient_id=patient_id, document_id=document_id
        )
    return PatientDocumentIndexResponse(
        document_id=payload.document_id,
        status=payload.status,
        chunks_created=created,
        replaced_chunks=replaced,
    )


@router.delete("/{documentId}", status_code=204)
async def delete_document(
    document_id: Annotated[UUID, Path(alias="documentId")],
    patient_id: Annotated[UUID, Query(alias="patientId")],
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """Retira los fragmentos del par autorizado de forma idempotente."""
    await deindex_patient_document(
        session, patient_id=str(patient_id), document_id=str(document_id), allow_missing=True
    )
    return Response(status_code=204)
