"""Única frontera HTTP con la fachada autorizada del backend; nunca lee planes en SQL."""

import asyncio
import json
from typing import Any
from uuid import UUID

import httpx
from pydantic import ValidationError

from app.core.config import get_settings
from app.vita.contracts import (
    INVALID_NUTRITION_DATE,
    INVALID_VITALS_RANGE,
    AssessmentsRecord,
    ContextEnvelope,
    DocumentContent,
    DocumentListArguments,
    DocumentsRecord,
    DocumentTicket,
    MedicationsRecord,
    NutritionArguments,
    NutritionRecord,
    ProfileRecord,
    RecordReadResult,
    RecordResource,
    Scope,
    VitalsArguments,
    VitalsRecord,
)
from app.vita.document_download import fetch_document_text

MAX_RESPONSE_BYTES = 64_000
_RECORD_MODELS = {
    "profile": ProfileRecord,
    "vitals": VitalsRecord,
    "medications": MedicationsRecord,
    "nutrition": NutritionRecord,
    "assessments": AssessmentsRecord,
    "documents": DocumentsRecord,
}
_LARGE_RECORDS = {"nutrition", "assessments"}
_RECORD_UNAVAILABLE = RecordReadResult(
    status="unavailable",
    code="vita_record_unavailable",
    message="Estos datos no están disponibles ahora.",
)
_RECORD_TIMEOUT = RecordReadResult(
    status="unavailable",
    code="tool_timeout",
    message="La consulta tardó demasiado. Puedes intentarlo de nuevo.",
)
_IDENTITY_REQUIRED = RecordReadResult(
    status="forbidden",
    code="vita_identity_required",
    message="No pude verificar tu sesión para consultar estos datos.",
)
_DOCUMENT_UNAVAILABLE = RecordReadResult(
    status="unavailable",
    code="vita_document_unavailable",
    message="Ese documento no está disponible.",
)
INVALID_DOCUMENT_ARGUMENTS = RecordReadResult(
    status="unavailable",
    code="invalid_document_arguments",
    message="Indica una página válida o elige un documento de tus resultados.",
)


class VitaAccessError(Exception):
    """Fallo cerrado del catálogo/identidad, sin incluir payload ni credenciales."""


async def _get(path: str, user_id: str, **params: str | int) -> tuple[int, Any]:
    settings = get_settings()
    if not user_id or not settings.backend_base_url or not settings.internal_api_key:
        raise VitaAccessError("vita_unavailable")
    async with (
        asyncio.timeout(settings.vita_http_timeout),
        httpx.AsyncClient(
            timeout=settings.vita_http_timeout,
            follow_redirects=False,
        ) as client,
        client.stream(
            "GET",
            f"{settings.backend_base_url.rstrip('/')}/api/v1/internal/vita/{path}",
            params={**params, "userId": user_id},
            headers={"X-Internal-Key": settings.internal_api_key},
        ) as response,
    ):
        body = bytearray()
        max_bytes = 512_000 if path in _LARGE_RECORDS else MAX_RESPONSE_BYTES
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > max_bytes:
                raise VitaAccessError("vita_response_limit")
        return response.status_code, json.loads(body) if body else None


async def read_context(scope: Scope, user_id: str) -> ContextEnvelope:
    """La identidad la aporta el runtime, nunca el esquema de argumentos del LLM."""
    try:
        code, body = await _get("context", user_id, scope=scope)
        if code == 401:
            return ContextEnvelope(status="forbidden")
        statuses = {200: {"ok", "empty"}, 403: {"forbidden"}, 409: {"stale"}, 503: {"unavailable"}}
        if code not in statuses:
            # 404 de una fachada ausente no acredita que el paciente no tenga plan.
            return ContextEnvelope(status="unavailable")
        envelope = ContextEnvelope.model_validate(body)
        if envelope.status not in statuses[code]:
            return ContextEnvelope(status="unavailable")
        if envelope.status != "ok":
            envelope.actions = []
        if envelope.status not in {"ok", "empty"}:
            envelope.data = None
            envelope.sources = []
        return envelope
    except (httpx.HTTPError, TimeoutError, ValueError, ValidationError, VitaAccessError):
        return ContextEnvelope(status="unavailable")


async def read_catalog(user_id: str) -> Any:
    try:
        code, body = await _get("agents", user_id)
        if code != 200:
            raise VitaAccessError("vita_forbidden" if code in (401, 403) else "vita_unavailable")
        return body
    except (httpx.HTTPError, TimeoutError, ValueError) as exc:
        raise VitaAccessError("vita_unavailable") from exc


def _record_failure(status: int, body: object, resource: RecordResource) -> RecordReadResult:
    detail = body.get("detail") if isinstance(body, dict) else None
    code = body.get("code") if isinstance(body, dict) else None
    if status == 403:
        if detail == "VITA_PATIENT_UNAVAILABLE":
            return RecordReadResult(
                status="forbidden",
                code="vita_patient_unavailable",
                message="No hay un perfil activo.",
            )
        if detail == "VITA_MEDICATIONS_FORBIDDEN":
            return RecordReadResult(
                status="forbidden",
                code="vita_medications_forbidden",
                message="El acceso a la medicación está deshabilitado.",
            )
        if code == "CONSENT_REQUIRED":
            return RecordReadResult(
                status="forbidden",
                code="consent_required",
                message="Se necesita el consentimiento de chat para consultar estos datos.",
            )
    if status == 503 and code == "CONSENT_DOCUMENT_UNAVAILABLE":
        return RecordReadResult(
            status="unavailable",
            code="consent_document_unavailable",
            message="No puedo verificar el consentimiento ahora.",
        )
    if status == 404:
        return RecordReadResult(
            status="unavailable",
            code="vita_module_unavailable",
            message="Esta consulta no está disponible.",
        )
    if status == 400 and resource == "vitals":
        return INVALID_VITALS_RANGE
    if status == 400 and resource == "nutrition":
        return INVALID_NUTRITION_DATE
    if status == 400 and resource == "documents":
        return INVALID_DOCUMENT_ARGUMENTS
    return _RECORD_UNAVAILABLE


def _owner_id(user_id: str | None) -> str | None:
    try:
        owner = UUID(str(user_id))
    except ValueError:
        return None
    return str(owner) if owner.int else None


def _read_result(data) -> RecordReadResult:
    if isinstance(data, NutritionRecord) and data.plan is None:
        return RecordReadResult(
            status="ok" if data.history or data.recommendations else "empty",
            data=data,
            message="No hay un plan publicado para esa fecha.",
        )
    if isinstance(data, AssessmentsRecord) and not data.batteries:
        return RecordReadResult(status="empty", data=data, message="No hay baterías registradas.")
    if isinstance(data, DocumentsRecord) and not data.items:
        message = (
            "No hay documentos en esta página." if data.total else "No hay documentos registrados."
        )
        return RecordReadResult(status="empty", data=data, message=message)
    if isinstance(data, MedicationsRecord) and not data.treatments:
        return RecordReadResult(status="empty", data=data, message="No hay tratamientos vigentes.")
    if isinstance(data, VitalsRecord) and not (data.latest or data.history):
        return RecordReadResult(status="empty", data=data)
    return RecordReadResult(status="ok", data=data)


async def read_record(
    resource: RecordResource,
    user_id: str | None,
    *,
    arguments: VitalsArguments | NutritionArguments | DocumentListArguments | None = None,
) -> RecordReadResult:
    """Consume el DTO interno con la identidad del turno y propaga su cancelación."""
    owner = _owner_id(user_id)
    if owner is None:
        return _IDENTITY_REQUIRED

    try:
        params = (
            arguments.model_dump(mode="json", by_alias=True, exclude_none=True) if arguments else {}
        )
        status, body = await _get(resource, owner, **params)
        if status != 200:
            return _record_failure(status, body, resource)
        data = _RECORD_MODELS[resource].model_validate(body)
    except (httpx.TimeoutException, TimeoutError):
        return _RECORD_TIMEOUT
    except (httpx.HTTPError, ValueError, VitaAccessError):
        return _RECORD_UNAVAILABLE

    return _read_result(data)


async def read_document(user_id: str | None, document_id: UUID) -> RecordReadResult:
    owner = _owner_id(user_id)
    if owner is None:
        return _IDENTITY_REQUIRED
    settings = get_settings()
    try:
        async with asyncio.timeout(settings.vita_http_timeout):
            status, body = await _get(f"documents/{document_id}", owner)
            if status == 404:
                return _DOCUMENT_UNAVAILABLE
            if status != 200:
                return _record_failure(status, body, "documents")
            ticket = DocumentTicket.model_validate(body)
            if ticket.document.id != document_id:
                return _DOCUMENT_UNAVAILABLE
            raw_url = ticket.url.get_secret_value()
            extraction, text, truncated = await fetch_document_text(raw_url, settings)
    except (httpx.TimeoutException, TimeoutError):
        return _RECORD_TIMEOUT
    except (httpx.HTTPError, httpx.InvalidURL, ValueError, VitaAccessError, OSError):
        return _DOCUMENT_UNAVAILABLE
    if extraction == "timeout":
        return _RECORD_TIMEOUT
    if extraction == "no_text":
        return RecordReadResult(
            status="unavailable",
            code="vita_document_no_text",
            message="El documento está disponible, pero no pude extraer texto de su contenido.",
        )
    if extraction == "too_large":
        return RecordReadResult(
            status="unavailable",
            code="vita_document_too_large",
            message="El documento supera el tamaño que puedo leer aquí. Puedes abrirlo en la app.",
        )
    if extraction != "ok":
        return _DOCUMENT_UNAVAILABLE
    return RecordReadResult(
        status="ok",
        data=DocumentContent(document=ticket.document, text=text, truncated=truncated),
    )
