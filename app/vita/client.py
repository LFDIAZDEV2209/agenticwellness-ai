"""Única frontera HTTP con la fachada autorizada del backend; nunca lee planes en SQL."""

import asyncio
import json
from typing import Any
from uuid import UUID

import httpx
from pydantic import ValidationError

from app.core.config import get_settings
from app.vita.contracts import (
    INVALID_VITALS_RANGE,
    ContextEnvelope,
    MedicationsRecord,
    ProfileRecord,
    RecordReadResult,
    RecordResource,
    Scope,
    VitalsArguments,
    VitalsRecord,
)

MAX_RESPONSE_BYTES = 64_000
_RECORD_MODELS = {
    "profile": ProfileRecord,
    "vitals": VitalsRecord,
    "medications": MedicationsRecord,
}
_RECORD_UNAVAILABLE = RecordReadResult(
    status="unavailable",
    code="vita_record_unavailable",
    message="Estos datos no están disponibles ahora.",
)


class VitaAccessError(Exception):
    """Fallo cerrado del catálogo/identidad, sin incluir payload ni credenciales."""


async def _get(path: str, user_id: str, **params: str) -> tuple[int, Any]:
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
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
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
    return _RECORD_UNAVAILABLE


async def read_record(
    resource: RecordResource, user_id: str | None, *, dates: VitalsArguments | None = None
) -> RecordReadResult:
    """Consume el DTO interno con la identidad del turno y propaga su cancelación."""
    try:
        owner = UUID(str(user_id))
    except ValueError:
        owner = None
    if not owner or not owner.int:
        return RecordReadResult(
            status="forbidden",
            code="vita_identity_required",
            message="No pude verificar tu sesión para consultar estos datos.",
        )

    try:
        params = dates.model_dump(mode="json", by_alias=True, exclude_none=True) if dates else {}
        status, body = await _get(resource, str(owner), **params)
        if status != 200:
            return _record_failure(status, body, resource)
        data = _RECORD_MODELS[resource].model_validate(body)
    except (httpx.TimeoutException, TimeoutError):
        return RecordReadResult(
            status="unavailable",
            code="tool_timeout",
            message="La consulta tardó demasiado. Puedes intentarlo de nuevo.",
        )
    except (httpx.HTTPError, ValueError, VitaAccessError):
        return _RECORD_UNAVAILABLE

    empty = (isinstance(data, VitalsRecord) and not (data.latest or data.history)) or (
        isinstance(data, MedicationsRecord) and not data.treatments
    )
    return RecordReadResult(status="empty" if empty else "ok", data=data)
