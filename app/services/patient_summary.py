"""Resumen clínico breve con citas verificables; el backend conserva la decisión."""

import asyncio
import json
import logging
import re
from datetime import UTC, datetime

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.errors import ConfigError
from app.llm.factory import get_chat_model
from app.schemas.patient_summary import (
    PatientSummaryDraft,
    PatientSummaryRequest,
    PatientSummaryResponse,
)

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[([^\[\]]+)\]")
_UNSAFE = re.compile(r"https?://|<[^>]+>", re.IGNORECASE)


def get_patient_summary_model() -> BaseChatModel | None:
    """Reutiliza Anthropic; sin configuración se sirve el resumen determinista."""
    try:
        # Los modelos Anthropic actuales rechazan temperature: no enviarlo.
        return get_chat_model(provider="anthropic", max_tokens=2048).model_copy(
            update={"temperature": None}
        )
    except ConfigError:
        return None


def _citations(text: str) -> set[str]:
    return set(_CITATION.findall(text))


def _safe(text: str) -> bool:
    return not _UNSAFE.search(text)


def _insufficient() -> PatientSummaryResponse:
    return PatientSummaryResponse(status="insufficient", generated_at=datetime.now(UTC))


def deterministic_summary(payload: PatientSummaryRequest) -> PatientSummaryResponse:
    """Resumen de respaldo con las mismas citas, sin inventar contenido clínico.

    Se usa si el proveedor no está configurado o devuelve una respuesta que no
    cumple el contrato. No añade interpretación: solo enumera hechos ya
    registrados y conserva la referencia de cada uno.
    """
    prefix = (
        "Resumen automático de los últimos 7 días. "
        if payload.language == "es"
        else "Automated summary of the last 7 days. "
    )
    parts: list[str] = []
    refs: list[str] = []
    for fact in payload.facts:
        candidate = f"[{fact.ref}] {fact.text.strip()}"
        if len(prefix) + len(" ".join([*parts, candidate])) > 1950:
            break
        parts.append(candidate)
        refs.append(fact.ref)
    if not parts:
        return _insufficient()
    return PatientSummaryResponse(
        status="ok",
        summary=prefix + " ".join(parts),
        source_refs=refs,
        attention=[],
        model=None,
        generated_at=datetime.now(UTC),
    )


def _valid(draft: PatientSummaryDraft, allowed: set[str]) -> bool:
    """Replica la frontera de confianza del backend antes de responder."""
    if not draft.summary or len(draft.summary) > 2000 or not _safe(draft.summary):
        return False
    refs = set(draft.source_refs)
    if not refs or not refs.issubset(allowed) or _citations(draft.summary) != refs:
        return False
    for attention in draft.attention:
        if not attention.text or len(attention.text) > 2000 or not _safe(attention.text):
            return False
        cited = _citations(attention.text)
        attention_refs = set(attention.source_refs)
        if not attention_refs or not attention_refs.issubset(allowed):
            return False
        if not cited.issubset(attention_refs):
            return False
    return True


async def generate_patient_summary(
    payload: PatientSummaryRequest, model: BaseChatModel | None
) -> PatientSummaryResponse:
    if model is None:
        return deterministic_summary(payload)
    allowed = {fact.ref for fact in payload.facts}
    system = (
        "Eres un asistente clínico de Agentic Wellness IA. Redacta un resumen breve "
        f"(máximo 900 caracteres) en {'español' if payload.language == 'es' else 'inglés'} "
        "usando SOLO los hechos entregados. Cada afirmación debe citar la referencia "
        "del hecho entre corchetes, por ejemplo [program.current]. source_refs debe "
        "contener exactamente las referencias citadas en summary. attention: hasta 3 "
        "puntos que requieran seguimiento, cada uno con sus referencias. No inventes "
        "datos, no des diagnósticos, no afirmes normalidad, no incluyas URLs ni HTML. "
        "No incluyas datos identificatorios."
    )
    messages = [
        SystemMessage(content=system),
        HumanMessage(
            content=json.dumps(
                {
                    "language": payload.language,
                    "facts": [fact.model_dump() for fact in payload.facts],
                },
                ensure_ascii=False,
            )
        ),
    ]
    try:
        runnable = model.with_structured_output(PatientSummaryDraft.model_json_schema())
        for _ in range(2):
            result = await asyncio.wait_for(runnable.ainvoke(messages), timeout=12)
            try:
                draft = PatientSummaryDraft.model_validate(result)
            except Exception:
                continue
            if not _valid(draft, allowed):
                continue
            model_name = getattr(model, "model", None) or getattr(model, "model_name", None)
            return PatientSummaryResponse(
                status="ok",
                summary=draft.summary,
                source_refs=draft.source_refs,
                attention=draft.attention,
                model=model_name if isinstance(model_name, str) and model_name else None,
                generated_at=datetime.now(UTC),
            )
        return deterministic_summary(payload)
    except Exception:
        # No registrar el payload ni la excepción: podrían contener datos sensibles.
        logger.warning("patient_summary_fallback")
        return deterministic_summary(payload)
