"""Redacción estructurada; el motor C# sigue siendo la autoridad nutricional."""

import asyncio
import json
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.errors import ConfigError
from app.llm.factory import get_chat_model
from app.schemas.nutrition_draft import (
    BilingualText,
    NutritionDraftNarrative,
    NutritionDraftRequest,
    NutritionDraftResponse,
)

logger = logging.getLogger(__name__)


def get_nutrition_model() -> BaseChatModel | None:
    """Reutiliza Anthropic; ausencia de configuración habilita el fallback previsto."""
    try:
        # Los modelos Anthropic actuales rechazan temperature: no enviarlo.
        # La salida es solo narrativa; 4096 tokens cubre los cuatro textos
        # bilingües sin disparar la latencia del structured output.
        return get_chat_model(provider="anthropic", max_tokens=4096).model_copy(
            update={"temperature": None}
        )
    except ConfigError:
        return None


def deterministic_draft(payload: NutritionDraftRequest) -> NutritionDraftResponse:
    """Devuelve la selección del motor, sin inferir datos clínicos ausentes."""
    return NutritionDraftResponse(
        narratives=BilingualText(
            es="Propuesta determinista pendiente de revisión individual por nutrición.",
            en="Deterministic proposal pending individual nutrition review.",
        ),
        instructions=BilingualText(
            es="Conservar las porciones calculadas y revisar las alertas antes de aprobar.",
            en="Keep the calculated portions and review the alerts before approval.",
        ),
        menus=[m.model_copy(deep=True) for m in payload.menus],
        source="deterministic_fallback",
    )


def redaction_context(payload: NutritionDraftRequest) -> dict:
    """Contexto mínimo para redactar: metas, nombres de menú y perfil medido.

    No se envían ítems ni gramos: el motor C# ya es la autoridad nutricional y
    su selección se conserva tal cual. Pedirle al LLM que repita los menús
    completos es lo que llevaba la llamada a 30-45 s y rompía el timeout del
    borde (CloudFront corta a los 30 s).
    """
    return {
        "targets": payload.targets.model_dump(),
        "menus": [{"key": menu.key, "name": menu.name} for menu in payload.menus],
        "clinical_profile": (
            payload.clinical_profile.model_dump() if payload.clinical_profile else None
        ),
    }


async def generate_nutrition_draft(
    payload: NutritionDraftRequest, model: BaseChatModel | None
) -> NutritionDraftResponse:
    if model is None:
        return deterministic_draft(payload)
    try:
        runnable = model.with_structured_output(NutritionDraftNarrative.model_json_schema())
        result = await asyncio.wait_for(
            runnable.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "Redacta una narrativa breve y las indicaciones en español e inglés "
                            "para la propuesta nutricional. No repitas menús, alimentos ni gramos: "
                            "el backend conserva la selección del motor y valida las porciones. "
                            "No inventes datos clínicos ni modifiques metas. "
                            "Nunca afirmes que el plan ya fue revisado o aprobado. "
                            "El perfil medido y las reglas aplicadas explican la propuesta; "
                            "respeta su procedencia y faltantes. No deduzcas glicemia ni grasa "
                            "desde IMC/HbA1c ni conviertas una ausencia en un valor normal. "
                            "Devuelve únicamente narratives e instructions."
                        )
                    ),
                    HumanMessage(
                        content=json.dumps(redaction_context(payload), ensure_ascii=False)
                    ),
                ]
            ),
            timeout=15,
        )
        content = NutritionDraftNarrative.model_validate(result)
        return NutritionDraftResponse(
            narratives=content.narratives,
            instructions=content.instructions,
            menus=[menu.model_copy(deep=True) for menu in payload.menus],
            source="llm",
        )
    except Exception:
        # No registrar el payload ni la excepción: podrían contener datos sensibles.
        logger.warning("nutrition_draft_fallback")
        return deterministic_draft(payload)
