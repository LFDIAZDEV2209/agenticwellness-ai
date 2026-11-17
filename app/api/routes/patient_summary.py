"""Endpoint interno del resumen de paciente (contrato camelCase del backend)."""

from fastapi import APIRouter, Depends
from langchain_core.language_models.chat_models import BaseChatModel

from app.api.security import require_internal_key
from app.schemas.patient_summary import PatientSummaryRequest, PatientSummaryResponse
from app.services.patient_summary import generate_patient_summary, get_patient_summary_model

router = APIRouter(
    prefix="/internal/program",
    tags=["internal"],
    dependencies=[Depends(require_internal_key)],
)


@router.post(
    "/patient-summary",
    response_model=PatientSummaryResponse,
    response_model_by_alias=True,
)
async def patient_summary(
    payload: PatientSummaryRequest,
    model: BaseChatModel | None = Depends(get_patient_summary_model),
) -> PatientSummaryResponse:
    """Redacta el resumen; el backend decide suficiencia, consentimiento y alcance."""
    return await generate_patient_summary(payload, model)
