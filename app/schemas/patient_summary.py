"""Contrato interno del resumen de paciente (camelCase, espejo del backend .NET).

El backend serializa con `JsonSerializerDefaults.Web` (camelCase) y valida la
respuesta con `UnmappedMemberHandling.Disallow`: cualquier campo extra o una
cita inventada invalida el resumen. El LLM solo redacta; las referencias y la
decisión de suficiencia siguen siendo del backend.
"""

from datetime import date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Section = Literal["program", "tests", "nutrition", "exercise", "podcasts", "measurements", "alerts"]
Ref = str
FactText = str


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, str_strip_whitespace=True)


class PatientSummaryFact(StrictModel):
    ref: Ref = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z][a-zA-Z0-9_.:-]{0,79}$")
    section: Section
    text: FactText = Field(min_length=1, max_length=2000)


class PatientSummaryRequest(StrictModel):
    """Entrada exacta del backend: `startDate`, `endDate`, `language`, `facts`."""

    start_date: date = Field(alias="startDate")
    end_date: date = Field(alias="endDate")
    language: Literal["es", "en"]
    facts: list[PatientSummaryFact] = Field(min_length=1, max_length=60)

    @model_validator(mode="after")
    def validate_interval_and_refs(self) -> Self:
        span = self.end_date.toordinal() - self.start_date.toordinal()
        if span < 0 or span >= 7:
            raise ValueError("El intervalo debe cubrir entre 1 y 7 días")
        refs = [fact.ref for fact in self.facts]
        if len(refs) != len(set(refs)):
            raise ValueError("Las referencias deben ser únicas")
        return self


class PatientSummaryAttention(StrictModel):
    text: FactText = Field(min_length=1, max_length=2000)
    source_refs: list[Ref] = Field(alias="sourceRefs", min_length=1, max_length=10)


class PatientSummaryDraft(StrictModel):
    """Salida del LLM: resumen, referencias citadas y puntos de atención."""

    summary: FactText = Field(min_length=1, max_length=2000)
    source_refs: list[Ref] = Field(alias="sourceRefs", min_length=1, max_length=60)
    attention: list[PatientSummaryAttention] = Field(default_factory=list, max_length=5)


class PatientSummaryResponse(StrictModel):
    """Respuesta exacta que deserializa el backend (sin campos adicionales)."""

    status: Literal["ok", "insufficient"]
    kind: Literal["patient_summary"] = "patient_summary"
    is_estimate: bool = Field(default=True, alias="isEstimate")
    summary: str = ""
    source_refs: list[Ref] = Field(default_factory=list, alias="sourceRefs")
    attention: list[PatientSummaryAttention] = Field(default_factory=list)
    model: str | None = None
    generated_at: datetime = Field(alias="generatedAt")
