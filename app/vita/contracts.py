"""Contrato BACK v1: lecturas vigentes y navegación acotada, sin escrituras."""

import re
from datetime import date, datetime
from typing import Annotated, Any, Literal, Self
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_camel

Scope = Literal["day", "nutrition", "assessments", "podcast", "exercise", "progress"]


class AppAction(BaseModel):
    model_config = ConfigDict(extra="forbid", alias_generator=to_camel, serialize_by_alias=True)

    type: Literal["navigate"]
    destination: Literal["program", "nutrition", "assessments", "podcast", "exercise", "progress"]
    resource_id: str | None = None
    local_date: date
    resource_version: str = Field(min_length=1)


class ContextEnvelope(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, serialize_by_alias=True)
    status: Literal["ok", "empty", "unavailable", "forbidden", "stale"]
    as_of: datetime | None = None
    local_date: date | None = None
    resource_version: str | None = None
    data: Any = None
    actions: list[AppAction] = Field(default_factory=list)
    sources: list[dict[str, Any] | str] = Field(default_factory=list)


class EmptyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VitalsArguments(EmptyArguments):
    from_date: date | None = Field(
        default=None, alias="from", description="Fecha local inicial inclusiva (YYYY-MM-DD)."
    )
    to: date | None = Field(default=None, description="Fecha local final inclusiva (YYYY-MM-DD).")

    @field_validator("from_date", "to", mode="before")
    @classmethod
    def local_calendar_date(cls, value: object) -> object:
        if value is None or (
            isinstance(value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value)
        ):
            return value
        raise ValueError("Usa fechas locales en formato YYYY-MM-DD")

    @model_validator(mode="after")
    def inclusive_range(self) -> Self:
        if (
            self.from_date is not None
            and self.to is not None
            and not 0 <= (self.to - self.from_date).days < 31
        ):
            raise ValueError("El intervalo debe cubrir entre 1 y 31 días inclusivos")
        return self


RecordResource = Literal["profile", "vitals", "medications"]
MeasurementSource = Literal["device", "patient", "professional", "lab"]
JsonNumber = StrictInt | Annotated[float, Field(strict=True, allow_inf_nan=False)]


class RecordDto(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, serialize_by_alias=True)


class Measurement(RecordDto):
    metric_code: str
    value: JsonNumber
    unit_code: str
    unit_symbol: str
    observed_at: AwareDatetime


class AnthropometryPoint(Measurement):
    source: MeasurementSource | None


class VitalPoint(Measurement):
    source: MeasurementSource


class PatientProfile(RecordDto):
    first_name: str
    age_years: StrictInt | None
    gender: str | None
    time_zone: str


class NutritionGoals(RecordDto):
    source: Literal["nutrition_prescription", "nutrition_plan"]
    target_condition: str | None
    daily_calorie_target: JsonNumber | None
    daily_protein_target_g: JsonNumber | None
    daily_carbs_target_g: JsonNumber | None
    daily_fat_target_g: JsonNumber | None
    daily_fiber_target_g: JsonNumber | None


class Anthropometry(RecordDto):
    weight: AnthropometryPoint | None
    height: AnthropometryPoint | None
    bmi: AnthropometryPoint | None
    has_more: bool
    history: list[AnthropometryPoint] = Field(max_length=100)


class ProfileRecord(RecordDto):
    as_of: AwareDatetime
    profile: PatientProfile
    goals: NutritionGoals | None
    allergies: list[str]
    restrictions: list[str]
    anthropometry: Anthropometry


class VitalsRecord(RecordDto):
    as_of: AwareDatetime
    time_zone: str
    from_date: date = Field(alias="from")
    to: date
    latest: list[VitalPoint]
    has_more: bool
    history: list[VitalPoint] = Field(max_length=200)


class MedicationDose(RecordDto):
    scheduled_at: AwareDatetime
    status: Literal["pending", "taken", "omitted", "cancelled"]
    occurred_at: AwareDatetime | None


class Treatment(RecordDto):
    id: UUID
    name: str
    product_type: Literal["medication", "vitamin", "supplement"]
    dosage: JsonNumber
    unit: str
    route: str
    time_zone: str
    times: list[Annotated[str, Field(pattern=r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]$")]]
    start_date: date
    end_date: date | None
    doses: list[MedicationDose]


class MedicationsRecord(RecordDto):
    as_of: AwareDatetime
    time_zone: str
    local_date: date
    treatments: list[Treatment]


class RecordReadResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "empty", "unavailable", "forbidden"]
    data: ProfileRecord | VitalsRecord | MedicationsRecord | None = None
    code: str | None = None
    message: str | None = None

    def to_tool_json(self) -> str:
        optional = {key for key in ("data", "code", "message") if getattr(self, key) is None}
        return self.model_dump_json(exclude=optional)


INVALID_VITALS_RANGE = RecordReadResult(
    status="unavailable",
    code="invalid_date_range",
    message="Indica un intervalo válido de hasta 31 días en fechas locales.",
)
