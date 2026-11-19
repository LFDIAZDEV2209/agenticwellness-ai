"""Lecturas selectivas con identidad derivada del turno autenticado."""

import json
from datetime import UTC, date
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from pydantic import ValidationError

from app.tools.vita_capture import VITA_CAPTURE_TOOLS
from app.vita.client import INVALID_DOCUMENT_ARGUMENTS, read_context, read_document, read_record
from app.vita.contracts import (
    INVALID_NUTRITION_DATE,
    INVALID_VITALS_RANGE,
    ContextEnvelope,
    DocumentListArguments,
    DocumentReadArguments,
    EmptyArguments,
    NutritionArguments,
    RecordReadResult,
    RecordResource,
    Scope,
    VitalPoint,
    VitalsArguments,
    VitalsRecord,
)

VITA_RECORD_TOOL_NAMES = frozenset(
    {
        "get_my_profile",
        "get_my_vitals",
        "get_my_medications",
        "get_my_nutrition_plan",
        "get_my_assessment_results",
        "list_my_documents",
        "read_my_document",
    }
)

VITA_READ_TOOL_NAMES = VITA_RECORD_TOOL_NAMES | frozenset(
    {
        "get_my_day",
        "get_my_nutrition",
        "get_my_assessments",
        "get_my_podcast",
        "get_my_exercise",
        "get_my_progress",
    }
)

RECORD_TOPIC_TOOLS = {
    "profile": ("get_my_profile",),
    "vitals": ("get_my_vitals",),
    "medications": ("get_my_medications",),
    "assessments": ("get_my_assessment_results",),
    "nutrition_plan": ("get_my_nutrition_plan",),
    "documents": ("list_my_documents", "read_my_document"),
}
RECORD_TOOL_TOPICS = {
    **{tool: topic for topic, tools in RECORD_TOPIC_TOOLS.items() for tool in tools},
    "get_my_nutrition": "nutrition_plan",
    "get_my_assessments": "assessments",
}
_VOICE_VITAL_METRICS = (
    "step_count",
    "activity_kcal",
    "distance_m",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "o2_saturation",
    "glucose_fasting",
    "weight",
    "temperature_c",
    "sleep_minutes",
    "body_fat",
)


def _vitals_tool_json(result: RecordReadResult, *, voice: bool) -> str:
    record = result.data
    if not isinstance(record, VitalsRecord):
        return result.to_tool_json()
    if result.status != "ok":
        # El contrato textual conserva el envelope vacío de BACK.
        if not voice:
            return result.to_tool_json()
        return result.model_copy(update={"data": None}).to_tool_json()

    try:
        zone = ZoneInfo(record.time_zone)
    except (ValueError, ZoneInfoNotFoundError):
        if not voice:
            return result.to_tool_json()
        return RecordReadResult(status="unavailable", code="invalid_time_zone").to_tool_json()

    def local_point(point: VitalPoint) -> dict:
        local = point.observed_at.astimezone(zone)
        payload = {
            **point.model_dump(mode="json"),
            "localDate": local.date().isoformat(),
            "localTime": local.strftime("%H:%M"),
        }
        if voice:
            payload["observedAt"] = (
                point.observed_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
            )
        return payload

    payload = result.model_dump(
        mode="json", exclude_none=True, exclude={"data": {"latest", "history"}}
    )
    data = payload["data"]
    if not voice:
        data.update(
            latest=[local_point(point) for point in record.latest],
            history=[local_point(point) for point in record.history],
        )
        return json.dumps(payload, ensure_ascii=False)

    # Latest puede incluir antropometría anterior al intervalo solicitado.
    # Solo cuenta para el resumen diario si pertenece al intervalo local de BACK.
    daily: dict[date, dict[str, VitalPoint]] = {}
    for point in sorted(
        [*record.latest, *record.history], key=lambda p: p.observed_at, reverse=True
    ):
        day = point.observed_at.astimezone(zone).date()
        if record.from_date <= day <= record.to and point.metric_code in _VOICE_VITAL_METRICS:
            daily.setdefault(day, {}).setdefault(point.metric_code, point)

    days = sorted(daily, reverse=True)[:7]
    points = [
        daily[day][code] for day in days for code in _VOICE_VITAL_METRICS if code in daily[day]
    ]

    # El historial oral contiene un punto compacto por día y métrica. Evita
    # duplicar ahí los metadatos crudos que ya conserva el DTO para otros canales.
    def compact(point: VitalPoint) -> dict:
        value = local_point(point)
        value.pop("unitCode")
        return value

    latest = sorted(record.latest, key=lambda point: point.metric_code not in _VOICE_VITAL_METRICS)[
        : len(_VOICE_VITAL_METRICS)
    ]
    data.update(
        latest=[local_point(point) for point in latest],
        history=[compact(point) for point in points],
        daysReturned=len(days),
        truncated=(
            record.has_more
            or len(record.latest) > len(latest)
            or any(point not in points for point in record.history)
        ),
    )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def make_context_tool(name: str, scope: Scope, description: str) -> StructuredTool:
    async def read() -> str:
        user_id = get_config().get("configurable", {}).get("user_id")
        result = (
            await read_context(scope, str(user_id))
            if user_id
            else ContextEnvelope(status="forbidden")
        )
        return result.model_dump_json(exclude_none=True)

    return StructuredTool.from_function(
        coroutine=read,
        name=name,
        description=description,
        args_schema=EmptyArguments,
    )


get_my_day = make_context_tool(
    "get_my_day",
    "day",
    "Read my effective program and tasks for today / programa y tareas de hoy.",
)
get_my_nutrition = make_context_tool(
    "get_my_nutrition",
    "nutrition",
    "Read today's current published nutrition plan and meals / plan nutricional y menú; no drafts.",
)
get_my_assessments = make_context_tool(
    "get_my_assessments",
    "assessments",
    "Read my assigned tests / mis evaluaciones; never answer or submit them.",
)
get_my_podcast = make_context_tool(
    "get_my_podcast",
    "podcast",
    "Read my assigned podcast and authorized shortcut / podcast asignado; no XP on open.",
)
get_my_exercise = make_context_tool(
    "get_my_exercise",
    "exercise",
    "Read my effective exercise or rest day / rutina o descanso de hoy; no changes.",
)
get_my_progress = make_context_tool(
    "get_my_progress",
    "progress",
    "Read my recorded progress with period/sources / mi progreso; no clinical judgments.",
)


def _make_record_tool(
    resource: RecordResource,
    description: str,
    *,
    name: str | None = None,
    schema: type[VitalsArguments | NutritionArguments | DocumentListArguments] | None = None,
    invalid: RecordReadResult | None = None,
) -> StructuredTool:
    async def read(**arguments: object) -> str:
        try:
            parsed = schema.model_validate(arguments) if schema else None
        except ValidationError:
            assert invalid is not None
            return invalid.to_tool_json()
        configurable = get_config().get("configurable", {})
        result = await read_record(resource, configurable.get("user_id"), arguments=parsed)
        if resource == "vitals":
            return _vitals_tool_json(result, voice=configurable.get("message_channel") == "voice")
        return result.to_tool_json()

    return StructuredTool.from_function(
        coroutine=read,
        name=name or f"get_my_{resource}",
        description=description,
        # JSON Schema conserva los nombres del contrato en ambos proveedores.
        args_schema=schema.model_json_schema() if schema else EmptyArguments,
    )


get_my_profile = _make_record_tool(
    "profile",
    "Lee mi perfil activo, edad, metas registradas, alergias, restricciones y el historial "
    "de peso, talla e IMC / Read my profile, goals, allergies, restrictions and anthropometry. "
    "Usa esta herramienta para preguntas de peso, talla o IMC: incluye la última medición.",
)
get_my_vitals = _make_record_tool(
    "vitals",
    (
        "Lee mis últimos signos vitales y su historial con unidad, fecha y origen / Read my "
        "recorded vital signs and biometrics. Intervalo local inclusivo de hasta 31 días; "
        "por defecto el backend devuelve los últimos siete días. Si la fecha solicitada es "
        "relativa y no conoces el día local, usa el rango por defecto y las fechas de la "
        "respuesta. "
        "Para peso, talla o IMC usa get_my_profile, no esta herramienta."
    ),
    schema=VitalsArguments,
    invalid=INVALID_VITALS_RANGE,
)
get_my_medications = _make_record_tool(
    "medications",
    "Lee mis tratamientos vigentes, medicamentos y suplementos, dosis, horarios y tomas "
    "del día / Read my current treatments and today's recorded dose status.",
)

get_my_nutrition_plan = _make_record_tool(
    "nutrition",
    "Lee mi ciclo nutricional publicado de hasta 30 días, recomendaciones visibles e historial "
    "de comidas registradas. Usa date local YYYY-MM-DD solo cuando se solicita otra fecha.",
    name="get_my_nutrition_plan",
    schema=NutritionArguments,
    invalid=INVALID_NUTRITION_DATE,
)
get_my_assessment_results = _make_record_tool(
    "assessments",
    "Lee mis baterías y resultados completados visibles, con puntajes e interpretaciones "
    "registradas. Para pendientes o en curso devuelve solo estado y metadatos.",
    name="get_my_assessment_results",
)
list_my_documents = _make_record_tool(
    "documents",
    "Lista mis documentos de expediente con ID, nombre, tipo, fecha y estado. Usa page y "
    "pageSize si necesitas otra página. Para consultar el texto usa read_my_document con el ID.",
    name="list_my_documents",
    schema=DocumentListArguments,
    invalid=INVALID_DOCUMENT_ARGUMENTS,
)


async def _read_document(**arguments: object) -> str:
    try:
        parsed = DocumentReadArguments.model_validate(arguments)
    except ValidationError:
        return INVALID_DOCUMENT_ARGUMENTS.to_tool_json()
    user_id = get_config().get("configurable", {}).get("user_id")
    return (await read_document(user_id, parsed.id)).to_tool_json()


read_my_document = StructuredTool.from_function(
    coroutine=_read_document,
    name="read_my_document",
    description="Lee el texto extraíble de un documento propio usando el ID de list_my_documents.",
    args_schema=DocumentReadArguments.model_json_schema(),
)

VITA_TOOL_ARGUMENTS = {
    tool.name: frozenset(tool.args_schema["properties"])
    for tool in (get_my_vitals, get_my_nutrition_plan, list_my_documents, read_my_document)
}
# Captura: solo los campos públicos del esquema (el estado inyectado no es argumento).
VITA_TOOL_ARGUMENTS.update(
    {
        tool.name: frozenset(tool.tool_call_schema.model_json_schema()["properties"])
        for tool in VITA_CAPTURE_TOOLS
    }
)
