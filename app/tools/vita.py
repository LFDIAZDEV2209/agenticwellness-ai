"""Lecturas selectivas con identidad derivada del turno autenticado."""

from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from pydantic import ValidationError

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
    VitalsArguments,
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
        user_id = get_config().get("configurable", {}).get("user_id")
        result = await read_record(resource, user_id, arguments=parsed)
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
    "de peso, talla e IMC / Read my profile, goals, allergies, restrictions and anthropometry.",
)
get_my_vitals = _make_record_tool(
    "vitals",
    (
        "Lee mis últimos signos vitales y su historial con unidad, fecha y origen / Read my "
        "recorded vital signs and biometrics. Intervalo local inclusivo de hasta 31 días; "
        "por defecto el backend devuelve los últimos siete días. Si la fecha solicitada es "
        "relativa y no conoces el día local, usa el rango por defecto y las fechas de la respuesta."
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
