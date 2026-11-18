"""Lecturas selectivas con identidad derivada del turno autenticado."""

from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from pydantic import ValidationError

from app.vita.client import read_context, read_record
from app.vita.contracts import (
    INVALID_VITALS_RANGE,
    ContextEnvelope,
    EmptyArguments,
    RecordResource,
    Scope,
    VitalsArguments,
)

VITA_RECORD_TOOL_NAMES = frozenset({"get_my_profile", "get_my_vitals", "get_my_medications"})

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


def _make_record_tool(resource: RecordResource, description: str) -> StructuredTool:
    async def read() -> str:
        user_id = get_config().get("configurable", {}).get("user_id")
        result = await read_record(resource, user_id)
        return result.to_tool_json()

    return StructuredTool.from_function(
        coroutine=read,
        name=f"get_my_{resource}",
        description=description,
        args_schema=EmptyArguments,
    )


async def _read_vitals(**arguments: object) -> str:
    try:
        dates = VitalsArguments.model_validate(arguments)
    except ValidationError:
        return INVALID_VITALS_RANGE.to_tool_json()
    user_id = get_config().get("configurable", {}).get("user_id")
    result = await read_record("vitals", user_id, dates=dates)
    return result.to_tool_json()


get_my_profile = _make_record_tool(
    "profile",
    "Lee mi perfil activo, edad, metas registradas, alergias, restricciones y el historial "
    "de peso, talla e IMC / Read my profile, goals, allergies, restrictions and anthropometry.",
)
get_my_vitals = StructuredTool.from_function(
    coroutine=_read_vitals,
    name="get_my_vitals",
    description=(
        "Lee mis últimos signos vitales y su historial con unidad, fecha y origen / Read my "
        "recorded vital signs and biometrics. Intervalo local inclusivo de hasta 31 días; "
        "por defecto el backend devuelve los últimos siete días. Si la fecha solicitada es "
        "relativa y no conoces el día local, usa el rango por defecto y las fechas de la respuesta."
    ),
    # Publicar JSON Schema conserva el alias `from` en ambos proveedores.
    # _read_vitals valida la entrada contra el mismo modelo Pydantic.
    args_schema=VitalsArguments.model_json_schema(),
)
get_my_medications = _make_record_tool(
    "medications",
    "Lee mis tratamientos vigentes, medicamentos y suplementos, dosis, horarios y tomas "
    "del día / Read my current treatments and today's recorded dose status.",
)
