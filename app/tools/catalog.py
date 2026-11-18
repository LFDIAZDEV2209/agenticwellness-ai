"""Validación del catálogo de tools contra el esquema nativo de cada proveedor.

`bind_tools` convierte cada tool al formato de Anthropic/OpenAI en cada turno.
Si una tool no convierte (p. ej. un nombre `str` sin resolver, incidente P0 de
WhatsApp del 12-11-2026: `ValueError: Unsupported function calculate`), el
turno revienta con 500. Este módulo adelanta esa conversión al arranque y a la
suite de pruebas para que el fallo nunca llegue a producción.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from langchain_anthropic.chat_models import convert_to_anthropic_tool
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.core.errors import ToolCatalogError
from app.tools.registry import ALL_TOOLS, TOOLS_BY_NAME, legacy_tools, patient_tools

# Conversores que usan `ChatAnthropic.bind_tools` / `ChatOpenAI.bind_tools`.
PROVIDER_CONVERTERS: dict[str, Callable[[Any], Any]] = {
    "anthropic": convert_to_anthropic_tool,
    "openai": convert_to_openai_tool,
}


def tool_label(item: object) -> str:
    """Nombre legible de la tool para logs (nunca su contenido ni argumentos)."""
    if isinstance(item, BaseTool):
        return item.name
    if isinstance(item, str):
        return f"{item!r} (str sin resolver)"
    return f"{type(item).__name__} (tipo no soportado)"


def conversion_failures(tools: Iterable[object]) -> list[str]:
    """Devuelve una línea por cada (tool, proveedor) que no convierte; [] si todo convierte."""
    failures: list[str] = []
    for item in tools:
        for provider, convert in PROVIDER_CONVERTERS.items():
            try:
                convert(item)
            except Exception as exc:  # cualquier fallo de conversión es del catálogo
                failures.append(f"{tool_label(item)} -> {provider}: {type(exc).__name__}")
    return failures


def ensure_convertible(tools: Iterable[object], *, context: str) -> None:
    """Lanza `ToolCatalogError` nombrando la(s) tool(s) culpable(s) si alguna no convierte."""
    failures = conversion_failures(tools)
    if failures:
        raise ToolCatalogError(
            f"Tools no convertibles al esquema del proveedor en {context}: " + "; ".join(failures)
        )


def resolve_profile_tools(names: Iterable[str | BaseTool], *, context: str) -> list[BaseTool]:
    """Resuelve nombres del perfil a `BaseTool`; un nombre desconocido es error de catálogo."""
    resolved: list[BaseTool] = []
    for item in names:
        if isinstance(item, BaseTool):
            resolved.append(item)
        elif item in TOOLS_BY_NAME:
            resolved.append(TOOLS_BY_NAME[item])
        else:
            raise ToolCatalogError(f"Tool {item!r} declarada en {context} no está registrada")
    return resolved


def profile_tool_catalog() -> dict[str, list[BaseTool]]:
    """Catálogo efectivo por perfil de código, con la misma política que `_active_tools`."""
    from app.agents.registry import AGENTS

    catalog: dict[str, list[BaseTool]] = {
        "registry:all": list(ALL_TOOLS),
        "registry:legacy": legacy_tools(include_patient=True),
    }
    for key, profile in AGENTS.items():
        base = (
            resolve_profile_tools(profile.tools, context=f"perfil {key!r}")
            if profile.tools
            else legacy_tools()
        )
        catalog[f"profile:{key}"] = patient_tools([*base, *profile.extra_tools])
    return catalog


def validate_tool_catalog() -> None:
    """Valida todo el catálogo registrado; se invoca en el arranque del servicio."""
    for context, tools in profile_tool_catalog().items():
        ensure_convertible(tools, context=context)
