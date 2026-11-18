"""Registro tipado de tools: permisos por canal, presupuesto, deadline y reintentos.

Fuente única de política para el harness Vita (tarea 2.6). El catálogo de
`app/tools/registry.py` dice QUÉ tools existen; este módulo dice DÓNDE y CÓMO
pueden ejecutarse:

- `channels`: canales que pueden ver/invocar la tool. Fuera de ellos la tool no
  se enlaza al modelo y, si el modelo la pide igualmente, se rechaza con
  `tool_not_authorized` sin ejecutarla.
- `max_calls_per_turn`: presupuesto por tool dentro de un turno (además del
  presupuesto global `max_tool_calls` del grafo).
- `deadline_seconds`: tiempo máximo total de la llamada, incluidos los reintentos.
- `retries`: reintentos SOLO ante timeout y SOLO para tools idempotentes.
- `status`: `active` o `disabled`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.tools import BaseTool

Channel = Literal["text", "voice", "whatsapp"]
ToolStatus = Literal["active", "disabled"]

ALL_CHANNELS: frozenset[Channel] = frozenset({"text", "voice", "whatsapp"})
APP_CHANNELS: frozenset[Channel] = frozenset({"text", "voice"})


@dataclass(frozen=True)
class ToolSpec:
    """Política de ejecución de una tool."""

    name: str
    channels: frozenset[Channel] = field(default=ALL_CHANNELS)
    max_calls_per_turn: int = 2
    deadline_seconds: float = 3.0
    retries: int = 0
    idempotent: bool = True
    status: ToolStatus = "active"

    def __post_init__(self) -> None:
        if not self.channels <= ALL_CHANNELS:
            raise ValueError(f"Canal desconocido en la política de {self.name!r}")
        if self.max_calls_per_turn < 1 or self.deadline_seconds <= 0 or self.retries < 0:
            raise ValueError(f"Presupuesto inválido en la política de {self.name!r}")
        if self.retries and not self.idempotent:
            raise ValueError(f"{self.name!r} no es idempotente: no admite reintentos")

    def allows(self, channel: Channel) -> bool:
        return self.status == "active" and channel in self.channels


def _read(name: str, channels: frozenset[Channel] = ALL_CHANNELS) -> ToolSpec:
    """Lectura personal idempotente: una llamada por turno y un reintento por timeout."""
    return ToolSpec(name, channels, max_calls_per_turn=1, deadline_seconds=3.0, retries=1)


TOOL_SPECS: Mapping[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec("calculate", max_calls_per_turn=3, deadline_seconds=1.0),
        ToolSpec("get_current_time", max_calls_per_turn=2, deadline_seconds=1.0),
        _read("get_patient_data", APP_CHANNELS),
        # Flujo de controles del programa: solo existe en la app (chat de texto).
        ToolSpec(
            "mark_control_declined",
            frozenset({"text"}),
            max_calls_per_turn=1,
            idempotent=False,
        ),
        # Citas retiradas del canal paciente (ver DISABLED_PATIENT_TOOLS).
        ToolSpec("suggest_appointment", frozenset(), max_calls_per_turn=1, status="disabled"),
        _read("get_my_day"),
        _read("get_my_nutrition"),
        _read("get_my_assessments"),
        _read("get_my_podcast"),
        _read("get_my_exercise"),
        _read("get_my_progress"),
        ToolSpec("retrieve_knowledge", max_calls_per_turn=2, deadline_seconds=3.0, retries=1),
        _read("get_my_profile", APP_CHANNELS),
        _read("get_my_vitals", APP_CHANNELS),
        _read("get_my_medications", APP_CHANNELS),
        _read("get_my_nutrition_plan", APP_CHANNELS),
        _read("get_my_assessment_results", APP_CHANNELS),
        _read("list_my_documents", APP_CHANNELS),
        ToolSpec("read_my_document", APP_CHANNELS, max_calls_per_turn=1, deadline_seconds=8),
    )
}


def spec_for(name: str) -> ToolSpec:
    """Política de una tool ya seleccionada por el grafo o el perfil efectivo.

    Las tools inyectadas explícitamente heredan los canales del grafo o perfil.
    El test de catálogo exige entrada explícita para toda tool registrada.
    """
    return TOOL_SPECS.get(name) or ToolSpec(name, max_calls_per_turn=1_000)


def resolve_channel(configurable: Mapping[str, Any], profile_key: str | None = None) -> Channel:
    """Canal efectivo del turno: voz/WhatsApp explícitos o por perfil; si no, texto."""
    channel = configurable.get("message_channel")
    if channel in ("text", "voice", "whatsapp"):
        return channel
    if profile_key == "whatsapp":
        return "whatsapp"
    return "text"


def filter_for_channel(tools: Iterable[BaseTool], channel: Channel) -> list[BaseTool]:
    """Tools que el canal puede ver; el resto ni siquiera se ofrece al modelo."""
    return [tool for tool in tools if spec_for(tool.name).allows(channel)]
