"""Captura confirmada de Vita (KAN-70): contrato, identidad del turno y gate de confirmación.

Flujo en dos fases, idéntico en chat, voz y WhatsApp:

1. Una tool `record_*` pide a BACK una **propuesta** firmada (AES-GCM, 5 min). No escribe.
   El token viaja solo en el `artifact` del ToolMessage (checkpointer), nunca en el texto
   que ve el modelo.
2. `confirm_capture` escribe **solo** si hay propuestas del turno inmediatamente anterior
   y el mensaje actual del paciente es una confirmación explícita (regla determinista, no
   decisión del modelo). BACK valida token, dueño, canal y `turnId`; el recibo
   `(patient, channel, turn_id, tool)` hace idempotente el reintento.

La identidad (`user_id`) y el canal salen del `configurable` del turno autenticado por BACK,
nunca de los argumentos de la tool ni del texto.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from app.tools.policy import Channel, resolve_channel
from app.vita.routing import normalize

MEASUREMENT_TOOL = "record_measurement"
MEDICATION_TOOL = "record_medication_dose"
MEAL_TOOL = "record_meal"
CHECKIN_TOOL = "record_checkin"
CONFIRM_TOOL = "confirm_capture"
CAPTURE_PROPOSE_TOOLS = (MEASUREMENT_TOOL, MEDICATION_TOOL, MEAL_TOOL, CHECKIN_TOOL)
VITA_CAPTURE_TOOL_NAMES = frozenset({*CAPTURE_PROPOSE_TOOLS, CONFIRM_TOOL})

# Canal del harness → procedencia canónica del backend (`VitaCaptureRules.IsChannel`).
CHANNEL_SOURCES: Mapping[Channel, str] = {
    "text": "vita_chat",
    "voice": "vita_voice",
    "whatsapp": "vita_whatsapp",
}
PENDING_ARTIFACT = "vita_capture_pending"
MAX_PENDING = 3
MAX_TURN_ID = 256  # `VitaCaptureRules.IsTurnId`


class CaptureArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MeasurementArguments(CaptureArguments):
    metric_code: Literal[
        "weight", "heart_rate", "systolic_bp", "diastolic_bp", "o2_saturation", "temperature_c"
    ] = Field(description="Métrica canónica. La presión se registra como dos mediciones.")
    value: float = Field(gt=0, le=1000, description="Valor tal como lo dijo el paciente.")
    unit_code: Literal["kg", "bpm", "mmhg", "pct", "celsius"] = Field(
        description="weight=kg, heart_rate=bpm, *_bp=mmhg, o2_saturation=pct, "
        "temperature_c=celsius. Nunca conviertas unidades: si dio otra, pregunta."
    )
    observed_at: AwareDatetime | None = Field(
        default=None, description="Momento de la toma con zona horaria; omítelo si fue ahora."
    )
    supersedes_id: UUID | None = Field(
        default=None, description="ID de una medición propia previa que corrige, si aplica."
    )


class MedicationDoseArguments(CaptureArguments):
    dose_id: UUID = Field(description="ID de la toma del día devuelto por get_my_medications.")
    scheduled_at: AwareDatetime = Field(description="Hora programada exacta de esa toma.")
    status: Literal["taken", "omitted"]
    observed_at: AwareDatetime | None = None


class MealArguments(CaptureArguments):
    meal_code: Literal["des", "alm", "mer", "cen"] = Field(
        description="des=desayuno, alm=almuerzo, mer=merienda, cen=cena."
    )
    calories: float | None = Field(default=None, ge=0, le=10_000)
    protein_g: float | None = Field(default=None, ge=0, le=1_000)
    carbs_g: float | None = Field(default=None, ge=0, le=1_000)
    fat_g: float | None = Field(default=None, ge=0, le=1_000)
    fiber_g: float | None = Field(default=None, ge=0, le=1_000)
    observed_at: AwareDatetime | None = None


class CheckinArguments(CaptureArguments):
    mood_score: int = Field(ge=1, le=5, description="Ánimo del día de 1 (muy mal) a 5 (muy bien).")
    observed_at: AwareDatetime | None = None


class ConfirmArguments(CaptureArguments):
    confirmed: bool = Field(
        description="true solo si el paciente acaba de confirmar; false si lo rechazó."
    )
    confirmed_unusual: bool = Field(
        default=False,
        description="true solo si la propuesta pedía confirmación adicional y el paciente la dio.",
    )


def backend_payload(tool: str, arguments: CaptureArguments) -> dict[str, Any]:
    """Cuerpo `ProposeVitaCaptureRequest` (camelCase) sin `channel`, `turnId` ni `observedAt`."""
    data = arguments.model_dump(mode="json", exclude_none=True, exclude={"observed_at"})
    if tool == MEASUREMENT_TOOL:
        return {
            "metricCode": data["metric_code"],
            "value": data["value"],
            "unitCode": data["unit_code"],
            **({"supersedesId": data["supersedes_id"]} if "supersedes_id" in data else {}),
        }
    if tool == MEDICATION_TOOL:
        return {
            "doseId": data["dose_id"],
            "scheduledAt": data["scheduled_at"],
            "status": data["status"],
        }
    if tool == MEAL_TOOL:
        names = {"protein_g": "proteinG", "carbs_g": "carbsG", "fat_g": "fatG", "fiber_g": "fiberG"}
        meal = {"mealCode": data.pop("meal_code")}
        meal.update({names.get(key, key): value for key, value in data.items()})
        return {"meal": meal}
    if tool == CHECKIN_TOOL:
        return {"moodScore": data["mood_score"]}
    raise ValueError(f"Tool de captura desconocida: {tool!r}")


def capture_channel(configurable: Mapping[str, Any]) -> str:
    return CHANNEL_SOURCES[resolve_channel(configurable)]


def owner_id(configurable: Mapping[str, Any]) -> str | None:
    try:
        owner = UUID(str(configurable.get("user_id")))
    except ValueError:
        return None
    return str(owner) if owner.int else None


def turn_base(configurable: Mapping[str, Any]) -> str | None:
    """ID estable del turno de origen: `wamid` (WhatsApp), mensaje de voz o ejecución (chat)."""
    channel = resolve_channel(configurable)
    if channel == "whatsapp":
        base = configurable.get("channel_message_id")
    elif channel == "voice":
        base = configurable.get("user_message_id")
    else:
        base = configurable.get("execution_id")
    return str(base) if base else None


def capture_turn_id(base: str, tool: str, payload: Mapping[str, Any]) -> str:
    """`turnId` determinista por turno y contenido.

    El recibo es único por `(paciente, canal, turnId, tool)`: dos mediciones del mismo
    turno (sistólica y diastólica) necesitan IDs distintos, y el reintento del mismo
    mensaje (p. ej. redelivery del `wamid`) debe producir el mismo ID para reutilizar
    el recibo. Si el ID de origen es largo o tiene controles, se resume con SHA-256.
    """
    canonical = json.dumps({"tool": tool, **payload}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    turn_id = f"{base}:{digest}"
    if len(turn_id) > MAX_TURN_ID or turn_id != turn_id.strip() or _has_control(turn_id):
        turn_id = f"sha256:{hashlib.sha256(base.encode()).hexdigest()}:{digest}"
    return turn_id


def _has_control(value: str) -> bool:
    return any(ord(char) < 32 or 127 <= ord(char) < 160 for char in value)


def observed_at(arguments: CaptureArguments, now: datetime | None = None) -> datetime | None:
    """Momento de la toma; `None` si es futuro (BACK lo rechazaría igualmente)."""
    now = now or datetime.now(UTC)
    value = getattr(arguments, "observed_at", None) or now
    return None if value > now else value


# Confirmación explícita y breve. Cualquier cifra, negación o matiz exige una propuesta nueva.
_AFFIRM = re.compile(
    r"^(?:si|yes|yep|yeah|sure|ok|okay|vale|dale|claro|correcto|correct|exacto|exactly|"
    r"confirmo|confirmado|confirma|confirm|confirmed|de acuerdo|adelante|go ahead|asi es|"
    r"that'?s right|right|por supuesto|of course|registralo|guardalo|hazlo|do it|save it|"
    r"record it|please do)\b"
)
_HEDGE = re.compile(
    r"\b(?:no|not|nope|cancela|cancelar|cancel|espera|wait|todavia|aun no|don'?t|dont|"
    r"nunca|never|pero|but|cambia|change|corrige|correct it|actually|en realidad|mejor)\b"
)


def explicit_confirmation(text: str | None) -> bool:
    value = normalize(text or "").strip(" \t\n.,;:!¡?¿")
    if not value or len(value) > 60 or re.search(r"\d", value) or _HEDGE.search(value):
        return False
    return bool(_AFFIRM.match(value))


def confirmation_text(state: Mapping[str, Any], configurable: Mapping[str, Any]) -> str:
    """Mensaje actual del paciente; en voz, solo la última frase transcrita."""
    voice_text = configurable.get("voice_user_text")
    if resolve_channel(configurable) == "voice" and isinstance(voice_text, str):
        return voice_text
    return str(state.get("input") or "")


def pending_artifact(proposals: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    return {"kind": PENDING_ARTIFACT, "proposals": list(proposals)} if proposals else None


def previous_turn_proposals(
    messages: Sequence[BaseMessage], turn_start_index: int, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Propuestas vigentes emitidas en el turno inmediatamente anterior.

    Solo el turno anterior: un «sí» varios turnos después no confirma algo que el
    paciente ya no tiene delante. Las propuestas del turno actual tampoco cuentan,
    así el modelo no puede proponer y confirmar sin respuesta del paciente.
    """
    now = now or datetime.now(UTC)
    prior = list(messages[: max(turn_start_index, 0)])
    starts = [index for index, message in enumerate(prior) if isinstance(message, HumanMessage)]
    if not starts:
        return []
    proposals: dict[str, dict[str, Any]] = {}
    for message in prior[starts[-1] + 1 :]:
        artifact = getattr(message, "artifact", None)
        if not (
            isinstance(message, ToolMessage)
            and message.name in VITA_CAPTURE_TOOL_NAMES
            and isinstance(artifact, dict)
            and artifact.get("kind") == PENDING_ARTIFACT
        ):
            continue
        for proposal in artifact.get("proposals", ()):
            try:
                expires = datetime.fromisoformat(str(proposal["expires_at"]))
            except (KeyError, ValueError):
                continue
            if expires.tzinfo is not None and expires > now:
                proposals[str(proposal["turn_id"])] = dict(proposal)
    return list(proposals.values())[-MAX_PENDING:]


def has_pending_proposals(state: Mapping[str, Any]) -> bool:
    """Propuestas del último turno persistido (para enrutar el «sí» al planificador)."""
    messages = list(state.get("messages", ()))
    return bool(previous_turn_proposals(messages, len(messages)))
