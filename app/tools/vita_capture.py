"""Tools de captura confirmada de Vita (KAN-70). Ver `app/vita/capture.py` para el contrato.

Las tools `record_*` solo PROPONEN. `confirm_capture` es la única que escribe y exige
propuestas del turno anterior más una confirmación explícita del paciente en el mensaje
actual. Ninguna acepta `userId`, `patientId`, canal ni `turnId` como argumento.
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

from langchain_core.tools import BaseTool, StructuredTool, tool
from langgraph.config import get_config
from langgraph.prebuilt import InjectedState
from pydantic import ValidationError

from app.vita.capture import (
    CHECKIN_TOOL,
    CONFIRM_TOOL,
    MEAL_TOOL,
    MEASUREMENT_TOOL,
    MEDICATION_TOOL,
    CaptureArguments,
    CheckinArguments,
    MealArguments,
    MeasurementArguments,
    MedicationDoseArguments,
    backend_payload,
    capture_channel,
    capture_turn_id,
    confirmation_text,
    explicit_confirmation,
    observed_at,
    owner_id,
    pending_artifact,
    previous_turn_proposals,
    turn_base,
)
from app.vita.client import post_capture

# Margen bajo el deadline del nodo de tools (3 s por defecto): la tool responde con su
# propio resultado y conserva la propuesta pendiente en lugar de ser cancelada.
_PROPOSE_TIMEOUT = 2.5
_CONFIRM_TIMEOUT = 2.5
_RETRYABLE = frozenset({"tool_timeout", "vita_capture_unavailable"})

ToolResult = tuple[str, dict[str, Any] | None]


def _result(payload: dict[str, Any], artifact: dict[str, Any] | None = None) -> ToolResult:
    return json.dumps(payload, ensure_ascii=False, default=str), artifact


def _failure(code: str) -> ToolResult:
    return _result(
        {
            "status": "unavailable",
            "code": code,
            "saved": False,
            "message": "No se guardó nada. Explica el motivo al paciente sin inventar datos.",
        }
    )


async def _propose(
    name: str, schema: type[CaptureArguments], arguments: dict[str, Any]
) -> ToolResult:
    try:
        parsed = schema.model_validate(arguments)
    except ValidationError:
        return _failure("invalid_capture_arguments")
    configurable = get_config().get("configurable", {})
    user_id, base = owner_id(configurable), turn_base(configurable)
    if user_id is None or base is None:
        return _failure("vita_identity_required")
    when = observed_at(parsed)
    if when is None:
        return _failure("vita_capture_invalid_date")
    payload = backend_payload(name, parsed)
    # El momento explícito forma parte del contenido; «ahora» no, para que el
    # reintento del mismo mensaje conserve su turnId.
    explicit_when = getattr(parsed, "observed_at", None)
    identity = {**payload, "observedAt": explicit_when.isoformat()} if explicit_when else payload
    channel = capture_channel(configurable)
    turn_id = capture_turn_id(base, name, identity)
    ok, response = await post_capture(
        "propose",
        user_id,
        {
            "tool": name,
            "channel": channel,
            "turnId": turn_id,
            "observedAt": when.isoformat(),
            **payload,
        },
        _PROPOSE_TIMEOUT,
    )
    if not ok:
        return _failure(str(response))
    token, expires = response.get("token"), response.get("expiresAt")
    if not isinstance(token, str) or not token or not isinstance(expires, str):
        return _failure("vita_capture_unavailable")
    unusual = bool(response.get("requiresAdditionalConfirmation"))
    return _result(
        {
            "status": "proposed",
            "saved": False,
            "tool": name,
            "data": response.get("data"),
            "expires_at": expires,
            "requires_additional_confirmation": unusual,
            "instruction": (
                "Nothing is saved yet. Summarize these exact values and ask the patient to "
                "confirm explicitly (yes/no). Call confirm_capture only in the NEXT turn, after "
                "the patient replies."
                + (
                    " The value is unusual: ask for an additional explicit confirmation."
                    if unusual
                    else ""
                )
            ),
        },
        pending_artifact(
            [
                {
                    "tool": name,
                    "token": token,
                    "channel": channel,
                    "turn_id": turn_id,
                    "expires_at": expires,
                    "requires_additional_confirmation": unusual,
                }
            ]
        ),
    )


def _propose_tool(name: str, schema: type[CaptureArguments], description: str) -> StructuredTool:
    async def propose(**arguments: Any) -> ToolResult:
        return await _propose(name, schema, arguments)

    return StructuredTool.from_function(
        coroutine=propose,
        name=name,
        description=description,
        args_schema=schema,
        response_format="content_and_artifact",
    )


record_measurement = _propose_tool(
    MEASUREMENT_TOOL,
    MeasurementArguments,
    "Propone registrar una medición que el paciente reporta (peso, pulso, presión, "
    "saturación, temperatura). NO guarda: devuelve una propuesta que el paciente debe "
    "confirmar / Propose recording a patient-reported measurement; nothing is saved yet.",
)
record_medication_dose = _propose_tool(
    MEDICATION_TOOL,
    MedicationDoseArguments,
    "Propone marcar una toma del día como tomada u omitida. Usa dose_id y scheduled_at de "
    "get_my_medications. NO guarda hasta que el paciente confirme / Propose a dose update.",
)
record_meal = _propose_tool(
    MEAL_TOOL,
    MealArguments,
    "Propone registrar una comida del programa (desayuno, almuerzo, merienda o cena) con "
    "nutrientes opcionales que el paciente indique. NO guarda / Propose logging a meal.",
)
record_checkin = _propose_tool(
    CHECKIN_TOOL,
    CheckinArguments,
    "Propone registrar el check-in de ánimo del día (1 a 5). NO guarda / Propose a mood "
    "check-in; nothing is saved until the patient confirms.",
)


async def _confirm_one(user_id: str, proposal: dict[str, Any], unusual: bool) -> dict[str, Any]:
    ok, response = await post_capture(
        "confirm",
        user_id,
        {
            "token": proposal["token"],
            "channel": proposal["channel"],
            "turnId": proposal["turn_id"],
            "confirmed": True,
            "confirmedUnusual": unusual and proposal.get("requires_additional_confirmation"),
        },
        _CONFIRM_TIMEOUT,
    )
    if ok:
        return {
            "tool": proposal["tool"],
            "status": "recorded",
            "receipt_id": response.get("receiptId"),
            "entity_type": response.get("entityType"),
        }
    return {"tool": proposal["tool"], "status": "failed", "code": str(response)}


@tool(CONFIRM_TOOL, response_format="content_and_artifact")
async def confirm_capture(
    confirmed: bool,
    state: Annotated[dict, InjectedState],
    confirmed_unusual: bool = False,
) -> ToolResult:
    """Confirma o descarta las capturas propuestas en el turno anterior. Úsala SOLO cuando
    el paciente acaba de responder sí o no a esa propuesta; nunca en el mismo turno de la
    propuesta. Confirm or discard the captures proposed in the previous turn.

    Args:
        confirmed: true si el paciente confirmó explícitamente; false si lo rechazó.
        confirmed_unusual: true solo si se pidió confirmación adicional y el paciente la dio.
    """
    configurable = get_config().get("configurable", {})
    proposals = previous_turn_proposals(
        list(state.get("messages", ())), int(state.get("turn_start_index", 0))
    )
    if not proposals:
        return _failure("vita_capture_no_pending_proposal")
    if not confirmed:
        return _result({"status": "cancelled", "saved": False})
    user_id = owner_id(configurable)
    channel = capture_channel(configurable)
    if user_id is None or any(p.get("channel") != channel for p in proposals):
        return _failure("vita_capture_turn_mismatch")
    if not explicit_confirmation(confirmation_text(state, configurable)):
        # El modelo no puede sustituir el «sí» del paciente. La propuesta sigue vigente.
        return _result(
            {
                "status": "needs_confirmation",
                "code": "vita_capture_explicit_confirmation_required",
                "saved": False,
                "message": "Pide al paciente un sí o no explícito.",
            },
            pending_artifact(proposals),
        )
    if any(p.get("requires_additional_confirmation") for p in proposals) and not confirmed_unusual:
        return _result(
            {
                "status": "needs_confirmation",
                "code": "vita_capture_additional_confirmation_required",
                "saved": False,
            },
            pending_artifact(proposals),
        )
    outcomes = await asyncio.gather(
        *(_confirm_one(user_id, proposal, confirmed_unusual) for proposal in proposals)
    )
    # Solo un fallo transitorio conserva la propuesta: el siguiente «sí» reintenta con
    # el mismo token/turnId y BACK devuelve el mismo recibo (sin duplicar).
    retry = [
        proposal
        for proposal, outcome in zip(proposals, outcomes, strict=True)
        if outcome.get("code") in _RETRYABLE
    ]
    return _result(
        {
            "status": "recorded" if all(o["status"] == "recorded" for o in outcomes) else "partial",
            "saved": any(o["status"] == "recorded" for o in outcomes),
            "results": outcomes,
        },
        pending_artifact(retry),
    )


VITA_CAPTURE_TOOLS: tuple[BaseTool, ...] = (
    record_measurement,
    record_medication_dose,
    record_meal,
    record_checkin,
    confirm_capture,
)
