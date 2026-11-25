"""Nodos del grafo supervisor."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from collections.abc import Awaitable, Callable
from contextlib import suppress
from uuid import uuid4

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.config import get_config
from langgraph.prebuilt import ToolNode
from langgraph.types import Overwrite

from app.agents.prompts import (
    BASE_SYSTEM_PROMPT,
    PATIENT_GUIDANCE_POLICY,
    PATIENT_SCOPE_POLICY,
    SERVICE_DELIVERY_POLICY,
    TEXT_DELIVERY_POLICY,
    VITA_CAPTURE_POLICY,
    VOICE_DELIVERY_POLICY,
    WHATSAPP_DATA_POLICY,
    WHATSAPP_FORMAT_POLICY,
    WHATSAPP_SCOPE_POLICY,
    build_control_guidance,
    build_language_instruction,
    build_turn_policy,
)
from app.agents.registry import AgentProfile
from app.api.message_text import extract_message_text
from app.core.config import get_settings
from app.graph.language import detect_turn_language
from app.graph.state import AgentState
from app.memory.adaptive import AdaptiveMemoryService
from app.memory.service import UserMemoryService
from app.safety.guardrails import check_input_guardrails
from app.tools.catalog import conversion_failures, tool_label
from app.tools.policy import TOOL_SPECS, filter_for_channel, resolve_channel, spec_for
from app.tools.registry import TOOLS_BY_NAME, legacy_tools, patient_tools
from app.tools.vita import RECORD_TOOL_TOPICS, RECORD_TOPIC_TOOLS, VITA_TOOL_ARGUMENTS
from app.vita.capture import VITA_CAPTURE_TOOL_NAMES
from app.vita.context_builder import (
    MESSAGE_OVERHEAD_TOKENS,
    ContextBudget,
    build_context,
    estimate_tokens,
)
from app.vita.routing import application_name_reply, service_query
from app.voice.delivery import format_voice_answer, voice_turn_has_results

logger = logging.getLogger(__name__)

# Código estable que viaja en `additional_kwargs` cuando el turno no puede
# enlazar sus tools: el canal (chat, voz, WhatsApp) recibe un texto seguro y
# el backend puede distinguirlo de una respuesta normal sin parsear el texto.
TOOL_CATALOG_ERROR_CODE = "tool_catalog_unavailable"


def guardrails_node(state: AgentState, *, record_topic: str | None = None) -> dict:
    """Nodo de entrada: valida y sanitiza el mensaje del usuario.

    Si es inseguro, termina el turno (el edge condicional lo enruta a END)
    con un mensaje de rechazo. Si es seguro, deja la versión sanitizada en
    `state["input"]` y el flujo continúa al agente.
    """
    result = check_input_guardrails(state.get("input", ""))
    configurable = get_config().get("configurable", {})
    stored_messages = list(state.get("messages", []))
    prior_messages = completed_tool_history(stored_messages)
    interrupted = len(prior_messages) != len(stored_messages)
    amended_at = None
    if configurable.get("message_channel") == "voice" and configurable.get("user_message_id"):
        amended_at = next(
            (
                index
                for index, message in enumerate(prior_messages)
                if isinstance(message, HumanMessage)
                and message.id == configurable["user_message_id"]
            ),
            None,
        )

    def turn_messages(messages):
        # El mismo ID solo vuelve a ejecutarse tras una enmienda ASR admitida.
        # Sustituye su borrador completo, incluidas tools y respuesta provisional;
        # no deja evidencia antigua en el prompt ni duplica el Human visible.
        return (
            Overwrite([*prior_messages[:amended_at], *messages])
            if amended_at is not None or interrupted
            else messages
        )

    visible_input = result.sanitized
    if configurable.get("message_channel") == "voice" and isinstance(
        configurable.get("voice_user_text"), str
    ):
        fragment = check_input_guardrails(configurable["voice_user_text"])
        visible_input = fragment.sanitized
        if not fragment.safe:
            result = fragment
    message_metadata = {
        "channel": configurable.get("message_channel", "text"),
        "status": "generated" if configurable.get("message_channel") == "voice" else "final",
    }
    if configurable.get("channel_message_id"):
        # Referencia del proveedor (p. ej. `wamid`): permite deduplicar y enlazar el
        # mensaje del hilo compartido con la app sin copiar contenido.
        message_metadata["channel_message_id"] = configurable["channel_message_id"]
    if configurable.get("voice_session_id"):
        message_metadata["voice_session_id"] = configurable["voice_session_id"]
        message_metadata["source"] = "vita_runtime"
    attachments = configurable.get("vita_attachments")
    # Referencia mínima del adjunto en el hilo (id + tipo); jamás su contenido.
    attachment_refs = [dict(ref) for ref in getattr(attachments, "refs", ())]
    # El checkpointer conserva el historial, pero estas salidas pertenecen a
    # UN turno. Overwrite evita que operator.add conserve acciones anteriores.
    turn_state = {
        "tools_used": Overwrite([]),
        "suggestions": Overwrite([]),
        "rag_sources": [],
        "turn_start_index": amended_at if amended_at is not None else len(prior_messages),
        "tool_call_count": 0,
        "tool_budget_exhausted": False,
        "record_topic": None,
        "app_actions": Overwrite([]),
        "data_sources": Overwrite([]),
        "turn_context_changed": state.get("context_agent") != state.get("agent", "base"),
    }
    if configurable.get("vita_history") is not None:
        turn_state.update(memory_context="", experience_context="")

    if not result.safe:
        return {
            **turn_state,
            "guardrail": {
                "safe": False,
                "reason": result.reason,
                "pattern": result.pattern,
            },
            "messages": turn_messages(
                [
                    AIMessage(
                        content=result.reason or "Mensaje bloqueado por los guardrails.",
                        id=configurable.get("assistant_message_id"),
                        additional_kwargs={
                            **message_metadata,
                            "execution_id": configurable.get("execution_id"),
                        },
                    )
                ]
            ),
        }

    return {
        **turn_state,
        "record_topic": record_topic if configurable.get("message_channel") == "voice" else None,
        "input": result.sanitized,
        # Una sola entrada por turno: el ciclo agent/tools reutiliza el historial.
        "messages": turn_messages(
            [
                HumanMessage(
                    content=visible_input,
                    id=configurable.get("user_message_id"),
                    additional_kwargs={
                        **message_metadata,
                        "status": (
                            "provider_input"
                            if configurable.get("message_channel") == "voice"
                            else "final"
                        ),
                        **({"attachments": attachment_refs} if attachment_refs else {}),
                    },
                )
            ]
        ),
        "guardrail": {"safe": True, "pattern": None, "sanitized": result.sanitized},
    }


def completed_tool_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Conserva turnos anteriores sin reenviar tool_use interrumpidos al proveedor."""
    history: list[BaseMessage] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        index += 1
        if isinstance(message, ToolMessage):
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            expected = {call["id"] for call in message.tool_calls}
            results: list[ToolMessage] = []
            while index < len(messages) and isinstance(messages[index], ToolMessage):
                result = messages[index]
                if result.tool_call_id in expected:
                    results.append(result)
                index += 1
            if {result.tool_call_id for result in results} == expected:
                history.extend([message, *results])
            # Una llamada incompleta no es una respuesta utilizable. Conservamos
            # los turnos completos anteriores y permitimos volver a preguntar.
            continue
        history.append(message)
    return history


def _active_tools(tools: list[BaseTool] | None, profile: AgentProfile | None) -> list[BaseTool]:
    """Una única política para enlazar Y ejecutar; [] explícito no hereda nada."""
    selected = (
        profile.tools if profile and profile.tools else (legacy_tools() if tools is None else tools)
    )
    resolved = []
    for item in selected:
        if isinstance(item, BaseTool):
            resolved.append(item)
        elif item in TOOLS_BY_NAME:
            resolved.append(TOOLS_BY_NAME[item])
        else:
            raise ValueError(f"Tool '{item}' desconocida para el perfil activo")
    return patient_tools([*resolved, *(profile.extra_tools if profile else ())])


async def _emit_policy(tool: str, decision: str, code: str | None = None) -> None:
    """Evento de política para la traza del turno (sin argumentos ni resultados)."""
    # Fuera de un run de LangChain no hay a quién notificar: la traza es opcional.
    with suppress(RuntimeError):
        await adispatch_custom_event(
            "vita_tool_policy",
            {"tool": tool if tool in TOOL_SPECS else "unknown", "decision": decision, "code": code},
        )


def _unavailable(call: dict, code: str) -> ToolMessage:
    payload = {"status": "unavailable", "code": code}
    if code == "tool_timeout":
        payload["message"] = "La consulta tardó demasiado. Puedes intentarlo de nuevo."
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        name=call["name"],
        tool_call_id=call["id"],
        status="error",
    )


async def _run_call_with_policy(
    node: ToolNode, state: AgentState, call: dict, timeout_cap: float
) -> tuple[ToolMessage, bool]:
    """Ejecuta una tool con su deadline y reintentos por timeout (solo idempotentes).

    Returns:
        (ToolMessage, timed_out)
    """
    spec = spec_for(call["name"])
    attempts = spec.retries + 1 if spec.idempotent else 1
    expires = asyncio.get_running_loop().time() + min(spec.deadline_seconds, timeout_cap)
    for attempt in range(attempts):
        try:
            remaining = expires - asyncio.get_running_loop().time()
            async with asyncio.timeout(remaining / (attempts - attempt)):
                output = await node.ainvoke(
                    {
                        **state,
                        "messages": [
                            *state["messages"][:-1],
                            state["messages"][-1].model_copy(update={"tool_calls": [call]}),
                        ],
                    }
                )
            return output["messages"][0], False
        except TimeoutError:
            if attempt + 1 < attempts:
                await _emit_policy(call["name"], "retry", "tool_timeout")
    return _unavailable(call, "tool_timeout"), True


def _tool_catalog_failure(
    state: AgentState,
    configurable: dict,
    profile: AgentProfile | None,
    tools: list[BaseTool] | None,
    model: BaseChatModel,
    exc: Exception,
) -> dict:
    """Respuesta controlada cuando las tools del turno no se pueden enlazar."""
    declared = list(profile.tools if profile and profile.tools else (tools or []))
    extra = list(profile.extra_tools) if profile else []
    culprits = conversion_failures([*declared, *extra])
    logger.error(
        "Catálogo de tools no enlazable: perfil=%s canal=%s tools=%s culpables=%s error=%s",
        profile.key if profile else "default",
        configurable.get("message_channel", "text"),
        [tool_label(item) for item in [*declared, *extra]],
        culprits or ["(resolución del perfil)"],
        type(exc).__name__,
    )
    language = detect_turn_language(state) or configurable.get("language")
    response = AIMessage(
        content=(
            "The assistant is temporarily unavailable. Please try again in a few minutes."
            if language == "en"
            else "El asistente no está disponible en este momento. "
            "Inténtalo de nuevo en unos minutos."
        ),
        additional_kwargs={
            "channel": configurable.get("message_channel", "text"),
            "status": "final",
            "error_code": TOOL_CATALOG_ERROR_CODE,
        },
    )
    if configurable.get("execution_id"):
        response.additional_kwargs["execution_id"] = configurable["execution_id"]
    if configurable.get("assistant_message_id"):
        response.id = configurable["assistant_message_id"]
    agent = profile.key if profile else state.get("agent", "base")
    return {
        "messages": [response],
        "provider": model.__class__.__name__,
        "agent": agent,
        "context_agent": agent,
    }


def _budget_after_attachments(total: int, attachment_block: str) -> int:
    """Reserva tokens para los datos adjuntos sin dejar sin espacio a políticas e historial."""
    if not attachment_block:
        return total
    return max(total // 2, total - estimate_tokens(attachment_block) - MESSAGE_OVERHEAD_TOKENS)


def _with_attachment_data(context, attachment_block: str) -> list[BaseMessage]:
    """Coloca los datos adjuntos justo antes de la pregunta vigente del paciente.

    Van como mensaje de usuario delimitado (no de sistema): no heredan autoridad
    de instrucción. No se devuelven en el estado, así que nunca se persisten.
    """
    prompt = list(context.prompt)
    data = HumanMessage(content=attachment_block)
    offset = len(prompt) - len(context.history)
    for index in range(len(context.history) - 1, -1, -1):
        if isinstance(context.history[index], HumanMessage):
            return [*prompt[: offset + index], data, *prompt[offset + index :]]
    return [*prompt, data]


def make_agent_node(
    model: BaseChatModel,
    system_prompt: str = BASE_SYSTEM_PROMPT,
    tools: list[BaseTool] | None = None,
    profile_resolver: Callable[[AgentState], Awaitable[AgentProfile | None]] | None = None,
    final_response: bool = False,
) -> Callable[[AgentState], Awaitable[dict]]:
    """Crea el nodo que invoca al LLM con las tools enlazadas (`bind_tools`).

    Args:
        model: modelo de chat a usar.
        system_prompt: prompt de sistema por defecto (se usa si no hay
            `profile_resolver` o el perfil resuelto no trae prompt propio).
        tools: tools a exponer al modelo; None → todas las registradas.
        profile_resolver: opcional. Callable ASYNC que devuelve el
            `AgentProfile` activo para el estado actual (routing por intención).
            Permite que el nodo cambie de "sombrero" (prompt/tools del perfil)
            en cada turno, leyendo la config de la BD cuando corresponda.

    Returns:
        Callable async que ejecuta un turno del agente.
    """

    async def agent_node(state: AgentState) -> dict:
        # Resolución del perfil activo: el precedente gana sobre el default.
        profile = await profile_resolver(state) if profile_resolver else None
        active_prompt = profile.system_prompt if profile else system_prompt
        configurable = get_config().get("configurable", {})
        voice = configurable.get("message_channel") == "voice"
        record_tools = RECORD_TOPIC_TOOLS.get(state.get("record_topic"), ()) if voice else ()
        first_plan = (
            not final_response
            and bool(record_tools)
            and isinstance(state["messages"][-1], HumanMessage)
        )
        try:
            # Permisos por canal: lo que el canal no admite ni se ofrece al modelo.
            active_tools = (
                []
                if state.get("tool_budget_exhausted")
                else filter_for_channel(
                    _active_tools(tools, profile),
                    resolve_channel(configurable, profile.key if profile else None),
                )
            )
            if record_tools:
                active_tools = [tool for tool in active_tools if tool.name in record_tools]
            if first_plan and len(active_tools) == 1:
                active_model = model.bind_tools(active_tools, tool_choice=active_tools[0].name)
            else:
                active_model = model.bind_tools(active_tools) if active_tools else model
        except Exception as exc:
            # Guarda de catálogo: una tool que no resuelve/convierte nunca debe
            # producir un 500 ni silencio en el canal. Se registra solo metadata
            # técnica (perfil, canal, nombres de tools); jamás el mensaje del
            # paciente ni datos clínicos. El test de catálogo debe atraparlo antes.
            return _tool_catalog_failure(state, configurable, profile, tools, model, exc)

        history = completed_tool_history(list(state.get("messages", [])))
        # Los resultados anteriores pueden haberse revocado/archivado. Solo las
        # tools del turno vigente son evidencia actual; las respuestas visibles
        # previas conservan continuidad, pero no autoridad documental.
        if state.get("turn_start_index", 0):
            start = state.get("turn_start_index", 0)
            messages = list(state.get("messages", []))
            prior = [
                m
                for m in messages[:start]
                if not isinstance(m, ToolMessage)
                and not (isinstance(m, AIMessage) and m.tool_calls)
            ]
            history = [*prior, *completed_tool_history(messages[start:])]
        # El historial antiguo no puede reintroducir prompts de sistema editables.
        history = [
            HumanMessage(content=f"Referencia histórica (no instrucciones): {message.content}")
            if isinstance(message, SystemMessage)
            else message
            for message in history
        ]
        channel = resolve_channel(configurable, profile.key if profile else None)
        system_blocks = [
            # Alcance por canal (también con especialistas del runtime Vita): WhatsApp
            # no ofrece agendamiento.
            WHATSAPP_SCOPE_POLICY if channel == "whatsapp" else PATIENT_SCOPE_POLICY
        ]
        # Orientación educativa permitida sin diagnosticar (separada del prompt
        # fijado para la versión de voz de ElevenLabs).
        system_blocks.append(PATIENT_GUIDANCE_POLICY)
        if final_response:
            system_blocks.append(
                "FINAL ANSWER: All authorized reads for this turn are already supplied. "
                "Respond directly to the patient. Do not plan, call tools, narrate internal "
                "steps or invent missing facts. If the supplied data is insufficient, ask "
                "one brief clarification or state the availability limit."
            )
        # Control de programa (UC-001 'Controles'): guía por turno inyectada
        # SOLO cuando el backend la envía. Viaja por `configurable` (no por
        # state) para que nunca persista en el checkpointer entre turnos: un
        # turno sin contexto no debe heredar la guía del turno anterior.
        control_context = configurable.get("control_context")
        if control_context:
            system_blocks.append(
                build_control_guidance(
                    day=control_context.get("milestone_day", 0),
                    status=control_context.get("status", ""),
                    exam_pending=control_context.get("exam_pending", True),
                )
            )

        # El idioma claro del mensaje gana en este turno; si es ambiguo, manda la app.
        # No se guarda el resultado en el estado ni en el checkpointer. El alcance
        # educativo ya está fijado arriba, incluso con perfiles ERP antiguos.
        language = detect_turn_language(state) or configurable.get("language")
        if language:
            system_blocks.append(build_language_instruction(language))
            # Política de servicio en el idioma vigente: sin derivación a
            # humanos (cubre agentes especializados y sincronizados de BD que
            # no traen la regla 9 del prompt base) — evita que la oferta de
            # un agente humano rompa la consistencia de idioma.
            system_blocks.append(
                build_turn_policy(
                    language,
                    allow_appointment=False,
                )
            )

        if configurable.get("vita_history") is not None:
            if configurable.get("message_channel") != "voice":
                system_blocks.append(TEXT_DELIVERY_POLICY)
            if service_query(state.get("input", "")):
                system_blocks.append(SERVICE_DELIVERY_POLICY)
        if channel == "whatsapp" and not (profile is not None and profile.key == "whatsapp"):
            # El perfil legado ya lo trae en su prompt; los agentes Vita no.
            system_blocks.append(WHATSAPP_FORMAT_POLICY)
        if channel == "whatsapp":
            system_blocks.append(WHATSAPP_DATA_POLICY)
        if not final_response and any(t.name in VITA_CAPTURE_TOOL_NAMES for t in active_tools):
            system_blocks.append(VITA_CAPTURE_POLICY)

        # Adjuntos efímeros (Ola B): política de datos no confiables por turno.
        attachments = configurable.get("vita_attachments")
        attachment_block = getattr(attachments, "data_block", "") if attachments else ""
        if attachment_block:
            system_blocks.append(attachments.policy())

        if configurable.get("message_channel") == "voice":
            # La entrega oral cierra las políticas de sistema; los datos del
            # perfil/documentos no convierten un resumen en una lectura completa.
            system_blocks.append(VOICE_DELIVERY_POLICY)

        context = build_context(
            identity=active_prompt,
            memory=state.get("memory_context") or "",
            experience=state.get("experience_context") or "",
            policies=[SystemMessage(content=block) for block in system_blocks],
            history=history,
            budget=ContextBudget(
                total=_budget_after_attachments(
                    get_settings().vita_context_token_budget, attachment_block
                )
            ),
        )

        prompt = context.prompt
        if attachment_block:
            prompt = _with_attachment_data(context, attachment_block)
        product_reply = application_name_reply(state.get("input", "")) if voice else None
        response = (
            AIMessage(content=product_reply)
            if product_reply is not None
            else await active_model.ainvoke(prompt)
        )
        if (
            first_plan
            and active_tools
            and (
                not response.tool_calls
                or any(call["name"] not in record_tools for call in response.tool_calls)
            )
        ):
            # Un planificador que devuelve texto libre no puede prometer una
            # lectura. El rango/página por defecto sigue siendo decisión de BACK.
            response = AIMessage(
                content="",
                tool_calls=[{"name": active_tools[0].name, "args": {}, "id": f"read-{uuid4()}"}],
            )
        if (final_response or state.get("tool_budget_exhausted")) and response.tool_calls:
            response = AIMessage(
                content=(
                    "No pude completar la consulta. Inténtalo de nuevo."
                    if language != "en"
                    else "I couldn't complete the request. Please try again."
                ),
                additional_kwargs={"error_code": "tool_budget_exceeded"},
            )
        if not response.tool_calls:
            if voice:
                response = response.model_copy(
                    update={
                        "content": format_voice_answer(
                            extract_message_text(response.content),
                            language=language,
                            has_results=voice_turn_has_results(state),
                        )
                    }
                )
            response.additional_kwargs.pop("execution_id", None)
            if configurable.get("execution_id"):
                response.additional_kwargs["execution_id"] = configurable["execution_id"]
            if configurable.get("assistant_message_id"):
                response.id = configurable["assistant_message_id"]
            response.additional_kwargs.update(
                {
                    "channel": configurable.get("message_channel", "text"),
                    "status": "generated"
                    if configurable.get("message_channel") == "voice"
                    else "final",
                }
            )
            if configurable.get("voice_session_id"):
                response.additional_kwargs.update(
                    {
                        "voice_session_id": configurable["voice_session_id"],
                        "source": "vita_runtime",
                    }
                )
        return {
            # Guardrails ya persistió la entrada; no repetirla tras cada herramienta.
            "messages": [response],
            "provider": model.__class__.__name__,
            "agent": profile.key if profile else state.get("agent", "base"),
            "context_agent": profile.key if profile else state.get("agent", "base"),
        }

    return agent_node


def make_memory_load_node(
    service_factory: Callable[[], UserMemoryService] | None = None,
) -> Callable[[AgentState], Awaitable[dict]]:
    """Crea el nodo que carga la memoria del usuario antes del agente.

    Lee `user_id`/`agent_instance_id` del `configurable` de ejecución
    (propagado por el backend) y `agent_type_id` del estado; si no hay
    contexto de usuario, el nodo no hace nada (memoria deshabilitada).

    Args:
        service_factory: fábrica del servicio de memoria (inyectable en tests
            sin BD); por defecto usa la sesión del engine (`async_session`).
    """

    async def memory_load_node(state: AgentState) -> dict:
        configurable = get_config().get("configurable", {})
        # La memoria antigua no tiene thread/canal/revisión. No puede volver
        # a inyectar conversaciones eliminadas ni mezclarlas con ERP.
        if configurable.get("vita_history") is not None:
            return {"memory_context": ""}
        user_id = configurable.get("user_id")
        agent_type_id = state.get("agent")
        if not user_id or not agent_type_id:
            return {}

        service = service_factory() if service_factory is not None else _default_service()
        try:
            context = await service.load_context(
                user_id=user_id,
                agent_type_id=agent_type_id,
                agent_instance_id=configurable.get("agent_instance_id"),
            )
        except Exception:
            # La memoria nunca debe tumbar el chat: sin BD → sin memoria.
            context = ""
        return {"memory_context": context}

    return memory_load_node


def make_memory_save_node(
    service_factory: Callable[[], UserMemoryService] | None = None,
) -> Callable[[AgentState], Awaitable[dict]]:
    """Crea el nodo que persiste hechos y resumen tras el turno del agente.

    Extrae hechos declarativos del mensaje del usuario y rota el resumen
    rodante si el historial lo amerita. Nunca falla el turno por errores de
    persistencia.
    """

    async def memory_save_node(state: AgentState) -> dict:
        configurable = get_config().get("configurable", {})
        if configurable.get("vita_history") is not None:
            return {}
        user_id = configurable.get("user_id")
        agent_type_id = state.get("agent")
        input_text = state.get("input")
        if not user_id or not agent_type_id or not input_text:
            return {}

        service = service_factory() if service_factory is not None else _default_service()
        with suppress(Exception):
            await service.extract_and_save(
                user_id=user_id,
                agent_type_id=agent_type_id,
                agent_instance_id=configurable.get("agent_instance_id"),
                message=input_text,
            )
            # El servicio deja la transacción abierta (commit del llamador);
            # si no se persiste aquí, la sesión se descarta sin guardar nada.
            await service.commit()

        # Historial previo (sin la respuesta del turno) para el resumen rodante.
        history: list[str] = []
        for msg in state.get("messages", []):
            text = getattr(msg, "content", "")
            if isinstance(text, str) and text:
                history.append(text)

        with suppress(Exception):
            last_summary = await service.get_summary(user_id=user_id, agent_type_id=agent_type_id)
            await service.maybe_roll_summary(
                user_id=user_id,
                agent_type_id=agent_type_id,
                agent_instance_id=configurable.get("agent_instance_id"),
                history=history,
                last_summary=last_summary,
            )
            # Persiste también el resumen rodante si se rotó (transacción corta).
            await service.commit()

        return {}

    return memory_save_node


def make_experience_load_node(
    service_factory: Callable[[], AdaptiveMemoryService] | None = None,
) -> Callable[[AgentState], Awaitable[dict]]:
    """Crea el nodo que inyecta la experiencia aprendida del agente.

    Carga los patrones exitosos y recurrentes del `agent_type_id` actual y los
    expone en `experience_context` (bloque SEPARADO de `memory_context`: la
    experiencia pertenece al agente, la memoria al usuario).
    """

    async def experience_load_node(state: AgentState) -> dict:
        if get_config().get("configurable", {}).get("vita_history") is not None:
            return {"experience_context": ""}
        agent_type_id = state.get("agent")
        if not agent_type_id:
            return {}

        service = service_factory() if service_factory is not None else _default_adaptive_service()
        try:
            experiences = await service.load_experiences(agent_type_id=agent_type_id)
            context = service.format_experiences(experiences)
        except Exception:
            context = ""
        return {"experience_context": context}

    return experience_load_node


def _default_service() -> UserMemoryService:
    """Servicio de memoria con sesión corta del engine (producción)."""
    from app.db.engine import async_session
    from app.memory.service import UserMemoryService

    return UserMemoryService(async_session())


def _default_adaptive_service() -> AdaptiveMemoryService:
    """Servicio de adaptive memory con sesión corta del engine (producción)."""
    from app.db.engine import async_session

    return AdaptiveMemoryService(async_session())


def make_tools_node(
    tools: list[BaseTool] | None = None,
    profile_resolver: Callable[[AgentState], Awaitable[AgentProfile | None]] | None = None,
    max_tool_calls: int = 8,
    tool_timeout_seconds: float = 3,
) -> Callable[[AgentState], Awaitable[dict]]:
    """Crea el nodo de ejecución de tools para el subconjunto indicado.

    El wrapper es async a propósito: el `ToolNode` interno es un
    `RunnableCallable` cuyo `__call__` es síncrono; si se expone así, PregelNode
    lo ejecuta por la ruta sync (`_execute_tool_sync`) y las tools async
    (p. ej. `retrieve_knowledge`) fallan con "StructuredTool does not support
    sync invocation". Con `ainvoke` se usa `_execute_tool_async` y las tools
    coroutine funcionan.

    Además de ejecutar las tools, captura las sugerencias estructuradas
    emitidas por `suggest_appointment` (ver `_extract_suggestions`) y las
    acumula en `state["suggestions"]`.
    """

    async def tools_node(state: AgentState) -> dict:
        last = state.get("messages", [None])[-1]
        calls = getattr(last, "tool_calls", [])
        used = state.get("tool_call_count", 0)
        # Nunca ejecutar un lote que sobrepasa el presupuesto. Responder a cada
        # tool_use mantiene el historial válido incluso tras timeout/interrupción.
        exhausted = used + len(calls) > max_tool_calls
        if state.get("tool_budget_exhausted"):
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "No pude completar la consulta con los límites disponibles. "
                            "Puedes intentarlo de nuevo. / Please try again."
                        )
                    )
                ]
            }
        profile = await profile_resolver(state) if profile_resolver else None
        channel = resolve_channel(
            get_config().get("configurable", {}), profile.key if profile else None
        )
        # Misma política que al enlazar: una tool fuera del canal no se ejecuta
        # aunque el modelo la pida (p. ej. argumentos/tool_calls manipulados).
        selected = filter_for_channel(_active_tools(tools, profile), channel)
        record_tools = (
            RECORD_TOPIC_TOOLS.get(state.get("record_topic"), ()) if channel == "voice" else ()
        )
        if record_tools:
            selected = [tool for tool in selected if tool.name in record_tools]
        allowed = {tool.name for tool in selected}
        # Solo se admiten los campos públicos del esquema. Los selectores de
        # paciente siguen excluidos de cualquier lectura.
        invalid_arguments = any(
            (call["name"].startswith("get_my_") or call["name"] in VITA_TOOL_ARGUMENTS)
            and set(call.get("args", {})).difference(VITA_TOOL_ARGUMENTS.get(call["name"], ()))
            for call in calls
        )
        blocked = (
            exhausted or invalid_arguments or any(call["name"] not in allowed for call in calls)
        )
        executed: list[str] = []
        if blocked:
            reason = "tool_budget_exceeded" if exhausted else "tool_not_authorized"
            for call in calls:
                await _emit_policy(call["name"], "blocked", reason)
            result = {"messages": [_unavailable(call, reason) for call in calls]}
        else:
            # Presupuesto por tool dentro del turno (además del global del grafo).
            start = state.get("turn_start_index", 0)
            counts = Counter(
                message.name
                for message in state.get("messages", [])[start:]
                if isinstance(message, ToolMessage)
            )
            runnable, denied = [], {}
            for call in calls:
                name = call["name"]
                counts[name] += 1
                if counts[name] > spec_for(name).max_calls_per_turn:
                    reason = "tool_budget_exceeded"
                    denied[call["id"]] = _unavailable(call, reason)
                    await _emit_policy(name, "blocked", reason)
                    continue
                runnable.append(call)
            node = ToolNode(selected)
            outcomes = await asyncio.gather(
                *(
                    _run_call_with_policy(node, state, call, tool_timeout_seconds)
                    for call in runnable
                )
            )
            done = {
                call["id"]: message for call, (message, _) in zip(runnable, outcomes, strict=True)
            }
            # Un timeout agotado (tras reintentos) cierra el presupuesto del turno.
            exhausted = not runnable or any(timed_out for _, timed_out in outcomes)
            executed = [call["name"] for call in runnable]
            result = {"messages": [done.get(call["id"]) or denied[call["id"]] for call in calls]}
        actions, sources = [], []
        if not blocked:
            from app.observability.executions import extract_rag_sources
            from app.vita.contracts import ContextEnvelope

            sources.extend(extract_rag_sources(result.get("messages", [])))
            for message in result.get("messages", []):
                if getattr(message, "name", "").startswith("get_my_"):
                    with suppress(ValueError, TypeError):
                        envelope = ContextEnvelope.model_validate_json(message.content)
                        if envelope.status == "ok":
                            actions.extend(a.model_dump(mode="json") for a in envelope.actions)
                            sources.extend(s for s in envelope.sources if isinstance(s, dict))
        topics = {RECORD_TOOL_TOPICS[name] for name in executed if name in RECORD_TOOL_TOPICS}
        return {
            **result,
            **({"record_topic": topics.pop()} if channel == "voice" and len(topics) == 1 else {}),
            "tool_call_count": used + len(calls),
            "tool_budget_exhausted": blocked or exhausted or used + len(calls) >= max_tool_calls,
            "tools_used": executed,
            "suggestions": [],
            "app_actions": actions,
            "data_sources": sources,
        }

    return tools_node


def _extract_suggestions(messages: list) -> list[dict]:
    """Extrae las sugerencias estructuradas de los ToolMessages del turno.

    `suggest_appointment` devuelve un JSON serializado; aquí se parsea cada
    ToolMessage de esa tool y se devuelve la lista de dicts para acumular en
    el estado. Los mensajes malformados se descartan en silencio (nunca deben
    tumbar el turno).
    """
    suggestions: list[dict] = []
    for msg in messages:
        if getattr(msg, "name", None) != "suggest_appointment":
            continue
        content = getattr(msg, "content", "")
        if not isinstance(content, str) or not content:
            continue
        with suppress(Exception):
            data = json.loads(content)
            if isinstance(data, dict):
                suggestions.append(data)
    return suggestions


async def tools_node(state: AgentState) -> dict:
    """Nodo por defecto con todas las tools registradas."""
    return await make_tools_node()(state)
