"""Regresión del P0 de WhatsApp (12-11-2026): catálogo de tools convertible.

La revisión `cooppadresd-ai:49` enlazaba el perfil `whatsapp` con nombres `str`
(`"calculate"`) y el proveedor respondía `ValueError: Unsupported function`
→ `POST /api/v1/chat 500`. Estos tests convierten TODO el catálogo de cada
perfil con los conversores reales de Anthropic/OpenAI (no con fakes) y
verifican que, si aun así una tool no enlaza, el turno responde un mensaje
seguro sin 500 y sin registrar el texto del paciente.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agents.registry import AGENTS, AgentProfile, get_agent_profile
from app.core.errors import ToolCatalogError
from app.graph.graph import build_graph
from app.graph.nodes import TOOL_CATALOG_ERROR_CODE
from app.tools.catalog import (
    PROVIDER_CONVERTERS,
    conversion_failures,
    ensure_convertible,
    profile_tool_catalog,
    validate_tool_catalog,
)
from app.tools.patient_context import get_patient_data
from app.tools.retrieval import make_retrieve_tool
from tests.fakes import FakeToolAwareModel
from tests.test_api import HEADERS, FakeDbSession

CATALOG = profile_tool_catalog()


def test_catalogo_cubre_todos_los_perfiles_y_tools_registradas():
    assert {f"profile:{key}" for key in AGENTS} <= set(CATALOG)
    assert [t.name for t in CATALOG["profile:whatsapp"]] == ["calculate", "get_current_time"]


@pytest.mark.parametrize("provider", sorted(PROVIDER_CONVERTERS))
@pytest.mark.parametrize("context", sorted(CATALOG))
def test_cada_tool_del_perfil_convierte_al_esquema_del_proveedor(context, provider):
    convert = PROVIDER_CONVERTERS[provider]
    for tool in CATALOG[context]:
        try:
            schema = convert(tool)
        except Exception as exc:
            pytest.fail(
                f"La tool '{tool.name}' de {context} no convierte a {provider}: "
                f"{type(exc).__name__}: {exc}"
            )
        assert schema, f"{tool.name} produjo un esquema vacío para {provider}"


@pytest.mark.parametrize("key", sorted(AGENTS))
def test_lo_que_recibe_bind_tools_convierte_para_cada_perfil(key):
    """Ruta real del bug: `_active_tools` es lo que el nodo pasa a `bind_tools`.

    En la imagen :49 el perfil whatsapp llegaba como `("calculate", ...)` sin
    resolver; aquí cualquier `str` o tool rota falla nombrando la culpable.
    """
    from app.graph.nodes import _active_tools

    selected = _active_tools(None, get_agent_profile(key))
    assert selected, f"perfil {key!r} sin tools"
    failures = conversion_failures(selected)
    assert not failures, f"perfil {key!r}: tools no convertibles -> {failures}"


def test_tools_dinamicas_del_resolver_convierten():
    """get_patient_data y retrieve_knowledge se inyectan fuera de AGENTS."""
    ensure_convertible(
        [get_patient_data, make_retrieve_tool(["kb-test"], 3)], context="extra_tools"
    )


def test_validacion_de_arranque_pasa_con_el_catalogo_actual():
    validate_tool_catalog()


def test_nombre_sin_resolver_falla_nombrando_la_tool_culpable():
    with pytest.raises(ToolCatalogError, match=r"'calculate' \(str sin resolver\)"):
        ensure_convertible(["calculate"], context="perfil 'whatsapp'")
    assert len(conversion_failures(["calculate"])) == len(PROVIDER_CONVERTERS)


def test_perfil_con_tool_no_registrada_falla_en_validacion(monkeypatch):
    broken = AgentProfile(
        key="roto", name="x", description="x", system_prompt="x", tools=("tool_inexistente",)
    )
    monkeypatch.setitem(AGENTS, "roto", broken)
    with pytest.raises(ToolCatalogError, match="tool_inexistente"):
        validate_tool_catalog()


class _UnsupportedBindModel(FakeToolAwareModel):
    """Reproduce el error exacto del proveedor en la imagen :49."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        raise ValueError("Unsupported function\n\ncalculate")


async def _turn(model, profile, text="Hola, mi glucosa salió en 180"):
    async def resolve(_):
        return profile

    graph = build_graph(model=model, checkpointer=MemorySaver(), profile_resolver=resolve)
    return await graph.ainvoke(
        {"input": text, "messages": []},
        config={
            "configurable": {
                "thread_id": f"catalog-guard-{profile.key}",
                "message_channel": "whatsapp",
                "execution_id": "exec-1",
            }
        },
    )


async def test_guarda_responde_mensaje_seguro_si_bind_tools_falla(caplog):
    model = _UnsupportedBindModel(responses=[AIMessage(content="no debe usarse")])
    with caplog.at_level(logging.ERROR, logger="app.graph.nodes"):
        result = await _turn(model, get_agent_profile("whatsapp"))

    answer = result["messages"][-1]
    assert isinstance(answer, AIMessage)
    assert "no está disponible" in answer.content
    assert answer.additional_kwargs["error_code"] == TOOL_CATALOG_ERROR_CODE
    assert answer.additional_kwargs["execution_id"] == "exec-1"
    assert not answer.tool_calls
    # Log técnico con perfil, canal y tools; nunca el mensaje del paciente.
    log = "\n".join(record.getMessage() for record in caplog.records)
    assert "perfil=whatsapp" in log and "canal=whatsapp" in log and "calculate" in log
    assert "glucosa" not in log and "180" not in log


async def test_guarda_cubre_tool_desconocida_en_perfil_de_bd():
    profile = AgentProfile(
        key="db-agent", name="x", description="x", system_prompt="x", tools=("tool_inexistente",)
    )
    model = FakeToolAwareModel(responses=[AIMessage(content="no debe usarse")])
    result = await _turn(model, profile, text="Hello, could you please explain my plan for today?")

    answer = result["messages"][-1]
    assert answer.additional_kwargs["error_code"] == TOOL_CATALOG_ERROR_CODE
    assert answer.content.startswith("The assistant is temporarily unavailable")


@pytest.fixture
def http_client():
    from fastapi.testclient import TestClient

    from app.api.deps import get_db_session, get_graph
    from app.api.main import create_app

    app = create_app()
    fake = FakeToolAwareModel(responses=[AIMessage(content="no debe usarse")])
    app.dependency_overrides[get_graph] = lambda: build_graph(
        model=fake, checkpointer=MemorySaver()
    )
    app.dependency_overrides[get_db_session] = FakeDbSession
    with TestClient(app) as test_client:
        yield test_client


def test_chat_http_no_devuelve_500_si_las_tools_no_enlazan(http_client, monkeypatch):
    """Borde HTTP: el canal recibe 200 con mensaje seguro en vez de 500."""
    import app.graph.nodes as nodes

    def broken(*_args, **_kwargs):
        raise ValueError("Unsupported function\n\ncalculate")

    monkeypatch.setattr(nodes, "_active_tools", broken)
    response = http_client.post(
        "/api/v1/chat",
        json={"message": "Hola", "thread_id": "t-guard"},
        headers=HEADERS,
    )
    assert response.status_code == 200, response.text
    assert "no está disponible" in response.json()["answer"]
