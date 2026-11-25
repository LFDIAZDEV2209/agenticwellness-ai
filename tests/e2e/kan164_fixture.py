"""Fixture local: rutas ASGI reales, proveedor y dependencias externas simulados."""

from unittest.mock import AsyncMock, Mock

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.api.deps import get_db_session, get_graph
from app.api.main import create_app
from app.api.routes import chat
from app.core.config import get_settings
from app.graph.graph import build_graph
from tests.fakes import FakeToolAwareModel


class Provider(FakeToolAwareModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = messages[-1].content
        if "interrupción sintética" in text:
            raise RuntimeError("SYNTHETIC PROVIDER DETAIL")
        if "consejo para mi rutina" in text:
            answer = "Puedes anotar un hábito que te guste."
        elif "Objetivo de energía" in text:
            answer = "Tu objetivo será de 1500 kcal al día."
        else:
            answer = "Haz una pausa consciente. La dosis indicada es 50 mg."
        self.responses = [AIMessage(content=answer)]
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def create(patch=setattr):
    settings = get_settings()
    patch(settings, "vita_runtime_enabled", True)
    patch(settings, "vita_turn_log_enabled", False)
    patch(settings, "internal_api_key", "kan164-local-fixture")
    model = Provider(responses=[AIMessage(content="Puedes anotar un hábito que te guste.")])
    graph = build_graph(model=model, checkpointer=MemorySaver(), tools=[])
    app = create_app()
    app.dependency_overrides[get_graph] = lambda: graph
    app.dependency_overrides[get_db_session] = lambda: Mock(
        get=AsyncMock(return_value=None), commit=AsyncMock()
    )
    patch(chat, "_prepare_chat_history", AsyncMock(return_value=None))
    patch(chat, "_resolve_graph", AsyncMock(return_value=(graph, None, None, None, None)))
    return app
