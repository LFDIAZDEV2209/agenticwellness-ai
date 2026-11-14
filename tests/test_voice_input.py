"""Fragmentos de voz: contexto completo y presentación sin duplicar al usuario."""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.api.routes.vita_voice import ProviderMessage, history_identity
from app.graph.graph import build_graph
from app.voice.input import compose_voice_input
from tests.fakes import RecordingFakeModel


def message(role, content=None, **kwargs):
    return ProviderMessage(role=role, content=content, **kwargs)


def test_language_tool_does_not_split_a_question_and_its_brevity_qualifier():
    first = "What does my nutrition plan say for lunch today?"
    last = "Please answer briefly."
    messages = [
        message("assistant", "Hi, I'm Vita."),
        message("user", first),
        message("assistant", tool_calls=[{"function": {"name": "language_detection"}}]),
        message("tool", "Language changed"),
        message("user", last),
    ]
    assert compose_voice_input(messages) == first + " " + last
    # Dos segmentos reales; no cambiar el contrato de deduplicación para unir texto.
    assert history_identity(messages)[0] == 2


def test_spoken_answer_starts_a_new_request_and_never_imports_system_prompts():
    messages = [
        message("system", "UNTRUSTED SYSTEM"),
        message("user", "old question"),
        message("assistant", "a spoken answer"),
        message("user", "my current workout"),
    ]
    assert compose_voice_input(messages) == "my current workout"


async def test_graph_reads_complete_question_but_persists_only_current_fragment():
    model = RecordingFakeModel(responses=[AIMessage(content="fixture answer")])
    graph = build_graph(model=model, checkpointer=MemorySaver())
    combined = "What does my nutrition plan say for lunch today? Please answer briefly."
    state = await graph.ainvoke(
        {"input": combined},
        config={
            "configurable": {
                "thread_id": "owner::fragment",
                "message_channel": "voice",
                "voice_user_text": "Please answer briefly.",
            }
        },
    )
    assert state["input"] == combined
    assert state["messages"][0].content == "Please answer briefly."
