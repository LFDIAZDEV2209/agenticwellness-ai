"""Contexto de una intervención hablada fragmentada por endpointing/tool calls."""

from app.voice.language import latest_user_index


def compose_voice_input(messages) -> str:
    """Une fragmentos reales sin tratar una tool de idioma como respuesta hablada.

    La identidad y el ordinal siguen usando el historial original, sin borrar
    mensajes ni relajar sus hashes. Solo enriquece el input de routing/retrieval;
    el mensaje visible conserva el último fragmento original.
    """
    index = latest_user_index(messages)
    if index < 0:
        return ""
    parts = []
    for message in reversed(messages[: index + 1]):
        if (
            message.role == "assistant"
            and (message.content or "").strip()
            and not message.tool_calls
        ):
            break
        if message.role == "user" and (message.content or "").strip():
            parts.insert(0, message.content.strip())
    combined = " ".join(parts)
    # Nunca truncar el fragmento actual para hacer encajar contexto anterior.
    return combined if len(combined) <= 8000 else (messages[index].content or "")
