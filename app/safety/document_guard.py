"""Neutralización de texto extraído de documentos (Ola B, 6.5).

El texto de un adjunto es DATO, nunca instrucción. Este módulo:

1. Elimina caracteres de control y cualquier secuencia que imite los
   delimitadores del bloque de datos.
2. Detecta frases con forma de instrucción dirigida al asistente (EN/ES),
   reutilizando los patrones de entrada y añadiendo patrones propios de
   documentos (etiquetas de rol, exfiltración del prompt o de otros pacientes).
3. Sustituye cada coincidencia por una marca neutra, de modo que el modelo no
   reciba la directiva literal aunque el resto del documento siga disponible.
4. Envuelve el resultado en un bloque con delimitadores que incluyen un nonce
   aleatorio por turno: el documento no puede "cerrar" el bloque.

Nada de este módulo registra contenido: solo devuelve los nombres estáticos de
los patrones detectados para que el llamador los registre como metadatos.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass

from app.safety.guardrails import INJECTION_PATTERNS

REDACTION_MARK = "[texto con forma de instrucción omitido]"

DOCUMENT_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    *INJECTION_PATTERNS,
    (
        "reveal prompt or secrets",
        re.compile(
            r"(reveal|show|print|output|leak|repeat)\s+(me\s+)?(the\s+|your\s+)?"
            r"(system\s+prompt|hidden\s+instructions?|instructions?|api[\s_-]*keys?|"
            r"passwords?|secrets?|other\s+patients?)",
            re.I,
        ),
    ),
    (
        "revelar prompt o datos (es)",
        re.compile(
            r"(revela|muestra|imprime|filtra|repite|dime)\s+(me\s+)?(el\s+|los\s+|las\s+|tu\s+|"
            r"tus\s+)?(prompt|instrucciones|claves?|contraseñas?|secretos?|"
            r"datos\s+de\s+otros?\s+pacientes?|expedientes?\s+de\s+otros?)",
            re.I,
        ),
    ),
    (
        "role tag",
        re.compile(
            r"<\s*/?\s*(system|assistant|user|instructions?|im_start|im_end)\s*>"
            r"|^\s*(system|assistant|developer)\s*:",
            re.I | re.M,
        ),
    ),
    (
        "addressed to ai",
        re.compile(
            r"\b(ai|assistant|chatbot|model|llm|claude|gpt)\s*[,:]\s*"
            r"(you\s+must|ignore|forget|disregard|from\s+now\s+on)",
            re.I,
        ),
    ),
    (
        "dirigido al asistente (es)",
        re.compile(
            r"\b(asistente|modelo|ia|vita|chatbot)\s*[,:]\s*"
            r"(debes|tienes\s+que|ignora|olvida|a\s+partir\s+de\s+ahora)",
            re.I,
        ),
    ),
    (
        "tool or action request",
        re.compile(
            r"\b(call|invoke|execute|run)\s+(the\s+)?(tool|function|command)\b"
            r"|\b(llama|invoca|ejecuta)\s+(a\s+)?(la\s+)?(herramienta|funci[oó]n|comando)\b",
            re.I,
        ),
    ),
]

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F​-‏‪-‮⁦-⁩]")
# Cualquier intento de reproducir los delimitadores del bloque de datos.
_DELIMITER_LIKE = re.compile(r"<{2,}|>{2,}|\b(?:FIN_)?ADJUNTO_[0-9A-Za-z]+\b")


@dataclass(frozen=True)
class NeutralizedText:
    """Texto listo para entrar al prompt como dato y patrones detectados."""

    text: str
    patterns: tuple[str, ...]

    @property
    def injection_detected(self) -> bool:
        return bool(self.patterns)


def neutralize_document_text(text: str, *, max_chars: int) -> NeutralizedText:
    """Limpia, recorta y neutraliza directivas embebidas en texto extraído."""
    cleaned = _CONTROL_CHARS.sub("", text.replace("\r\n", "\n"))
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = _DELIMITER_LIKE.sub(" ", cleaned)
    detected: list[str] = []
    for name, pattern in DOCUMENT_INJECTION_PATTERNS:
        cleaned, count = pattern.subn(REDACTION_MARK, cleaned)
        if count:
            detected.append(name)
    cleaned = cleaned.strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "\n[contenido recortado por longitud]"
    return NeutralizedText(cleaned, tuple(detected))


def new_turn_nonce() -> str:
    """Nonce impredecible para los delimitadores de un turno."""
    return secrets.token_hex(8)


def wrap_data_block(*, nonce: str, index: int, header: str, body: str) -> str:
    """Delimita un adjunto. El nonce impide que el documento falsifique el cierre."""
    return f"<<<ADJUNTO_{nonce} n={index} {header}>>>\n{body}\n<<<FIN_ADJUNTO_{nonce} n={index}>>>"
