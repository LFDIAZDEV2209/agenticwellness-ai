"""Formato del hilo y emisión por frases de la respuesta de voz."""

import re

from langchain_core.messages import ToolMessage

from app.vita.routing import normalize

_BULLET = re.compile(r"^\s*(?:[-+*•]|\d+[.)])\s+(.*)$")
_HEADING = re.compile(r"^\s*(?:#{1,6}\s|[-*_]{3,}\s*$)")
_TABLE_RULE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
_SENTENCE_END = re.compile(
    r"(?<!\bSr\.)(?<!\bSr[a.]\.)(?<!\bDr\.)(?<!\bDr[a.]\.)(?<!\bMr\.)(?<!\bMrs\.)"
    r"(?<!\bMs\.)(?<!\bp\.)(?<!\ba\.)(?<!\be\.g\.)(?<!\bi\.e\.)"
    r"([.!?…]+[\"”\u2019»)]*)(?:\s+|$)",
    re.IGNORECASE,
)
_CLOCK_ABBREVIATION = re.compile(r"\b[ap]\.m$", re.IGNORECASE)
_SAFETY = re.compile(
    r"\b(?:llama|llame|llamar|contacta|contacte|contactar|marca|marque|acude|acuda|ve|"
    r"busca|busque|call|dial|contact|go to|seek|get)\b[^.!?]{0,85}"
    r"\b(?:emergencias?|emergency|urgencias|urgent medical (?:care|help)|sos|ambulance|"
    r"immediate medical attention|atencion medica (?:inmediata|urgente)|help immediately|"
    r"ambulancia|poison control|centro de toxicologia|crisis (?:line|hotline)|"
    r"linea de crisis|911|112|123)\b"
)
_DETAIL = re.compile(r"\b(?:chat|por escrito|in writing)\b", re.IGNORECASE)
_INLINE_MARKUP = re.compile(r"(?<!\w)_(?=\S)|(?<=\S)_(?!\w)|[*`~]")
_STREAM_MARKUP = re.compile(r"^\s*(?:[#>|]|[-+*•]\s|[-*_`]{3}|\d+[.)]\s)|\[", re.M)
_RESULT_WORD_LIMIT = 40
_GENERAL_WORD_LIMIT = 50
_SENTENCE_LIMIT = 3


def voice_turn_has_results(state: dict) -> bool:
    # Un ToolMessage de error también cuenta, aunque ninguna tool se haya ejecutado.
    start = state.get("turn_start_index", 0)
    return bool(state.get("tools_used")) or any(
        isinstance(message, ToolMessage) for message in state.get("messages", [])[start:]
    )


def _plain_text(line: str) -> str:
    line = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", line)
    line = re.sub(r"^\s*(?:>\s*)+", "", line)
    line = _INLINE_MARKUP.sub("", line)
    return " ".join(line.split())


def _sentences(text: str, *, include_tail: bool = True, final: bool = True) -> list[str]:
    sentences = []
    start = 0
    for end in _SENTENCE_END.finditer(text):
        # Un punto al fin de un chunk puede ser el comienzo de un decimal.
        if not final and end.end(1) == len(text):
            break
        if _CLOCK_ABBREVIATION.search(text, 0, end.start(1)):
            following = text[end.end() : end.end() + 1]
            if following.islower() or (not following and not final):
                continue
        sentences.append(text[start : end.end(1)].strip())
        start = end.end()
    if include_tail and (tail := text[start:].strip()):
        sentences.append(tail)
    return sentences


def _details_offer(language: str | None) -> str:
    return "More details in chat." if language == "en" else "Más detalles por chat."


def _continuation(sentences: list[str], offer: str) -> list[str]:
    return (
        []
        if any("?" in sentence or _DETAIL.search(sentence) for sentence in sentences)
        else [offer]
    )


def _fits(sentences: list[str], word_limit: int) -> bool:
    return (
        len(sentences) <= _SENTENCE_LIMIT and sum(len(s.split()) for s in sentences) <= word_limit
    )


def _bounded_prefix(
    sentences: list[str], limit: int, *, reserve_offer: str | None = None
) -> list[str]:
    prefix = []
    for sentence in sentences:
        candidate = [*prefix, sentence]
        continuation = _continuation(candidate, reserve_offer) if reserve_offer else []
        if not _fits([*candidate, *continuation], limit):
            break
        prefix.append(sentence)
    return prefix


class VoiceSentenceEmitter:
    """Emite un prefijo estable y completa el presupuesto con el mensaje guardado."""

    def __init__(self, language: str | None):
        self._offer = _details_offer(language)
        self._buffer = ""
        self._emitted = ""
        self._deferred = False

    def feed(self, fragment: str) -> str:
        if self._deferred:
            return ""
        self._buffer += fragment
        text = "\n".join(self._buffer.splitlines())
        if self._buffer and self._buffer[-1].isspace():
            text += " "
        markup = _STREAM_MARKUP.search(text)
        plain = _INLINE_MARKUP.sub("", text[: markup.start()] if markup else text)
        sentences = _sentences(re.sub(r"\s+", " ", plain).lstrip(), include_tail=False, final=False)
        # Una lista posterior puede reducir el presupuesto a 40 palabras. El
        # formato final también puede necesitar una frase para ofrecer el detalle.
        prefix = _bounded_prefix(sentences, _RESULT_WORD_LIMIT, reserve_offer=self._offer)
        self._deferred = (
            bool(markup) or len(prefix) < len(sentences) or len(prefix) == _SENTENCE_LIMIT
        )
        return self._new_text(" ".join(prefix))

    def finish(self, saved: str, *, has_results: bool, allow_standalone: bool = False) -> str:
        # El checkpoint ya está formateado. Solo se habla su prefijo completo
        # que quepa, incluso cuando el hilo conserva instrucciones urgentes largas.
        sentences = _sentences(saved, include_tail=False)
        if not sentences and allow_standalone and saved.strip():
            sentences = [saved.strip()]
        limit = _RESULT_WORD_LIMIT if has_results else _GENERAL_WORD_LIMIT
        prefix = " ".join(_bounded_prefix(sentences, limit))
        return self._new_text(prefix) if saved.startswith(prefix) else ""

    def _new_text(self, prefix: str) -> str:
        if self._emitted and not prefix.startswith(self._emitted + " "):
            return ""
        delta = prefix[len(self._emitted) :]
        self._emitted = prefix
        return delta


def format_voice_answer(text: str, *, language: str | None, has_results: bool) -> str:
    """Conserva frases completas y da preferencia a los primeros datos del modelo."""
    parts = []
    items = []
    item_count = 0
    lines = text.splitlines()
    urgent = bool(_SAFETY.search(normalize(text)))
    for index, line in enumerate(lines):
        if urgent:
            line = re.sub(r"^\s*#{1,6}\s+", "", line)
        if not line.strip() or _HEADING.match(line) or line.strip().startswith("```"):
            continue
        if _TABLE_RULE.match(line) or (
            index + 1 < len(lines)
            and line.lstrip().startswith("|")
            and _TABLE_RULE.match(lines[index + 1])
        ):
            continue
        bullet = _BULLET.match(line)
        table_row = line.strip().startswith("|") and line.strip().endswith("|")
        if bullet or table_row:
            item_count += 1
            item = (
                bullet[1]
                if bullet
                else ", ".join(cell.strip() for cell in line.strip().strip("|").split("|"))
            )
            items.append(_plain_text(item).rstrip(".;,"))
        else:
            if items:
                parts.append(items)
                items = []
            parts.append(_plain_text(line))
    if items:
        parts.append(items)

    def spoken(max_items: int) -> str:
        chunks = []
        for part in parts:
            if isinstance(part, str):
                chunks.append(part)
            elif part and max_items:
                lead = part[:max_items]
                max_items -= len(lead)
                chunks.append(", ".join(lead) + ".")
        return " ".join(filter(None, chunks))

    full = spoken(item_count)
    if urgent or _SAFETY.search(normalize(full)):
        return full
    word_limit = _RESULT_WORD_LIMIT if has_results or item_count else _GENERAL_WORD_LIMIT
    offer = _details_offer(language)
    for max_items in range(min(item_count, 5), -1, -1):
        sentences = _sentences(spoken(max_items))
        if _fits(sentences, word_limit):
            if item_count == max_items:
                return " ".join(sentences)
            if _fits([*sentences, offer], word_limit):
                break

    summary = _bounded_prefix(sentences, word_limit, reserve_offer=offer)
    return " ".join([*summary, *_continuation(summary, offer)])
