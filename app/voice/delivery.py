"""Formato hablado de la respuesta de voz: natural, breve y sin perder datos.

Esta capa es solo de experiencia de usuario. No decide qué responder ni llama
herramientas: recibe el texto que produjo el modelo, lo limpia de marcas
visuales, humaniza fechas y números, y lo entrega en frases completas.

Reglas de diseño:
- Nunca se descarta el dato principal: si el modelo devolvió una lista, se
  conservan los dos primeros elementos con sus valores.
- Nada de ofertas fijas tipo «Más detalles por chat»: la conversación continúa
  de forma natural en el siguiente turno.
- Fechas y horas se leen como las diría una persona («18 de noviembre», «ayer»).
- El presupuesto (3 frases; 40 palabras con resultados, 50 sin ellos) y el
  streaming por frases se mantienen para no afectar la latencia.
"""

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
_INLINE_MARKUP = re.compile(r"(?<!\w)_(?=\S)|(?<=\S)_(?!\w)|[*`~]")
_STREAM_MARKUP = re.compile(r"(?:^|\s)(?:[-+*•]\s|\d+[.)]\s|#{1,6}\s|\|)|\[", re.M)
_RESULT_WORD_LIMIT = 40
_GENERAL_WORD_LIMIT = 50
_SENTENCE_LIMIT = 3

_ES_MONTHS = {
    "ene": "enero",
    "feb": "febrero",
    "mar": "marzo",
    "abr": "abril",
    "may": "mayo",
    "jun": "junio",
    "jul": "julio",
    "ago": "agosto",
    "sep": "septiembre",
    "sept": "septiembre",
    "oct": "octubre",
    "nov": "noviembre",
    "dic": "diciembre",
}
_EN_MONTHS = {
    "jan": "January",
    "feb": "February",
    "mar": "March",
    "apr": "April",
    "may": "May",
    "jun": "June",
    "jul": "July",
    "aug": "August",
    "sep": "September",
    "sept": "September",
    "oct": "October",
    "nov": "November",
    "dec": "December",
}
_ES_MONTH_ABBR = re.compile(
    r"\b(\d{1,2})\s+(?:de\s+)?(ene|feb|mar|abr|may|jun|jul|ago|sept?|oct|nov|dic)\.?\b",
    re.IGNORECASE,
)
_EN_MONTH_ABBR = re.compile(
    r"\b(ene|feb|mar|abr|may|jun|jul|ago|sept?|oct|nov|dic)\.?\s+(\d{1,2})\b",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_THOUSANDS_ES = re.compile(r"(?<=\d),(?=\d{3}\b)")
_RELATIVE_DATE = re.compile(
    r"^(Ayer|Hoy|Anoche|Yesterday|Today|Last night),\s+\d{1,2}\s+de\s+\w+:?"
    r"|^(Ayer|Hoy|Anoche|Yesterday|Today|Last night),\s+\w+\s+\d{1,2}\b:?",
    re.IGNORECASE,
)
_CHAT_OFFER = re.compile(
    r"^(?:(?:más|mas) detalles|más información|more details|more info|"
    r"puedes verlo|you can see it).{0,40}\b(?:chat|escrito)\b[.!?]?$"
    r"|\b(?:por|in|by) chat[.!?]?$",
    re.IGNORECASE,
)
_METRIC_SEGMENT = re.compile(
    r"\d[\d.,]*\s+(?:calor\w*|kcal|pasos?|steps?|metros?|meters?|km|kilometros?|"
    r"kilometers?|bpm|lpm|mmhg|kg|kilos?|libras?|pounds?|horas?|hours?|minutos?|minutes?|%)\b",
    re.IGNORECASE,
)


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


def _month_es(abbr: str) -> str:
    return _ES_MONTHS.get(abbr.lower().rstrip("."), abbr)


def _month_en(abbr: str) -> str:
    return _EN_MONTHS.get(abbr.lower().rstrip("."), abbr)


def _humanize(text: str, language: str) -> str:
    """Fechas, horas y números como los diría una persona."""
    clean = text
    # La hora entre paréntesis es metadato del expediente: no se dice.
    clean = re.sub(r"\s*\(\d{1,2}:\d{2}\)", "", clean)
    # «54.17 (alto)» → «54.17, alto»: paréntesis cortos se leen como aclaración.
    clean = re.sub(r"\s*\(([^)]{1,40})\)", r", \1", clean)
    if language == "en":
        clean = _ISO_DATE.sub(lambda m: f"{_month_en(m.group(2))} {int(m.group(3))}", clean)
        clean = _EN_MONTH_ABBR.sub(lambda m: f"{_month_en(m.group(1))} {m.group(2)}", clean)
    else:
        clean = _ISO_DATE.sub(lambda m: f"{int(m.group(3))} de {_month_es(m.group(2))}", clean)
        clean = _ES_MONTH_ABBR.sub(lambda m: f"{m.group(1)} de {_month_es(m.group(2))}", clean)
        clean = _THOUSANDS_ES.sub(".", clean)
    # «Ayer, 18 de noviembre:» → «Ayer,» (el dato redundante no se dice).
    clean = _RELATIVE_DATE.sub(lambda m: (m.group(1) or m.group(2)) + ",", clean)
    clean = re.sub(r"::+", ":", clean)
    clean = re.sub(r"(?<=\d)\.0\b", "", clean)
    clean = re.sub(r"\s+([.,;:!?])", r"\1", clean)
    clean = re.sub(r"([:;,\u2013\u2014-])\s*$", "", clean.strip())
    return " ".join(clean.split())


def _trim_metric_list(sentence: str, language: str) -> str:
    """Tres o más métricas seguidas suenan a lectura de tabla.

    Conserva las dos primeras («2.306 calorías y 1.806 pasos») y descarta el
    resto. Solo actúa cuando todos los segmentos son número + unidad.
    """
    # Protege los separadores de miles («2,306») para que el split no rompa
    # el número en dos segmentos.
    protected = re.sub(r"(?<=\d),(?=\d{3}\b)", "\u0000", sentence.rstrip("."))
    parts = [part.replace("\u0000", ",").strip() for part in protected.split(",")]
    if len(parts) < 3:
        return sentence
    metric_at = None
    prefix = ""
    first = ""
    for index, part in enumerate(parts):
        match = _METRIC_SEGMENT.search(part)
        if match:
            metric_at = index
            prefix = part[: match.start()]
            first = part[match.start() :]
            break
    if metric_at is None or metric_at + 1 >= len(parts):
        return sentence
    second = parts[metric_at + 1]
    if not _METRIC_SEGMENT.match(second):
        return sentence
    conjunction = "and" if language == "en" else "y"
    head_parts = [*parts[:metric_at]]
    if prefix.strip():
        head_parts.append(prefix.strip())
    head = ", ".join(head_parts).strip(" ,")
    separator = " " if head.endswith(":") else ", "
    lead = f"{head}{separator}" if head else ""
    return f"{lead}{first} {conjunction} {second}."


def _lower_first(text: str) -> str:
    if not text or text[0].isdigit() or text[:2].isupper():
        return text
    first_word = text.split()[0]
    if "." in first_word or "®" in first_word:
        return text
    return text[0].lower() + text[1:]


def _item_sentence(item: str, language: str) -> str:
    raw = _plain_text(item).rstrip(".;,:")
    raw = re.sub(r"\s+[-\u2013\u2014]\s+", ", ", raw)
    value = _humanize(raw, language)
    if not value:
        return ""
    # «Riesgo cardiometabólico: 54.17» → «Riesgo cardiometabólico, 54.17».
    value = re.sub(r"^([^:]{1,48}):\s*", r"\1, ", value)
    return value if value.endswith((".", "!", "?", "…")) else value + "."


def _items_block(items: list[str], intro: str | None, language: str) -> list[str]:
    """Tres elementos como mucho, en frases naturales y con el valor intacto."""
    rendered = [sentence for item in items[:3] if (sentence := _item_sentence(item, language))]
    if not rendered:
        return []
    if intro:
        base = _humanize(intro, language).rstrip(":;, ")
        last = base.split()[-1].lower() if base.split() else ""
        glue = " " if last in {"con", "para", "de", "en", "sobre", "with", "for", "about"} else ": "
        connectors = [" Also, ", " And "] if language == "en" else [" También ", " Y "]
        merged = f"{base}{glue}{_lower_first(rendered[0])}"
        for index, sentence in enumerate(rendered[1:]):
            merged += connectors[index] + _lower_first(sentence)
        return [merged]
    return rendered


def _spoken_script(text: str, language: str, has_results: bool) -> tuple[list[str], bool]:
    lines = text.splitlines()
    blocks: list[tuple[str, object]] = []
    items: list[str] = []
    pending_intro: str | None = None
    for index, line in enumerate(lines):
        if not line.strip() or _HEADING.match(line) or line.strip().startswith("```"):
            continue
        if _TABLE_RULE.match(line) or (
            index + 1 < len(lines) and "|" in line and _TABLE_RULE.match(lines[index + 1])
        ):
            continue
        bullet = _BULLET.match(line)
        table_row = line.strip().startswith("|") and line.strip().endswith("|")
        if bullet or table_row:
            item = (
                bullet[1]
                if bullet
                else ", ".join(cell.strip() for cell in line.strip().strip("|").split("|"))
            )
            items.append(item)
            continue
        if items:
            blocks.append(("items", (items, pending_intro)))
            items = []
            pending_intro = None
        raw = _plain_text(line)
        if not raw:
            continue
        # El intro se detecta ANTES de humanizar y conserva los dos puntos: el
        # emisor no lo dice como frase hasta que llega el contenido.
        if raw.rstrip().endswith(":"):
            pending_intro = _humanize(raw, language).rstrip(":;, ") + ":"
            continue
        plain = _humanize(raw, language)
        if not plain:
            continue
        if pending_intro:
            # El intro con dos puntos se dice como frase propia; el párrafo
            # siguiente arranca con mayúscula y el streaming puede emitirlos
            # por separado sin esperar al final.
            blocks.append(("para", pending_intro.rstrip(":;, ") + "."))
            pending_intro = None
        blocks.append(("para", plain))
    if items:
        blocks.append(("items", (items, pending_intro)))

    sentences: list[str] = []
    for kind, payload in blocks:
        if kind == "items":
            group, intro = payload  # type: ignore[misc]
            sentences.extend(_items_block(group, intro, language))
        else:
            for sentence in _sentences(str(payload)):
                sentences.append(_trim_metric_list(sentence, language))
    has_items = any(kind == "items" for kind, _ in blocks)
    return sentences, has_items or has_results


def _fits(sentences: list[str], word_limit: int) -> bool:
    return (
        len(sentences) <= _SENTENCE_LIMIT
        and sum(len(sentence.split()) for sentence in sentences) <= word_limit
    )


def _bounded_prefix(sentences: list[str], limit: int) -> list[str]:
    prefix: list[str] = []
    for sentence in sentences:
        candidate = [*prefix, sentence]
        if not _fits(candidate, limit):
            break
        prefix.append(sentence)
    return prefix


def _first_clause(sentence: str, limit: int) -> str:
    """Recorta una frase que no cabe, siempre en coma o punto, nunca a mitad de palabra."""
    words = sentence.split()
    if len(words) <= limit:
        return sentence
    clauses = re.split(r"(?<=[,;:])\s+", sentence)
    chosen = ""
    for clause in clauses:
        candidate = f"{chosen} {clause}".strip()
        if len(candidate.split()) > limit:
            break
        chosen = candidate
    if chosen:
        return chosen.rstrip(" ,;:") + "."
    return " ".join(words[:limit]).rstrip(" ,;:") + "."


def format_voice_answer(text: str, *, language: str | None, has_results: bool) -> str:
    """Devuelve el texto que se habla y se guarda en el hilo."""
    if not text or not text.strip():
        return ""
    lang = "en" if language == "en" else "es"
    sentences, results = _spoken_script(text, lang, has_results)
    if not sentences:
        return ""
    urgent = bool(_SAFETY.search(normalize(text)))
    if urgent:
        return " ".join(sentences)
    limit = _RESULT_WORD_LIMIT if results else _GENERAL_WORD_LIMIT
    spoken = _bounded_prefix(sentences, limit)
    if not spoken:
        return _first_clause(sentences[0], limit)
    trimmed = [_trim_metric_list(sentence, lang) for sentence in spoken]
    # Una oferta de «más detalles por chat» no se dice en voz: el paciente ya
    # está conversando y el detalle llega en el siguiente turno.
    kept = [sentence for sentence in trimmed if not _CHAT_OFFER.search(sentence.strip())]
    return " ".join(kept or trimmed[:1])


class VoiceSentenceEmitter:
    """Emite un prefijo estable del texto hablado y completa en el cierre.

    Mantiene la latencia del streaming: en cuanto hay una frase completa sin
    marcas visuales, se entrega; si el modelo usa listas o tablas, el texto se
    entrega formateado al cerrar el turno.
    """

    def __init__(self, language: str | None):
        self._language = "en" if language == "en" else "es"
        self._buffer = ""
        self._emitted = ""
        self._deferred = False

    def feed(self, fragment: str) -> str:
        if self._deferred:
            return ""
        self._buffer += fragment
        # Un salto tras dos puntos separa frases en el texto hablado; el resto
        # de saltos son espacios. Así el streaming ve la misma estructura que
        # el formateador final.
        text = self._buffer.replace("\r", "")
        text = re.sub(r":\s*\n", ". ", text)
        text = text.replace("\n", " ")
        if self._buffer and self._buffer[-1].isspace() and not text.endswith(" "):
            text += " "
        markup = _STREAM_MARKUP.search(text)
        plain = _INLINE_MARKUP.sub("", text[: markup.start()] if markup else text)
        plain = re.sub(r"\s+", " ", plain).lstrip()
        sentences: list[str] = []
        for sentence in _sentences(plain, include_tail=False, final=False):
            human = _humanize(sentence, self._language)
            if human and not _CHAT_OFFER.search(human.strip()):
                sentences.append(_trim_metric_list(human, self._language))
        prefix = _bounded_prefix(sentences, _RESULT_WORD_LIMIT)
        self._deferred = (
            bool(markup) or len(prefix) < len(sentences) or len(prefix) == _SENTENCE_LIMIT
        )
        return self._new_text(" ".join(prefix))

    def finish(self, saved: str, *, has_results: bool, allow_standalone: bool = False) -> str:
        text = saved.strip()
        if not text:
            return ""
        if not self._emitted:
            if allow_standalone:
                self._emitted = text
                return text
            complete = " ".join(_sentences(text, include_tail=False))
            return self._new_text(complete)
        if not text.startswith(self._emitted):
            # El formateo cambió el texto ya hablado; no se repite nada.
            return ""
        remainder = text[len(self._emitted) :].strip()
        if not remainder:
            return ""
        delta = " ".join(_sentences(remainder, include_tail=False))
        if not delta:
            return ""
        self._emitted = f"{self._emitted} {delta}"
        # El chunk se concatena con el anterior: conserva el separador.
        return f" {delta}"

    def _new_text(self, prefix: str) -> str:
        if self._emitted and not prefix.startswith(self._emitted + " "):
            return ""
        delta = prefix[len(self._emitted) :]
        self._emitted = prefix
        return delta
