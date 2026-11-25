"""Selección acotada al catálogo autorizado del turno. Metadata no concede tools."""

import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFD", value.casefold())
    return "".join(c for c in value if unicodedata.category(c) != "Mn")


class RoutingMetadata(BaseModel):
    topics: list[str] = Field(default_factory=list, max_length=32)
    keywords: list[str] = Field(default_factory=list, max_length=32)
    channels: list[Literal["text", "voice"]] = Field(default_factory=list)


class AuthorizedAgent(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)
    id: str = Field(min_length=1)
    runtime_key: str = Field(min_length=1)
    slug: str | None = None
    name: str
    description: str | None = None
    specialty: str | None = None
    version_id: str = Field(min_length=1)
    version_number: int = Field(ge=1)
    routing: RoutingMetadata


@dataclass(frozen=True)
class RouteDecision:
    agent: AuthorizedAgent | None
    reason: str


Classifier = Callable[[str, list[AuthorizedAgent]], Awaitable[str | None]]

# Traducciones de etiquetas del catálogo, nunca IDs ni una lista de agentes fija.
# Especialistas nuevos siguen usando sus nombres/metadata y el clasificador acotado.
_LABEL_TRANSLATIONS = (
    {"psychology", "psicologia", "mental health", "salud mental"},
    {"nutrition", "nutricion"},
    {"medical", "medicine", "medicina"},
)


def _positive_text(value: str) -> str:
    # Evita que «no quiero nutrición, quiero ...» active el dominio negado.
    return re.sub(
        r"\b(?:no|not|don't|do not)\s+(?:(?:quiero|want|necesito|need)\s+)?[^,;.!?]+",
        "",
        normalize(value),
    )


def _availability_query(text: str) -> bool:
    # Estado de la conversación, no documentación clínica ni datos del plan.
    return bool(
        re.fullmatch(
            r"(?:vita[ ,¿]*)?(?:y |and )?(?:"
            r"por que (?:(?:ahora (?:si )?|antes no |no ))?(?:funcionas?|respondes|respondias)|"
            r"(?:ya |ahora )?(?:estas (?:funcionando|disponible|lista)|funcionas|respondes)|"
            r"why (?:are you|is the (?:chat|voice call|app)) (?:working|responding)(?: now)?|"
            r"are you (?:working|responding|available|ready)(?: now)?)",
            text.strip("¿?! .,;"),
        )
    )


APPLICATION_NAME = "Agentic Wellness IA"
_LEGACY_APPLICATION_NAME = r"(?:copp[-\s\u2010-\u2015]*add?res+d|antares)"
# Los límites distinguen la marca de dominios, direcciones y nombres técnicos.
LEGACY_APPLICATION_NAME = re.compile(
    rf"(?<![\w@./-]){_LEGACY_APPLICATION_NAME}"
    rf"(?:\s*/\s*{_LEGACY_APPLICATION_NAME})*(?![\w@/-]|\.\w)",
    re.IGNORECASE,
)


def application_name_statement(language: str | None) -> str:
    if language == "en":
        return f"The app is called {APPLICATION_NAME}."
    return f"La aplicación se llama {APPLICATION_NAME}."


def application_name_reply(text: str, language: str | None = None) -> str | None:
    """Resuelve preguntas directas sobre el nombre del producto, en ES y EN."""
    question = re.sub(r"\s+", " ", normalize(text)).strip("¿?¡! .,;").replace("\u2019", "'")
    question = re.sub(r"^vita[ ,¿]+", "", question)
    if LEGACY_APPLICATION_NAME.fullmatch(question):
        return application_name_statement(language)
    if re.fullmatch(
        r"(?:como se llama (?:la|esta|tu) (?:app|aplicacion)|"
        r"cual es el nombre de (?:la|esta|tu) (?:app|aplicacion)|"
        r"(?:(?:la|esta) (?:app|aplicacion)|esto) (?:se llama|sigue siendo|es) "
        rf"{_LEGACY_APPLICATION_NAME}|"
        rf"estoy (?:en|usando) (?:la (?:app|aplicacion) de )?{_LEGACY_APPLICATION_NAME})"
        r"(?:[, ]+por favor)?",
        question,
    ):
        return application_name_statement("es")
    if re.fullmatch(
        r"(?:what(?: is|'s) (?:the name of (?:the|this|your) (?:app|application)|"
        r"(?:the|this|your) (?:app|application) called)|"
        r"is (?:this(?: (?:app|application))?|the (?:app|application)) (?:still )?(?:called )?"
        rf"{_LEGACY_APPLICATION_NAME}|"
        rf"am i (?:using|on) (?:the (?:app|application) from )?{_LEGACY_APPLICATION_NAME})"
        r"(?:[, ]+please)?",
        question,
    ):
        return application_name_statement("en")
    return None


def _service_query(text: str) -> bool:
    if application_name_reply(text) is not None:
        return True
    text = text.strip("¿?! .,;")
    arithmetic = re.sub(r"^(?:cuanto es|calcula|calculate|what is|how much is)\s+", "", text)
    arithmetic = re.sub(r"\s+(?:please|por favor)$", "", arithmetic)
    arithmetic = re.sub(
        r"\b(?:por|mas|menos|dividido entre|plus|minus|times|divided by)\b", "+", arithmetic
    )
    if re.search(r"\d", arithmetic) and re.fullmatch(r"[\d\s.,()+*/%=^\-]+", arithmetic):
        return True
    if re.fullmatch(
        r"(?:hola|hello|hi|hey|buenos dias|buenas tardes|good morning|good evening)"
        r"(?: vita)?[ ,.!]*(?:(?:como estas|how are you)(?: doing)?(?: hoy| today)?)?",
        text,
    ):
        return True
    if re.search(r"\b(quien eres|who are you|tu nombre|your name)\b", text):
        return True
    if _availability_query(text):
        return True
    # Solo ayuda genérica completa: un tema añadido debe conservar su lectura/RAG.
    if re.fullmatch(
        r"(?:vita[ ,¿]*)?(?:que puedes hacer(?: por mi)?|en que puedes ayudarme|"
        r"como (?:me puedes ayudar|puedes ayudarme)|what can you do(?: for me)?|"
        r"what can you help me with|how can you help me)(?: hoy| today)?",
        text,
    ):
        return True
    if re.match(r"^(?:gracias|thank you|thanks)\b", text) and not re.search(
        r"\b(pero|but|ahora|now|also)\b", text
    ):
        return True
    return bool(
        re.search(r"\b(app|aplicacion|application)\b", text)
        and re.search(
            r"\b(ayuda|usar|uso|utilizar|abro|abrir|help|use|using|open|settings)\b", text
        )
    )


async def select_agent(
    text: str,
    catalog: list[AuthorizedAgent],
    *,
    channel: str = "text",
    explicit: str | None = None,
    previous: str | None = None,
    classifier: Classifier | None = None,
) -> RouteDecision:
    # El catálogo del backend solo distingue texto y voz. WhatsApp es un canal
    # textual: hereda la elegibilidad de texto para mantener paridad con la app.
    eligibility = "text" if channel == "whatsapp" else channel
    allowed = [agent for agent in catalog if eligibility in agent.routing.channels]
    if explicit and explicit != "base":
        matches = [a for a in allowed if explicit in {a.id, a.runtime_key, a.slug}]
        if not matches:
            raise ValueError("agent_not_authorized")
        return RouteDecision(matches[0], "explicit")
    # «No respondes» describe el servicio; no es la negación de un dominio.
    if _availability_query(normalize(text)):
        return RouteDecision(None, "service_fast_path")
    normalized = _positive_text(text)
    # Saludos/identidad no consultan un segundo modelo ni heredan especialista.
    if re.fullmatch(r"[\W]*(?:hola|hello|hi|hey|quien eres|who are you)[\W]*", normalized):
        return RouteDecision(None, "general")
    if _service_query(normalized):
        return RouteDecision(None, "service_fast_path")
    scored: list[tuple[int, AuthorizedAgent]] = []
    for agent in allowed:
        terms = [
            agent.name,
            agent.specialty or "",
            agent.slug or "",
            *agent.routing.topics,
            *agent.routing.keywords,
        ]
        labels = set(map(normalize, [agent.specialty or "", agent.slug or "", agent.name]))
        for translations in _LABEL_TRANSLATIONS:
            if labels & translations:
                terms.extend(translations)
        score = sum(
            1 + min(len(term.split()) - 1, 2)
            for term in set(map(normalize, terms))
            if term and re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", normalized)
        )
        if score:
            scored.append((score, agent))
    scored.sort(key=lambda item: (-item[0], item[1].id))
    if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return RouteDecision(scored[0][1], "catalog_rule")
    if (
        not scored
        and previous
        and re.match(
            r"^(?:y eso|y como|y que|y entonces|and (?:that|how|what)|how (?:do|can) i|eso)\b",
            normalized.strip("¿?! ."),
        )
    ):
        active = next((a for a in allowed if a.id == previous), None)
        if active:
            return RouteDecision(active, "continuity")
    nonliteral = (
        not scored
        and len(normalized.split()) >= 3
        and bool(
            re.search(
                r"[¿?]|\b(me|mi|mis|tengo|siento|estoy|necesito|quiero|busco|quisiera|puedo|"
                r"que|como|sobre|explica|explicame|orienta|my|i|feel|feeling|need|want|looking|"
                r"about|explain|tell|what|how|why)\b",
                normalized,
            )
        )
    )
    if classifier and allowed and (len(scored) > 1 or nonliteral):
        candidates = [a for _, a in scored] if scored else allowed
        selected = await classifier(text, candidates)
        active = next((a for a in candidates if a.id == selected), None)
        if active:
            return RouteDecision(active, "structured_classifier")
    return RouteDecision(None, "general" if not scored else "clarification")


def documentation_query(text: str) -> bool:
    """Intención documental ES/EN; no confunde un test personal con su protocolo."""
    return bool(
        re.search(
            r"\b(documentacion|documentos?|documentation|documents?|manual|manuales|"
            r"protocolo|protocolos|protocols?|guia|guias|guidelines?|fuentes?|sources?|"
            r"bibliografia|bibliography|rag|material (?:cargado|adjunto)|uploaded|attached)\b",
            _positive_text(text),
        )
    )


def service_query(text: str) -> bool:
    """Servicio/saludo sin lectura de ficha ni recuperación documental."""
    return _service_query(normalize(text))


def patient_document_query(text: str) -> bool:
    """Busca contenido del expediente; el listado y la apertura siguen por ID."""
    normalized = _positive_text(text)
    return bool(
        re.search(
            r"\b((mi|mis|my) (documentos?|archivos?|expediente|examen|informe|"
            r"(?:uploaded )?(?:documents?|files?|report|exam|chart))|"
            r"(documento|archivo|examen|informe) que (subi|cargue)|"
            r"(document|file|report|exam) i uploaded)\b",
            normalized,
        )
        and re.search(
            r"\b(segun|dice|dicen|contenido|resume|resumen|menciona|say|says|according|"
            r"contents?|summari[sz]e|mention)\b",
            normalized,
        )
    )


def patient_record_query(text: str) -> bool:
    """Abre el catálogo de lecturas si se pide un dato personal de la ficha."""
    text = _positive_text(text)
    return bool(
        re.search(
            r"\b(perfil|ficha|profile|record|mis datos|my data|edad|age|metas?|goals?|"
            r"alergias?|allergies|restricciones|restrictions|peso|talla|imc|weight|height|bmi|"
            r"antropometria|anthropometry|"
            r"signos vitales|vitals?|biometria|biometrics|presion|tension arterial|blood pressure|"
            r"pulso|heart rate|glucosa|glucose|blood sugar|saturacion|oxygen|temperatura|"
            r"temperature|grasa corporal|body fat|sueno|sleep|medicamentos?|medicacion|"
            r"meds|medications?|tratamientos?|treatments?|dosis|doses?|suplementos?|supplements?|"
            r"expediente|documentos?|documents?|archivos?|files?|baterias?|batteries|"
            r"resultados?|scores?|ciclo nutricional|nutrition cycle|recomendaciones|"
            r"recommendations|historial de comidas|meal history|plan completo|full meal plan|"
            r"plan de 30 dias|30.day plan)\b",
            text,
        )
        and re.search(
            r"\b(mi|mis|yo|me|my|i|mine|ayer|yesterday|hoy|today|tomo|tome|tomado|"
            r"muestrame|show|dime|tell me|registrad[oa]s?|recorded)\b",
            text,
        )
    )


_RECORD_TOPIC_PATTERNS = {
    "vitals": (
        r"signos vitales|vitals?|biometria|biometrics|presion|tension arterial|blood pressure|"
        r"pulso|frecuencia cardiaca|heart rate|glucosa|glucose|blood sugar|saturacion|oxygen|"
        r"temperatura|temperature|grasa corporal|body fat|sueno|sleep|reloj|manilla|anillo|"
        r"wearable|watch|tracker|spo2|pasos|steps|calorias activas|active calories|"
        r"distancia|distance"
    ),
    "profile": (
        r"perfil|ficha|profile|edad|age|metas?|goals?|alergias?|allergies|restricciones|"
        r"restrictions|peso|talla|imc|weight|height|bmi|antropometria|anthropometry"
    ),
    "medications": (
        r"medicamentos?|medicacion|meds|medications?|tratamientos?|treatments?|"
        r"dosis|doses?|suplementos?|supplements?"
    ),
    "assessments": (
        r"evaluaciones?|assessments?|tests?|baterias?|batter(?:y|ies)|puntajes?|scores?|"
        r"resultados? (?:de (?:mis )?)?(?:pruebas|evaluaciones|tests)"
    ),
    "nutrition_plan": (
        r"ciclo nutricional|nutrition cycle|historial de comidas|meal history|"
        r"plan completo|full meal plan|plan de 30 dias|30.day plan"
    ),
    "documents": r"expediente|documentos?|documents?|archivos?|files?|informes?|reports?",
}


def voice_record_topics(text: str) -> set[str]:
    normalized = _positive_text(text)
    return {
        topic
        for topic, pattern in _RECORD_TOPIC_PATTERNS.items()
        if re.search(rf"\b(?:{pattern})\b", normalized)
    }


def voice_record_lookup(text: str) -> bool:
    return patient_record_query(text) or bool(
        re.search(
            r"\b(?:mi|mis|my|mine|muestrame|dime|show me|tell me)\b|"
            r"^(?:los|las) (?:del|de la) (?:reloj|manilla|anillo|wearable)\b|"
            r"^(?:the ones from|from) (?:the|my) (?:watch|wearable|tracker)\b",
            _positive_text(text).strip("¿?! ."),
        )
    )


def voice_document_list_lookup(text: str) -> bool:
    # Mencionar el nombre del recurso al pedir el listado personal no basta
    # para activar RAG. Una referencia a una guía o un protocolo sí lo hace.
    without_resource = re.sub(r"\b(?:documentos?|documents?)\b", "", _positive_text(text))
    return voice_record_lookup(text) and not documentation_query(without_resource)


def voice_record_followup(text: str) -> bool:
    """Reconoce referencias elípticas, sin heredar el tema de cualquier pregunta."""
    normalized = _positive_text(text).strip("¿?¡! .,;:")
    return bool(
        re.fullmatch(
            r"(?:(?:y|entonces|and|so|what about|how about|muestrame|dime|show me|tell me) )?"
            r"(?:(?:los|las|the) (?:de |del |from )?)?"
            r"(?:de )?(?:ayer|hoy|anoche|esta semana|la semana pasada|ultim[oa]s? "
            r"(?:(?:\d+|siete|tres) )?dias?|yesterday|today|last night|this week|last week|"
            r"(?:last|past) (?:(?:\d+|few|seven|three) )?days?|"
            r"cuales(?: son| eran)?|cuantos|el resto|lo demas|which ones?|the rest|what else)"
            r"(?: por favor| please)?",
            normalized,
        )
        or re.match(
            r"^(?:como asi (?:el )?\d|de que (?:fecha|dia) hablas|"
            r"what do you mean (?:by |on )?(?:the )?(?:\d|january|february|march|april|may|"
            r"june|july|august|september|october|november|december)\b)",
            normalized,
        )
    )


def capture_query(text: str) -> bool:
    """El paciente reporta un dato para registrarlo (medición, toma, comida o ánimo).

    Solo enruta al planificador con tools de propuesta; ningún registro ocurre sin la
    confirmación explícita del turno siguiente.
    """
    text = _positive_text(text)
    domain = (
        r"\b(peso|pese|weight|weigh|presion|tension|blood pressure|pulso|frecuencia cardiaca|"
        r"heart rate|saturacion|oxigeno|oxygen|temperatura|temperature|fiebre|fever|"
        r"medicamentos?|medicinas?|pastillas?|dosis|toma|pill|pills|meds|medications?|doses?|"
        r"desayuno|almuerzo|merienda|cena|comida|breakfast|lunch|snack|dinner|meal|"
        r"animo|estado de animo|mood|check.?in)\b"
    )
    if not re.search(domain, text):
        return False
    record_verb = re.search(
        r"\b(registra|registrar|registrame|anota|anotar|apunta|apuntar|guarda|guardar|"
        r"agrega|agregar|anade|marca|marcar|record|log|save|add|track|mark)\b",
        text,
    )
    # «Ya me tomé la pastilla», «hoy pesé 80 kg», «mi presión es 120/80», «I took my meds».
    report = re.search(
        r"\b(ya (?:me )?tome|me tome|tome|olvide|se me olvido|desayune|almorce|cene|comi|"
        r"took|skipped|missed|forgot|had|ate)\b",
        text,
    ) or re.search(r"\d+(?:[.,]\d+)?\s*(?:/\s*\d+|kg|kilos?|bpm|lpm|%|°|grados|c\b)", text)
    question = re.search(r"^(?:que|cual|cuanto|cuando|como|what|which|how|when)\b", text)
    return bool(record_verb or (report and not question))


def capability_scope(text: str, *, unambiguous: bool = False) -> str | None:
    """Atajos a lecturas; los datos reales y sus permisos siempre vienen de BACK."""
    text = _positive_text(text)
    # Atajo conservador: las peticiones coordinadas mantienen el planificador,
    # aunque sus partes parezcan compartir scope. No inferir una sola intención.
    if unambiguous and re.search(r"\b(y|and|ademas|also|tambien|junto|plus)\b", text):
        return None
    # Una pregunta conceptual/documental nunca se convierte en una lectura
    # personal por contener «tests», «ejercicios» o «rutina».
    personal = bool(
        re.search(
            r"\b(mi|mis|me|my|mine|tengo|asignad[oa]s?|assigned|hoy|today|pendientes?|"
            r"pending|para hoy|for today|que puedo|what can i)\b",
            text,
        )
    )
    if documentation_query(text) and not (
        personal and re.search(r"\b(plan|menu|almuerzo|lunch|podcast)\b", text)
    ):
        return None
    if not personal and not re.search(
        r"\b(abre|abrir|muestra|muestrame|ver|pon|reproduce|open|show|play|tasks|tareas)\b",
        text,
    ):
        return None
    patterns = {
        "nutrition": r"\b(almuerzo|desayuno|cena|comida|comidas|menu|meal|meals|lunch|"
        r"breakfast|dinner|plan nutricional|plan de nutricion|nutrition plan|nutritional plan|"
        r"diet plan|meal plan|food plan|eating plan|alimentacion|mi dieta|my diet|dieta|diet|"
        r"que debo comer|que como|comer hoy|what should i eat|what do i eat|what to eat|"
        r"cual es mi plan|what is my plan|plan de hoy|plan del dia|mi plan de hoy|"
        r"my plan today|my plan for today)\b",
        "assessments": r"\b(test|tests|pruebas|evaluaciones|assessment|assessments)\b",
        "podcast": r"\b(podcast|podcasts)\b",
        "exercise": r"\b(ejercicio|rutina|routine|workout|exercise|descanso|rest day|"
        r"entreno|entrenamiento|training)\b",
        "progress": r"\b(progreso|racha|xp|progress|streak|pasos|steps)\b",
        "day": r"\b(para hoy|tengo hoy|my day|for today|tareas|tasks)\b",
    }
    scopes = [scope for scope, pattern in patterns.items() if re.search(pattern, text)]
    # «Mi almuerzo para hoy» pide una lectura, no además el resumen del día.
    # Conserva day si hay una petición propia de tareas/resumen junto al plan.
    if (
        unambiguous
        and len(scopes) > 1
        and "day" in scopes
        and not re.search(
            r"\b(tareas|tasks|my day)\b",
            text,
        )
    ):
        scopes.remove("day")
    if unambiguous and len(scopes) != 1:
        return None
    return next(iter(scopes), None)
