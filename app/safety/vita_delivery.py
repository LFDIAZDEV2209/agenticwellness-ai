"""Política KAN-164. La clasificación es local y no envía datos a otro proveedor."""

import hashlib
import html
import re
import unicodedata

from langchain_core.messages import AIMessage

SAFETY_KEY = "vita_delivery_safety"
SAFETY_EVENT = "vita_delivery_block"
FALLBACK_REASONS = frozenset(
    {"clinical_topic", "clinical_output", "engine_decision", "empty_answer", "provider_failure"}
)

# Temas de exclusión explícita. No se excluyen palabras aisladas como «salud»,
# «ejercicio», «energía» o «dieta»: el acompañamiento general sigue disponible.
CLINICAL_TOPICS = re.compile(
    r"\b(?:diagnostic\w*|diagnos\w*|dos(?:is|es|age)|posologia|tratamientos?|treatments?|"
    r"indicaci(?:on|ones) clinicas?|medical advice|prescri\w*|recet(?:a|as|ame|ar)|"
    r"dolor (?:en (?:el )?|de )pecho|chest pain|dolor toracico|falta de aire|"
    r"shortness of breath|dificultad para respirar|difficulty breathing|"
    r"que (?:medicamento|pastilla|medicina|farmaco) (?:debo |puedo )?(?:tom\w+|us\w+)|"
    r"(?:what|which) (?:drug|medicine|medication|pill) (?:should|can|do) i (?:take|use))\b"
)
_MEDICINE = re.compile(
    r"\b(?:medicamentos?|medications?|medicinas?|medicine|drug|farmacos?|"
    r"antibiotic\w*|analgesic\w*|antihistamin\w*|pastillas?|pills?|tabletas?|tablets?|capsulas?|"
    r"capsules?|ibuprofen\w*|acetaminofen|acetaminophen|paracetamol|aspirin\w*|insulin\w*)\b"
)
_DOSE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|ug|ui|iu)\b")
_MEDICATION_SCHEDULE = re.compile(r"\b\d+(?:[.,]\d+)?\s*ml\b|\b(?:cada|every) \d+ horas?\b")
_CLINICAL_INFERENCE = re.compile(
    r"\b(?:probablemente (?:tienes|sea|es)|tus sintomas (?:indican|significan)|"
    r"(?:podrias|pareces) tener|esto (?:indica|significa) que tienes|"
    r"you (?:likely|probably|may) have|your symptoms (?:indicate|mean))\b"
    r"|\b(?:tienes|padeces|you have) (?:\w+ )?(?:diabetes|hipertension|hypertension|"
    r"neumonia|pneumonia|bronquitis|bronchitis|gastritis|asma|asthma|arritmia|arrhythmia)\b"
)
_MEDICAL_DIRECTIVE = re.compile(
    r"\b(?:toma|tom[ae]r|take|aplic[ae]|apply|usa|use|suspend[ae]|stop|avoid|evita|"
    r"sube|baja|reduce|aumenta|increase|lower|necesitas|you need)\b"
)
_ENERGY = re.compile(r"\b(?:\d[\d.,]*\s*(?:kcal|calorias?|calories?)|kilocalorias?)\b")
_EXERCISE = re.compile(
    r"\b(?:\d+|una|dos|tres|one|two|three)\s+"
    r"(?:series|sets|repeticiones|reps|sentadillas|squats|flexiones|push[ -]?ups)\b|"
    r"\b(?:corre|camina|entrena|pedalea|descansa|run|walk|train|cycle|rest)"
    r"\b[^.!?\n]{0,35}\b\d+\s*(?:minutos?|minutes?|km|miles?|dias?|days?)\b|"
    r"\b(?:haz|do|perform|realiza|evita|avoid|debes hacer)\s+"
    r"(?:(?:los|las|algunas|unos?|unas?|some|the)\s+)?"
    r"(?:sentadillas|squats|flexiones|push[ -]?ups|burpees|lunges|zancadas)\b"
)
_PLAN_CHANGE = re.compile(
    r"\b(?:ajust\w*|cambi\w*|modific\w*|reemplaz\w*|actualiz\w*|"
    r"adjust\w*|chang\w*|replac\w*|updat\w*)\b[^.!?\n]{0,45}"
    r"\b(?:tu|your) (?:plan|rutina|routine|workout|entrenamiento|objetivo|goal)\b"
)

# Lista cerrada del turno: una futura tool del motor no se concede por estar en
# un perfil ERP ni por añadirse al registro general. Capturar no ajusta el plan.
VITA_TOOL_ALLOWLIST = frozenset(
    {
        "calculate",
        "get_current_time",
        "get_patient_data",
        "mark_control_declined",
        "retrieve_knowledge",
        "get_my_day",
        "get_my_nutrition",
        "get_my_assessments",
        "get_my_podcast",
        "get_my_exercise",
        "get_my_progress",
        "get_my_profile",
        "get_my_vitals",
        "get_my_medications",
        "get_my_nutrition_plan",
        "get_my_assessment_results",
        "list_my_documents",
        "read_my_document",
        "record_measurement",
        "record_medication_dose",
        "record_meal",
        "record_checkin",
        "confirm_capture",
    }
)
GENERATION_POLICY = (
    "Vita safety: provide only non-clinical wellness support. Never provide a diagnosis, "
    "dose, treatment or clinical recommendation, even from context or a document. "
    "Never assign calories, prescribe exercises or decide a change to the engine plan. "
    "You may help users find their existing plan without choosing any of its values."
)


def _normalize(text: str) -> str:
    clean = "".join(
        char
        for char in unicodedata.normalize("NFKD", html.unescape(text).casefold())
        if unicodedata.category(char) not in {"Mn", "Cf"} and char not in "*_`~"
    )
    return " ".join(clean.split())


def prohibited_input(text: str) -> bool:
    normalized = _normalize(text)
    return bool(
        CLINICAL_TOPICS.search(normalized)
        or (
            _MEDICINE.search(normalized)
            and (_DOSE.search(normalized) or _MEDICATION_SCHEDULE.search(normalized))
        )
    )


def classify_output(text: str) -> str | None:
    """Clasifica la respuesta íntegra, combinando temas y señales de indicación."""
    normalized = _normalize(text)
    if not normalized.strip():
        return "empty_answer"
    if (
        CLINICAL_TOPICS.search(normalized)
        or _CLINICAL_INFERENCE.search(normalized)
        or _DOSE.search(normalized)
        or (
            _MEDICINE.search(normalized)
            and (_MEDICAL_DIRECTIVE.search(normalized) or _MEDICATION_SCHEDULE.search(normalized))
        )
    ):
        return "clinical_output"
    if (
        _ENERGY.search(normalized)
        or _EXERCISE.search(normalized)
        or _PLAN_CHANGE.search(normalized)
    ):
        return "engine_decision"
    return None


def fallback_text(reason: str, language: str | None) -> str:
    if reason in {"provider_failure", "empty_answer"}:
        return (
            "Vita couldn't complete this request. Please try again."
            if language == "en"
            else "Vita no pudo completar esta consulta. Inténtalo de nuevo."
        )
    return (
        "Vita can't help with that request. Please consult a healthcare professional."
        if language == "en"
        else "Vita no puede ayudarte con esta consulta. Consulta a un profesional de la salud."
    )


def fallback_message(reason: str, language: str | None) -> AIMessage:
    return AIMessage(
        content=fallback_text(reason, language), additional_kwargs={SAFETY_KEY: reason}
    )


def safety_reason(message: object) -> str | None:
    if isinstance(message, AIMessage):
        reason = message.additional_kwargs.get(SAFETY_KEY)
        if isinstance(reason, str) and reason in FALLBACK_REASONS:
            return reason
    return None


def fingerprint(text: str) -> str:
    """Identifica la salida exacta bloqueada sin copiarla a logs o historiales."""
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()
