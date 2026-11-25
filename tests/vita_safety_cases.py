"""Bilingual synthetic cases for Vita's patient-safety boundaries."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGenerationChunk

from app.agents.prompts import PATIENT_SCOPE_POLICY, build_language_instruction
from tests.fakes import RecordingFakeModel

LANGUAGES = ("es", "en")
FIXTURE_PATH = Path(__file__).with_name("fixtures")
MATRIX_FILES = (
    "vita_safety_matrix.json",
    "vita_safety_direct_cases.json",
    "vita_safety_implicit_cases.json",
)

REFERRAL = {
    "es": "Habla con tu equipo de salud para una evaluación individual.",
    "en": "Talk with your healthcare team for an individual assessment.",
}
CLINICAL_REFERRAL = {
    "es": "Vita no puede ayudarte con esta consulta. Consulta a un profesional de la salud.",
    "en": "Vita can't help with that request. Please consult a healthcare professional.",
}
EMERGENCY_US = {
    "es": "Llama al 911 ahora. Contacta a los servicios de emergencia.",
    "en": "Call 911 now. Contact emergency services.",
}
EMERGENCY_LOCAL = {
    "es": "Contacta a los servicios locales de emergencia ahora.",
    "en": "Contact local emergency services now.",
}
PRIVACY_REFUSAL = {
    "es": "Solo puedo ayudarte con los datos de tu propia cuenta autorizada.",
    "en": "I can only help you with data from your own authorized account.",
}
CLARIFICATION = {
    "es": "¿Puedes aclarar a qué información te refieres?",
    "en": "Can you clarify which information you mean?",
}
WORKFLOW_BOUNDARY = {
    "es": "Los cambios se hacen por la opción autorizada de la app con su confirmación.",
    "en": "Changes go through the authorized option in the app with its confirmation.",
}
UNVERIFIED_RECORD = {
    "es": "El registro necesita una fuente verificable antes de confirmar ese dato.",
    "en": "The entry needs a verifiable source before confirming that value.",
}
FIXTURE_REPLIES = {
    "refer": REFERRAL,
    "urgent-us": EMERGENCY_US,
    "urgent-local": EMERGENCY_LOCAL,
    "privacy": PRIVACY_REFUSAL,
    "clarify": CLARIFICATION,
    "workflow": WORKFLOW_BOUNDARY,
    "unverified": UNVERIFIED_RECORD,
}
PRIOR_ANSWER = {"es": "Entiendo tu pregunta.", "en": "I understand your question."}


@dataclass(frozen=True)
class SafetyCase:
    language: str
    prompt: str
    outcome: str
    previous: str | None
    education: str | None

    @property
    def answer(self):
        if self.outcome in {"education", "mixed"}:
            assert self.education
            if self.outcome == "education":
                return self.education
            return f"{self.education} {REFERRAL[self.language]}"
        return FIXTURE_REPLIES[self.outcome][self.language]


def load_cases():
    rows = []
    for filename in MATRIX_FILES:
        rows.extend(json.loads((FIXTURE_PATH / filename).read_text(encoding="utf-8")))
    assert len(rows) >= 500, "Vita's regression matrix requires at least 500 bilingual scenarios"
    ids = {row["id"] for row in rows}
    assert len(ids) == len(rows), "Vita safety case IDs must be unique"
    seen_prompts = set()
    cases = []
    for row in rows:
        assert row["prompt"]["es"].strip() != row["prompt"]["en"].strip(), row["id"]
        for language in LANGUAGES:
            prompt = row["prompt"][language].strip()
            assert prompt, (row["id"], language)
            key = (language, " ".join(prompt.casefold().split()))
            assert key not in seen_prompts, (row["id"], language)
            seen_prompts.add(key)
            case = SafetyCase(
                language=language,
                prompt=prompt,
                outcome=row["outcome"],
                previous=row["previous"][language].strip() if "previous" in row else None,
                education=row["education"][language].strip() if "education" in row else None,
            )
            cases.append(pytest.param(case, id=f"{row['category']}-{row['id']}-{language}"))
    return cases


SAFETY_CASES = load_cases()

HIDDEN_MARKER = "PRIVATE_PROVIDER_FIXTURE"


class ChunkedSafetyModel(RecordingFakeModel):
    """Emit real provider-style increments with an additional hidden block."""

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        answer = self._generate(messages, stop=stop, **kwargs).generations[0].message.content
        assert isinstance(answer, str)
        yield ChatGenerationChunk(
            message=AIMessageChunk(content=[{"type": "thinking", "thinking": HIDDEN_MARKER}])
        )
        for index, word in enumerate(answer.split(" ")):
            yield ChatGenerationChunk(message=AIMessageChunk(content=f" {word}" if index else word))


def sse_payloads(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]


def assert_patient_scope(messages, language, user_text, previous=None):
    assert messages[0] == SystemMessage(content=PATIENT_SCOPE_POLICY)
    assert SystemMessage(content=build_language_instruction(language)) in messages
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == user_text
    if previous:
        assert any(
            isinstance(message, HumanMessage) and message.content == previous
            for message in messages[:-1]
        )


def assert_context_delivery(model, case, fragments):
    answer = "".join(fragments)
    if model.prompts:
        assert len(model.prompts) == 1
        assert_patient_scope(model.prompts[0], case.language, case.previous)
        assert answer == PRIOR_ANSWER[case.language]
    else:
        assert answer == CLINICAL_REFERRAL[case.language]


def assert_safe_delivery(case, fragments, prompts):
    assert fragments and all(fragments)
    answer = "".join(fragments)
    assert HIDDEN_MARKER not in answer
    if prompts:
        assert len(prompts) == 1
        assert_patient_scope(prompts[0], case.language, case.prompt, case.previous)
    elif case.outcome not in {"urgent-local", "urgent-us"}:
        assert case.outcome in {"refer", "mixed", "workflow"}
        assert answer == CLINICAL_REFERRAL[case.language]
        return answer
    assert answer == case.answer
    return answer
