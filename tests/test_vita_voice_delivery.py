"""Presupuesto incremental y correspondencia con el texto del hilo."""

import pytest

from app.voice.delivery import VoiceSentenceEmitter, format_voice_answer


@pytest.mark.parametrize(
    "language,opening,tail",
    [
        ("es", "Tu resultado está listo.", "La siguiente lectura"),
        ("en", "Your result is ready.", "The next reading was"),
    ],
)
def test_emits_complete_sentence_before_finish_and_drops_incomplete_tail(language, opening, tail):
    emitter = VoiceSentenceEmitter(language)
    assert emitter.feed(opening[:-1]) == ""
    assert emitter.feed(opening[-1]) == ""
    assert emitter.feed(f" {tail}") == opening
    saved = format_voice_answer(f"{opening} {tail}", language=language, has_results=True)
    assert emitter.finish(saved, has_results=True) == ""
    assert saved == f"{opening} {tail}"


@pytest.mark.parametrize(
    "language,fragments,sentence",
    [
        (
            "es",
            ["El peso fue 74.", "5 kg y la presión fue 120/", "80 mmHg. "],
            "El peso fue 74.5 kg y la presión fue 120/80 mmHg.",
        ),
        (
            "en",
            ["The pulse was 72.", "5 bpm at 10:", "30 a.m. ", "today. "],
            "The pulse was 72.5 bpm at 10:30 a.m. today.",
        ),
    ],
)
def test_decimal_time_and_units_survive_chunk_boundaries(language, fragments, sentence):
    emitter = VoiceSentenceEmitter(language)
    assert [emitter.feed(fragment) for fragment in fragments] == [
        *([""] * (len(fragments) - 1)),
        sentence,
    ]
    saved = format_voice_answer("".join(fragments), language=language, has_results=True)
    assert saved == sentence
    assert emitter.finish(saved, has_results=True) == ""


def test_urgent_instructions_are_preserved_beyond_the_budget():
    # Las instrucciones de seguridad no se recortan nunca; el audio y el hilo
    # conservan el texto completo.
    raw = "Call emergency services now. " + " ".join(["instruction"] * 45) + ". Stay nearby."
    saved = format_voice_answer(raw, language="en", has_results=True)
    emitter = VoiceSentenceEmitter("en")
    spoken = emitter.feed(raw + " ") + emitter.finish(saved, has_results=True)
    assert saved == raw
    assert spoken == raw


def test_non_urgent_answer_is_capped_to_three_sentences():
    raw = "Uno es el primero. Dos es el segundo. Tres es el tercero. Cuatro es el cuarto."
    saved = format_voice_answer(raw, language="es", has_results=True)
    assert saved == "Uno es el primero. Dos es el segundo. Tres es el tercero."


@pytest.mark.parametrize(
    "language,raw",
    [
        ("es", "Registraste 800 pasos. La medición es de ayer. Hay más entradas. Último detalle."),
        (
            "en",
            "Your result is ready.\n- Item one\n- Item two\n- Item three\n"
            "- Item four\n- Item five\n- Item six",
        ),
        ("es", "## Resumen.\nTu resultado fue de **74.5 kg**. ¿Quieres revisarlo? Otro detalle"),
        (
            "en",
            "Your result is ready. This is a [title. with punctuation](https://example.invalid).",
        ),
    ],
)
def test_every_increment_is_an_exact_prefix_of_the_formatted_answer(language, raw):
    saved = format_voice_answer(raw, language=language, has_results=True)
    emitter = VoiceSentenceEmitter(language)
    spoken = ""
    for char in raw:
        spoken += emitter.feed(char)
        assert saved.startswith(spoken)
    spoken += emitter.finish(saved, has_results=True)
    assert saved.startswith(spoken)
    assert spoken


def test_bullets_keep_the_first_three_items_without_dangling_colon():
    raw = (
        "Puedo ayudarte con:\n"
        "- Revisar tus signos vitales y actividad.\n"
        "- Resultados de tus tests.\n"
        "- Tu plan nutricional.\n"
        "- Tus documentos."
    )
    spoken = format_voice_answer(raw, language="es", has_results=False)
    assert "revisar tus signos vitales" in spoken.lower()
    assert "resultados de tus tests" in spoken.lower()
    assert "plan nutricional" in spoken.lower()
    assert "documentos" not in spoken.lower()
    assert not spoken.endswith(":")
    assert "ayudarte con:" not in spoken


def test_vitals_paragraph_humanizes_date_and_keeps_two_metrics():
    raw = (
        "Listo. Aquí están tus últimos datos del wearable:\n"
        "Ayer, 18 de nov (12:00): 2,306 calorías quemadas, 1,806 pasos, 1,324 metros."
    )
    spoken = format_voice_answer(raw, language="es", has_results=True)
    assert "Ayer, 2.306 calorías quemadas y 1.806 pasos." in spoken
    assert "(12:00)" not in spoken
    assert "1.324 metros" not in spoken
    assert "18 de noviembre" not in spoken


def test_month_abbreviation_is_spoken_in_full():
    spoken = format_voice_answer(
        "Tu evaluación inicial fue el 5 de nov.", language="es", has_results=True
    )
    assert "5 de noviembre" in spoken


def test_parenthetical_score_reads_as_clarification():
    spoken = format_voice_answer(
        "Riesgo cardiometabólico: 54.17 (alto).", language="es", has_results=True
    )
    assert "54.17, alto" in spoken
    assert "(" not in spoken


def test_chat_offer_is_not_spoken():
    spoken = format_voice_answer(
        "Tus resultados están listos. Más detalles por chat.",
        language="es",
        has_results=True,
    )
    assert "por chat" not in spoken.lower()
    assert "resultados están listos" in spoken


def test_english_metric_list_uses_and():
    spoken = format_voice_answer(
        "Nov 18: 2,306 calories burned, 1,806 steps, 1,324 meters.",
        language="en",
        has_results=True,
    )
    assert "2,306 calories burned and 1,806 steps." in spoken
    assert "meters" not in spoken


def test_empty_answer_stays_empty():
    assert format_voice_answer("   ", language="es", has_results=True) == ""
    assert format_voice_answer("", language="en", has_results=False) == ""
