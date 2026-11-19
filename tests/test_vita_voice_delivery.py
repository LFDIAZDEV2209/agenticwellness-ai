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


@pytest.mark.parametrize("has_results,limit", [(True, 40), (False, 50)])
def test_word_cap_keeps_a_sentence_at_the_limit_and_drops_everything_after(has_results, limit):
    # El hilo conserva todas las instrucciones urgentes. Aísla el tope del SSE
    # para comprobarlo incluso cuando el texto guardado supera el presupuesto.
    first = "Call emergency services now."
    second = " ".join(["instruction"] * (limit - len(first.split()))) + "."
    raw = f"{first} {second} Stay nearby."
    saved = format_voice_answer(raw, language="en", has_results=has_results)
    emitter = VoiceSentenceEmitter("en")

    assert emitter.feed(first + " ") == first
    partial = emitter.feed(second + " Stay nearby.")
    spoken = first + partial + emitter.finish(saved, has_results=has_results)

    assert saved == raw
    assert spoken == f"{first} {second}"
    assert len(spoken.split()) == limit


def test_emitter_stops_at_three_sentences_even_if_the_saved_answer_is_longer():
    emitter = VoiceSentenceEmitter("es")
    raw = "Llama a emergencias ahora. Busca ayuda. Quédate aquí. Sigue las indicaciones."
    saved = format_voice_answer(raw, language="es", has_results=True)
    spoken = emitter.feed(raw + " ") + emitter.finish(saved, has_results=True)
    assert spoken == "Llama a emergencias ahora. Busca ayuda. Quédate aquí."
    assert saved == raw


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
