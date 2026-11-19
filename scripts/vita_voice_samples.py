"""Muestras auditables de voz de Vita (KAN-55): mismas frases ES/EN por voz candidata.

Genera un MP3 por voz e idioma con texto equivalente (nombres, cifras y frases clínicas)
para que el owner compare antes de aprobar el cambio en el agente ElevenLabs. Solo llama
a text-to-speech: no modifica agentes ni voces. Consume créditos de ElevenLabs.

Uso (la clave sale del entorno, nunca de argumentos ni logs):

    uv run python scripts/vita_voice_samples.py --out ./voice-samples \
        --voice es=<VOICE_ID_ES> --voice en=<VOICE_ID_EN> [--voice es=<OTRA>] \
        [--model eleven_multilingual_v2]

Variables: ELEVENLABS_API_KEY (obligatoria), ELEVENLABS_BASE_URL (opcional).
Las voces de la biblioteca compartida deben añadirse antes al workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import httpx

# Mismos ajustes que el agente actual (lectura 18-11-2026) para comparar solo la voz.
VOICE_SETTINGS = {"stability": 0.35, "similarity_boost": 0.72, "speed": 1.05}

SAMPLES = {
    "es": (
        "Hola, María José. Soy Vita, tu asistente de Copp Adresd. "
        "Registré tu peso de setenta y nueve coma cinco kilos y tu presión de "
        "ciento veinte sobre ochenta. Tu próxima toma de metformina es a las ocho y media "
        "de la noche. Si sientes dolor en el pecho o falta de aire, llama a emergencias de "
        "inmediato."
    ),
    "en": (
        "Hi, María José. I'm Vita, your Copp Adresd assistant. "
        "I recorded your weight of seventy-nine point five kilograms and your blood pressure "
        "of one hundred twenty over eighty. Your next metformin dose is at eight thirty in the "
        "evening. If you have chest pain or shortness of breath, call emergency services "
        "right away."
    ),
}


def parse_voices(values: list[str]) -> list[tuple[str, str]]:
    voices = []
    for value in values:
        language, _, voice_id = value.partition("=")
        if language not in SAMPLES or not voice_id.strip():
            raise SystemExit(f"--voice inválido: {value!r} (usa es=<id> o en=<id>)")
        voices.append((language, voice_id.strip()))
    return voices


def main() -> int:
    parser = argparse.ArgumentParser(description="Muestras auditables de voz de Vita (KAN-55).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--voice", action="append", required=True, help="es=<id> | en=<id>")
    parser.add_argument("--model", default="eleven_multilingual_v2")
    args = parser.parse_args()

    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        print("Falta ELEVENLABS_API_KEY en el entorno.", file=sys.stderr)
        return 2
    base = os.environ.get("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io").rstrip("/")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    with httpx.Client(timeout=60) as client:
        for language, voice_id in parse_voices(args.voice):
            response = client.post(
                f"{base}/v1/text-to-speech/{voice_id}",
                params={"output_format": "mp3_44100_128"},
                headers={"xi-api-key": key},
                json={
                    "text": SAMPLES[language],
                    "model_id": args.model,
                    "language_code": language,
                    "voice_settings": VOICE_SETTINGS,
                },
            )
            if response.status_code != 200:
                print(f"{language}/{voice_id}: HTTP {response.status_code}", file=sys.stderr)
                continue
            path = args.out / f"vita-{language}-{voice_id}.mp3"
            path.write_bytes(response.content)
            manifest.append(
                {
                    "language": language,
                    "voice_id": voice_id,
                    "model_id": args.model,
                    "voice_settings": VOICE_SETTINGS,
                    "text": SAMPLES[language],
                    "file": path.name,
                    "sha256": hashlib.sha256(response.content).hexdigest(),
                }
            )
            print(f"OK {path}")
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
