"""Turno de audio sin egress, para el piloto de tocar para hablar."""

import asyncio
import base64
import io
import logging
import math
import wave
from contextlib import suppress
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app.agents.prompts import PATIENT_SCOPE_POLICY
from app.api.security import require_internal_key
from app.core.config import get_settings
from app.safety.vita_delivery import (
    classify_output,
    emergency_delivery_reason,
    fallback_text,
    protected_input_reason,
)

router = APIRouter(prefix="/internal/vita/voice", dependencies=[Depends(require_internal_key)])
logger = logging.getLogger(__name__)

MAX_BYTES = 2 * 1024 * 1024
MAX_SECONDS = 20
RATE = 16000
MAX_CHARS = 1400
MIN_SPEECH_LOGPROB = -0.85
MAX_NO_SPEECH = 0.6
FORMATS = {"audio/mp4": "mov", "audio/webm": "matroska", "audio/wav": "wav"}
slots = asyncio.Semaphore(2)

Language = Literal["es", "en"]


class TapVoiceResponse(BaseModel):
    transcript: str
    reply: str
    audio_base64: str = Field(serialization_alias="audioBase64")


class InvalidAudioError(Exception):
    pass


async def _exec(args: tuple[str, ...], data: bytes) -> bytes:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await process.communicate(data)
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
    if process.returncode:
        raise InvalidAudioError()
    return stdout


def _wave(pcm: bytes, rate: int, channels: int, width: int) -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as output:
        output.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        output.writeframes(pcm)
    return target.getvalue()


async def decode_audio(raw: bytes, media: str) -> bytes:
    # Sin rutas, protocolos de red, ni formato tomado del nombre del paciente.
    pcm = await _exec(
        (
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-protocol_whitelist",
            "pipe",
            "-f",
            FORMATS[media],
            "-i",
            "pipe:0",
            "-map",
            "0:a:0",
            "-t",
            str(MAX_SECONDS + 1),
            "-ac",
            "1",
            "-ar",
            str(RATE),
            "-f",
            "s16le",
            "pipe:1",
        ),
        raw,
    )
    if not pcm or len(pcm) > RATE * 2 * (MAX_SECONDS + 0.5):
        raise InvalidAudioError()
    return _wave(pcm, RATE, 1, 2)


async def speak(text: str, language: Language) -> bytes:
    audio = await _exec(
        ("espeak-ng", "--stdout", "--stdin", "-v", "es-419" if language == "es" else "en-us"),
        text.encode(),
    )
    with wave.open(io.BytesIO(audio), "rb") as stream:
        return _wave(
            stream.readframes(stream.getnframes()),
            stream.getframerate(),
            stream.getnchannels(),
            stream.getsampwidth(),
        )


def _unclear_recording(payload: dict) -> bool:
    segments = payload.get("segments")
    if not segments:
        return True
    for segment in segments:
        score = segment.get("avg_logprob")
        no_speech = segment.get("no_speech_prob")
        if (
            not isinstance(score, (float, int))
            or not isinstance(no_speech, (float, int))
            or not math.isfinite(score)
            or not math.isfinite(no_speech)
            or score < MIN_SPEECH_LOGPROB
            or no_speech >= MAX_NO_SPEECH
        ):
            return True
    return False


def _ask_to_repeat(language: Language) -> str:
    return (
        "No entendí bien la grabación. ¿Puedes decirlo de nuevo?"
        if language == "es"
        else "I couldn't make out the recording. Could you say it again?"
    )


@router.post("/turn", response_model=TapVoiceResponse)
async def voice_turn(request: Request, response: Response) -> TapVoiceResponse:
    response.headers["Cache-Control"] = "no-store"
    settings = get_settings()
    if not settings.vita_private_tap_enabled:
        raise HTTPException(404, "private_voice_disabled")
    if slots.locked():
        raise HTTPException(503, "private_voice_busy")

    language = request.headers.get("x-vita-language")
    if language not in {"es", "en"}:
        raise HTTPException(400, "invalid_voice_language")
    media = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media not in FORMATS:
        raise HTTPException(415, "invalid_audio_type")

    try:
        async with asyncio.timeout(27), slots:
            data = bytearray()
            async for part in request.stream():
                if len(data) + len(part) > MAX_BYTES:
                    raise HTTPException(413, "audio_too_large")
                data.extend(part)
            if not data:
                raise InvalidAudioError()
            wav = await decode_audio(bytes(data), media)
            # Sidecars sin salida a internet. No proxies de entorno, redirecciones,
            # credenciales de proveedores, ni URL recibida del usuario.
            async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as http:
                stt = await http.post(
                    f"http://127.0.0.1:{settings.vita_tap_stt_port}/inference",
                    data={
                        "language": language,
                        "response_format": "verbose_json",
                        "temperature": "0.0",
                        "temperature_inc": "0.0",
                        "no_language_probabilities": "true",
                    },
                    files={"file": ("utterance.wav", wav, "audio/wav")},
                    timeout=10,
                )
                stt.raise_for_status()
                speech = stt.json()
                transcript = " ".join(speech["text"].split())
                if len(transcript) > MAX_CHARS:
                    raise InvalidAudioError()
                reason = protected_input_reason(transcript)
                if not transcript or _unclear_recording(speech):
                    # Ante un audio ambiguo no entregar al modelo una frase que
                    # podría invertir lo que quiso decir la persona.
                    reply = _ask_to_repeat(language)
                    if reason not in {"emergency_local", "emergency_us"}:
                        reason = None
                        transcript = ""
                elif reason is None:
                    answer = await http.post(
                        f"http://127.0.0.1:{settings.vita_tap_chat_port}/v1/chat/completions",
                        json={
                            "model": settings.vita_tap_model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": PATIENT_SCOPE_POLICY
                                    + "\nNo access to patient records, tools, or plans. "
                                    + "Do not claim you looked up or saved anything. "
                                    + "Ask one clarification if the transcript is unclear. "
                                    + "No invented targets, quantities, or schedules. "
                                    + "Use at most two short sentences without Markdown. "
                                    + "Reply for speech in "
                                    + ("Spanish." if language == "es" else "English."),
                                },
                                {"role": "user", "content": transcript},
                            ],
                            "max_tokens": 120,
                            "temperature": 0.1,
                            "stream": False,
                        },
                        timeout=12,
                    )
                    answer.raise_for_status()
                    choice = answer.json()["choices"][0]
                    reply = choice["message"]["content"].strip()
                    if not reply or len(reply) > MAX_CHARS:
                        raise ValueError()
                    if choice.get("finish_reason") == "length":
                        reason = "provider_failure"
                    else:
                        reason = emergency_delivery_reason(
                            reply, transcript, ""
                        ) or classify_output(reply)
                else:
                    reply = ""
                if reason is not None:
                    reply = fallback_text(reason, language)
            try:
                audio = await speak(reply, language)
            except InvalidAudioError as exc:
                raise OSError("local synthesis failed") from exc
    except InvalidAudioError as exc:
        raise HTTPException(422, "invalid_voice_recording") from exc
    except (
        httpx.HTTPError,
        TimeoutError,
        OSError,
        ValueError,
        KeyError,
        IndexError,
        AttributeError,
        TypeError,
        wave.Error,
    ) as exc:
        logger.warning("vita_private_turn_unavailable error_type=%s", type(exc).__name__)
        raise HTTPException(503, "private_voice_unavailable") from exc
    return TapVoiceResponse(
        transcript=transcript,
        reply=reply,
        audio_base64=base64.b64encode(audio).decode("ascii"),
    )
