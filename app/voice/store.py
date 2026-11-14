"""Sesiones y recibos durables: ninguna garantía depende del proceso que atiende."""

import asyncio
import hashlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
    or_,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.engine import async_session


class VitaVoiceSession(Base):
    __tablename__ = "vita_voice_sessions"
    __table_args__ = (
        Index("ix_vita_voice_sessions_owner_thread_created", "user_id", "thread_id", "created_at"),
        {"schema": "ai"},
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    patient_id: Mapped[str | None] = mapped_column(String(128))
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    agent_id: Mapped[str] = mapped_column(String(128), nullable=False)
    version_id: Mapped[str | None] = mapped_column(String(128))
    runtime_mode: Mapped[str] = mapped_column(
        String(16), nullable=False, default="vita", server_default="vita"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    transcript_digest: Mapped[str | None] = mapped_column(String(64))
    transcript_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    language: Mapped[str] = mapped_column(String(2), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class VitaVoiceTurn(Base):
    __tablename__ = "vita_voice_turns"
    __table_args__ = ({"schema": "ai"},)

    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("ai.vita_voice_sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider_turn: Mapped[int] = mapped_column(Integer, nullable=False)
    prefix_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    response_message_id: Mapped[str] = mapped_column(String(128), nullable=False)


class VoiceContextError(Exception):
    """Error público acotado, nunca contiene contexto/token/historial."""

    def __init__(self, code: str, status: int = 403, *, metadata: dict | None = None):
        super().__init__(code)
        self.status = status
        # Solo flags calculados por el servidor, nunca hashes/texto/IDs.
        self.metadata = metadata or {}


@dataclass(frozen=True)
class VoiceBinding:
    """Identidad resuelta del registro tras verificar HMAC, nunca del body."""

    session_id: str
    user_id: str
    thread_id: str
    conversation_id: str


@dataclass(frozen=True)
class TurnAdmission:
    action: Literal["run", "replay", "wait", "obsolete", "control"]
    binding: VoiceBinding
    ordinal: int
    prefix_hash: str
    response_message_id: str
    previous_hash_matches: bool
    counter_reused: bool

    @property
    def metadata(self) -> dict[str, bool]:
        return {
            "previous_hash_matches": self.previous_hash_matches,
            "counter_reused": self.counter_reused,
        }


class VoiceStore:
    async def _binding_locked(self, session, binding: VoiceBinding):
        record = await session.get(VitaVoiceSession, binding.session_id, with_for_update=True)
        if record is None or record.revoked or record.expires_at <= datetime.now(UTC):
            raise VoiceContextError("voice_session_inactive")
        if (record.user_id, record.thread_id, record.conversation_id) != (
            binding.user_id,
            binding.thread_id,
            binding.conversation_id,
        ):
            raise VoiceContextError("voice_context_binding")
        return record

    async def _latest(self, session, session_id: str):
        return (
            await session.execute(
                select(VitaVoiceTurn)
                .where(VitaVoiceTurn.session_id == session_id)
                .order_by(VitaVoiceTurn.ordinal.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    async def admit_turn(
        self,
        binding: VoiceBinding,
        ordinal: int,
        provider_turn: int,
        prefix_hash: str,
        previous_hash: str,
        *,
        phase: Literal["utterance", "control"] = "utterance",
    ) -> TurnAdmission:
        """Una decisión durable; el contador externo solo es metadato.

        La fila de sesión serializa admisión, supersesión y fencing de escrituras.
        Nunca se espera un LLM ni un advisory lock dentro de esta transacción.
        El llamador entrega el ordinal/prefijo CANÓNICOS y una fase validada.
        """
        if (
            phase not in {"utterance", "control"}
            or type(ordinal) is not int
            or ordinal < 1
            or type(provider_turn) is not int
            or provider_turn < 0
        ):
            raise VoiceContextError("voice_turn_invalid", 422)
        for digest in (prefix_hash, previous_hash):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise VoiceContextError("voice_hash_invalid", 422)
        async with async_session() as session:
            await self._binding_locked(session, binding)
            latest = await self._latest(session, binding.session_id)
            existing = await session.get(VitaVoiceTurn, (binding.session_id, ordinal))
            prior = (
                await session.get(VitaVoiceTurn, (binding.session_id, ordinal - 1))
                if existing and ordinal > 1
                else latest
                if not existing
                else None
            )
            expected = prior.prefix_hash if prior else hashlib.sha256(b"[]").hexdigest()
            flags = {
                "previous_hash_matches": previous_hash == expected,
                "counter_reused": latest is not None and provider_turn == latest.provider_turn,
            }
            if existing:
                if not flags["previous_hash_matches"]:
                    raise VoiceContextError("voice_turn_conflict", 409, metadata=flags)
                if existing.prefix_hash != prefix_hash:
                    # ElevenLabs puede completar la última utterance mientras
                    # aún se genera. Solo esa fila provisional es reemplazable;
                    # el prefijo anterior queda intacto. El ID actúa como epoch
                    # durable: A→B→A no vuelve a habilitar al primer runner A.
                    if latest.ordinal != ordinal or existing.status not in {"running", "control"}:
                        raise VoiceContextError("voice_turn_conflict", 409, metadata=flags)
                    existing.prefix_hash = prefix_hash
                    existing.status = "control" if phase == "control" else "running"
                    existing.response_message_id = (
                        f"voice:{binding.session_id}:{ordinal}:bot:{uuid4().hex}"
                    )
                    action = "control" if phase == "control" else "run"
                elif latest.ordinal != ordinal or existing.status in {"superseded", "interrupted"}:
                    action = "obsolete"
                elif existing.status == "completed":
                    action = "replay"
                elif existing.status == "running":
                    action = "wait" if phase == "utterance" else "obsolete"
                elif existing.status == "control" and phase == "control":
                    action = "control"
                elif existing.status == "control":
                    existing.status = "running"
                    action = "run"
                else:
                    raise VoiceContextError("voice_turn_invalid_state", 409, metadata=flags)
                turn = existing
            else:
                if ordinal != (latest.ordinal + 1 if latest else 1):
                    raise VoiceContextError("voice_history_sequence", 409, metadata=flags)
                if not flags["previous_hash_matches"]:
                    raise VoiceContextError("voice_history_changed", 409, metadata=flags)
                if latest and latest.status in {"control", "running"}:
                    latest.status = "superseded"
                turn = VitaVoiceTurn(
                    session_id=binding.session_id,
                    ordinal=ordinal,
                    provider_turn=provider_turn,
                    prefix_hash=prefix_hash,
                    status="control" if phase == "control" else "running",
                    response_message_id=f"voice:{binding.session_id}:{ordinal}:bot",
                )
                session.add(turn)
                action = "control" if phase == "control" else "run"
            await session.commit()
            return TurnAdmission(
                action, binding, ordinal, prefix_hash, turn.response_message_id, **flags
            )

    async def _current_locked(self, session, admission: TurnAdmission, *, completed_ok=False):
        await self._binding_locked(session, admission.binding)
        latest = await self._latest(session, admission.binding.session_id)
        allowed = {"running", "completed"} if completed_ok else {"running"}
        if admission.action == "control":
            allowed = {"control"}
        if (
            latest is None
            or latest.ordinal != admission.ordinal
            or latest.prefix_hash != admission.prefix_hash
            or latest.response_message_id != admission.response_message_id
            or latest.status not in allowed
        ):
            raise VoiceContextError("voice_turn_obsolete", 409, metadata=admission.metadata)
        return latest

    async def check_current(self, admission: TurnAdmission, *, completed_ok=False) -> None:
        async with async_session() as session:
            await self._current_locked(session, admission, completed_ok=completed_ok)

    @asynccontextmanager
    async def write_fence(self, admission: TurnAdmission):
        """Linealiza cada escritura del saver con admisiones/revocaciones.

        Solo cubre IO corto de persistencia; nunca espera generación ni tokens.
        Una admisión posterior no puede confirmar mientras esta escritura sigue.
        """
        if admission.action != "run":
            raise VoiceContextError("voice_run_claim_required", 409)
        async with async_session() as session:
            await self._current_locked(session, admission)
            yield

    async def finish_admission(
        self, admission: TurnAdmission, status: Literal["completed", "interrupted"]
    ) -> bool:
        if admission.action != "run" or status not in {"completed", "interrupted"}:
            raise VoiceContextError("voice_run_claim_required", 409)
        async with async_session() as session:
            await self._binding_locked(session, admission.binding)
            turn = await self._latest(session, admission.binding.session_id)
            if (
                turn is None
                or turn.ordinal != admission.ordinal
                or turn.prefix_hash != admission.prefix_hash
                or turn.response_message_id != admission.response_message_id
                or turn.status != "running"
            ):
                return False  # La limpieza vieja no pisa un turno nuevo/supersedido.
            turn.status = status
            await session.commit()
            return True

    async def wait_for_turn(self, admission: TurnAdmission, *, timeout=20.0, interval=0.05):
        """Observa sin retener transacción/conexión entre polls; nunca reclama run."""
        if admission.action != "wait":
            raise VoiceContextError("voice_wait_claim_required", 409)
        async with asyncio.timeout(timeout):
            while True:
                async with async_session() as session:
                    await self._binding_locked(session, admission.binding)
                    latest = await self._latest(session, admission.binding.session_id)
                    if (
                        latest is None
                        or latest.ordinal != admission.ordinal
                        or latest.prefix_hash != admission.prefix_hash
                        or latest.response_message_id != admission.response_message_id
                        or latest.status in {"superseded", "interrupted"}
                    ):
                        return replace(admission, action="obsolete")
                    if latest.status == "completed":
                        return replace(admission, action="replay")
                await asyncio.sleep(interval)

    async def create(self, record: VitaVoiceSession) -> None:
        async with async_session() as session:
            session.add(record)
            await session.commit()

    async def get(self, session_id: str) -> VitaVoiceSession:
        async with async_session() as session:
            record = await session.get(VitaVoiceSession, session_id)
            if record is None or record.revoked or record.expires_at <= datetime.now(UTC):
                raise VoiceContextError("voice_session_inactive")
            return record

    async def get_owned(self, session_id: str, user_id: str, thread_id: str) -> VitaVoiceSession:
        async with async_session() as session:
            record = await session.get(VitaVoiceSession, session_id)
            if record is None or record.user_id != user_id or record.thread_id != thread_id:
                raise VoiceContextError("voice_session_not_found", 404)
            return record

    async def closed_for_thread(self, user_id: str, thread_id: str) -> list[VitaVoiceSession]:
        async with async_session() as session:
            rows = (
                (
                    await session.execute(
                        select(VitaVoiceSession)
                        .where(
                            VitaVoiceSession.user_id == user_id,
                            VitaVoiceSession.thread_id == thread_id,
                            or_(
                                VitaVoiceSession.revoked.is_(True),
                                VitaVoiceSession.expires_at <= datetime.now(UTC),
                            ),
                        )
                        .order_by(
                            VitaVoiceSession.transcript_synced_at.asc().nulls_first(),
                            VitaVoiceSession.created_at.desc(),
                            VitaVoiceSession.id,
                        )
                        .limit(3)
                    )
                )
                .scalars()
                .all()
            )
            return sorted(rows, key=lambda row: (row.created_at, row.id))

    async def mark_transcript_synced(self, session_id: str, digest: str) -> None:
        async with async_session() as session:
            record = await session.get(VitaVoiceSession, session_id, with_for_update=True)
            if record:
                record.transcript_digest = digest
                record.transcript_synced_at = datetime.now(UTC)
                await session.commit()

    async def revoke(self, session_id: str, user_id: str) -> None:
        async with async_session() as session:
            record = await session.get(VitaVoiceSession, session_id, with_for_update=True)
            if record is None or record.user_id != user_id:
                raise VoiceContextError("voice_session_not_found", 404)
            record.revoked = True
            await session.commit()

    async def start_turn(
        self,
        session_id: str,
        ordinal: int,
        provider_turn: int,
        prefix_hash: str,
        previous_hash: str,
    ) -> VitaVoiceTurn:
        async with async_session() as session:
            # Una fila de sesión serializa decisiones de ordinal entre réplicas.
            record = await session.get(VitaVoiceSession, session_id, with_for_update=True)
            if record is None or record.revoked or record.expires_at <= datetime.now(UTC):
                raise VoiceContextError("voice_session_inactive")
            latest = (
                await session.execute(
                    select(VitaVoiceTurn)
                    .where(VitaVoiceTurn.session_id == session_id)
                    .order_by(VitaVoiceTurn.ordinal.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            existing = await session.get(VitaVoiceTurn, (session_id, ordinal))
            if existing:
                if existing.prefix_hash != prefix_hash or existing.provider_turn != provider_turn:
                    raise VoiceContextError("voice_turn_conflict", 409)
                if existing.status != "completed":
                    raise VoiceContextError("voice_turn_inflight_or_interrupted", 409)
                return existing
            if ordinal != (latest.ordinal + 1 if latest else 1):
                raise VoiceContextError("voice_history_sequence", 409)
            if latest and (
                latest.prefix_hash != previous_hash or provider_turn <= latest.provider_turn
            ):
                raise VoiceContextError("voice_history_changed", 409)
            turn = VitaVoiceTurn(
                session_id=session_id,
                ordinal=ordinal,
                provider_turn=provider_turn,
                prefix_hash=prefix_hash,
                status="running",
                response_message_id=f"voice:{session_id}:{ordinal}:bot",
            )
            session.add(turn)
            await session.commit()
            return turn

    async def finish_turn(self, session_id: str, ordinal: int, status: str) -> None:
        async with async_session() as session:
            turn = await session.get(VitaVoiceTurn, (session_id, ordinal), with_for_update=True)
            if turn and turn.status == "running":
                turn.status = status
                await session.commit()


voice_store = VoiceStore()
