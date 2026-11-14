"""Admisión/fencing reales sobre PostgreSQL aislado; ningún paciente/proveedor real."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic.config import Config
from langchain_core.messages import AIMessage
from pydantic import Field
from sqlalchemy import create_engine, inspect, select, text

from alembic import command
from app.api.routes.vita_voice import ProviderMessage, history_identity
from app.graph.graph import build_graph
from app.voice import store
from app.voice.store import VoiceBinding, VoiceContextError
from app.voice.turns import AdmittedVoiceGraph, prepare_admitted_turn
from tests import test_vita_history_postgres as history_tests
from tests.fakes import RecordingFakeModel

rig = history_tests.rig
voice = history_tests.voice
isolated_test_url = history_tests.isolated_test_url


def identity(*inputs):
    return history_identity([ProviderMessage(role="user", content=t) for t in inputs])


async def binding(rig):
    sid = await voice(rig)
    row = await store.voice_store.get(sid)
    return VoiceBinding(row.id, row.user_id, row.thread_id, row.conversation_id)


async def admit(owner, *inputs, counter=1, phase="utterance", repository=None):
    ordinal, prefix, previous = identity(*inputs)
    return await (repository or store.voice_store).admit_turn(
        owner, ordinal, counter, prefix, previous, phase=phase
    )


def config(owner, turn):
    return {
        "configurable": {
            "thread_id": f"{owner.user_id}::{owner.thread_id}",
            "user_id": owner.user_id,
            "voice_session_id": owner.session_id,
            "message_channel": "voice",
            "user_message_id": f"voice:{owner.session_id}:{turn.ordinal}:user",
            "assistant_message_id": turn.response_message_id,
        }
    }


async def test_repeated_agent_counter_accepts_new_utterance_and_preserves_replay(rig):
    owner = await binding(rig)
    first = await admit(owner, "fixture one")
    assert first.action == "run"
    assert await store.voice_store.finish_admission(first, "completed")
    second = await admit(owner, "fixture one", "fixture two")
    assert second.action == "run" and second.counter_reused and second.previous_hash_matches
    # Ni siquiera un cambio de contador transforma un duplicado en generación.
    duplicate = await admit(owner, "fixture one", "fixture two", counter=9)
    assert duplicate.action == "wait"
    assert await store.voice_store.finish_admission(second, "completed")
    assert (await admit(owner, "fixture one", "fixture two", counter=10)).action == "replay"
    async with rig.sessions() as session:
        rows = (
            (
                await session.execute(
                    select(store.VitaVoiceTurn).order_by(store.VitaVoiceTurn.ordinal)
                )
            )
            .scalars()
            .all()
        )
        assert [r.provider_turn for r in rows] == [1, 1]
        assert [r.ordinal for r in rows] == [1, 2]


async def test_concurrent_duplicates_have_one_run_waiters_reuse_result_across_instances(rig):
    owner = await binding(rig)
    replicas = [store.VoiceStore() for _ in range(8)]
    admissions = await asyncio.gather(
        *[admit(owner, "fixture", repository=replica) for replica in replicas]
    )
    assert [a.action for a in admissions].count("run") == 1
    assert [a.action for a in admissions].count("wait") == 7
    runner = next(a for a in admissions if a.action == "run")
    waiters = [
        asyncio.create_task(replicas[i].wait_for_turn(a))
        for i, a in enumerate(admissions)
        if a.action == "wait"
    ]
    await store.voice_store.finish_admission(runner, "completed")
    assert all(a.action == "replay" for a in await asyncio.gather(*waiters))
    assert (await admit(owner, "fixture", repository=store.VoiceStore())).action == "replay"


async def test_valid_next_prefix_supersedes_running_without_allowing_old_replay(rig):
    owner = await binding(rig)
    first = await admit(owner, "fixture one")
    waiter = await admit(owner, "fixture one")
    second = await admit(owner, "fixture one", "fixture two", counter=0)
    assert second.action == "run"  # El contador externo no es autoridad ordinal.
    assert (await store.voice_store.wait_for_turn(waiter)).action == "obsolete"
    assert (await admit(owner, "fixture one")).action == "obsolete"
    with pytest.raises(VoiceContextError, match="obsolete"):
        await store.voice_store.check_current(first)
    assert not await store.voice_store.finish_admission(first, "completed")
    async with rig.sessions() as session:
        assert (
            await session.get(store.VitaVoiceTurn, (owner.session_id, 1))
        ).status == "superseded"


async def test_invalid_previous_prefix_rejects_without_superseding_and_has_safe_flags(rig):
    owner = await binding(rig)
    first = await admit(owner, "fixture one")
    ordinal, prefix, previous = identity("modified fixture", "fixture two")
    with pytest.raises(VoiceContextError, match="voice_history_changed") as exc:
        await store.voice_store.admit_turn(owner, ordinal, 1, prefix, previous)
    assert exc.value.metadata == {"previous_hash_matches": False, "counter_reused": True}
    assert "fixture" not in str(exc.value) and prefix not in str(exc.value)
    await store.voice_store.check_current(first)
    with pytest.raises(VoiceContextError, match="voice_turn_conflict"):
        await store.voice_store.admit_turn(
            owner, 1, 1, identity("changed fixture")[1], identity("wrong prior")[1]
        )
    with pytest.raises(VoiceContextError, match="voice_history_sequence"):
        await admit(owner, "fixture one", "fixture two", "fixture three")


async def test_control_roundtrip_reserves_one_logical_utterance_and_no_extra_run(rig):
    owner = await binding(rig)
    control = await admit(owner, "English fixture", phase="control")
    duplicate = await admit(owner, "English fixture", phase="control", counter=2)
    assert control.action == duplicate.action == "control"
    assert control.response_message_id == duplicate.response_message_id
    await store.voice_store.check_current(control)
    runner = await admit(owner, "English fixture", counter=2)
    assert runner.action == "run" and runner.ordinal == 1
    with pytest.raises(VoiceContextError, match="obsolete"):
        await store.voice_store.check_current(control)
    assert (await admit(owner, "English fixture", phase="control")).action == "obsolete"
    assert (await admit(owner, "English fixture")).action == "wait"
    assert rig.model.prompts == []


@pytest.mark.parametrize("initial_phase", ["utterance", "control"])
@pytest.mark.parametrize("new_phase", ["utterance", "control"])
async def test_latest_provisional_utterance_can_be_amended_with_revision_response_id(
    rig, initial_phase, new_phase
):
    owner = await binding(rig)
    old = await admit(owner, "provisional fixture", phase=initial_phase)
    duplicate = await admit(owner, "provisional fixture", phase=initial_phase)
    new = await admit(owner, "final amended fixture", phase=new_phase)
    assert new.action == ("control" if new_phase == "control" else "run")
    assert new.ordinal == old.ordinal and new.response_message_id != old.response_message_id
    assert new.response_message_id.startswith(old.response_message_id + ":")
    assert len(new.response_message_id) <= 128
    assert new.previous_hash_matches and new.counter_reused
    with pytest.raises(VoiceContextError, match="obsolete"):
        await store.voice_store.check_current(old)
    if duplicate.action == "wait":
        assert (await store.voice_store.wait_for_turn(duplicate)).action == "obsolete"
    if old.action == "run":
        assert not await store.voice_store.finish_admission(old, "completed")
        assert not await store.voice_store.finish_admission(old, "interrupted")
        with pytest.raises(VoiceContextError, match="obsolete"):
            async with store.voice_store.write_fence(old):
                pytest.fail("El hash viejo no puede escribir")
    await store.voice_store.check_current(new)
    async with rig.sessions() as session:
        row = await session.get(store.VitaVoiceTurn, (owner.session_id, old.ordinal))
        assert row.status == ("control" if new_phase == "control" else "running")
        assert row.prefix_hash == new.prefix_hash and row.provider_turn == 1
        assert row.response_message_id == new.response_message_id
    if new_phase == "control":
        revision_id = new.response_message_id
        new = await admit(owner, "final amended fixture")
        assert new.action == "run"
        assert new.response_message_id == revision_id
    duplicate_new = await admit(owner, "final amended fixture")
    assert (
        duplicate_new.action == "wait"
        and duplicate_new.response_message_id == new.response_message_id
    )
    assert await store.voice_store.finish_admission(new, "completed")
    assert (await admit(owner, "final amended fixture")).action == "replay"


@pytest.mark.parametrize("state", ["completed", "interrupted", "earlier"])
async def test_amending_finished_or_earlier_utterance_still_conflicts(rig, state):
    owner = await binding(rig)
    first = await admit(owner, "original fixture")
    if state == "earlier":
        current = await admit(owner, "original fixture", "next fixture")
    else:
        await store.voice_store.finish_admission(first, state)
        current = first
    with pytest.raises(VoiceContextError, match="voice_turn_conflict") as exc:
        await admit(owner, "changed fixture")
    assert exc.value.status == 409 and exc.value.metadata["previous_hash_matches"]
    async with rig.sessions() as session:
        row = await session.get(store.VitaVoiceTurn, (owner.session_id, 1))
        assert row.prefix_hash == first.prefix_hash
    if state == "earlier":
        await store.voice_store.check_current(current)


async def test_amendment_requires_exact_nonempty_previous_prefix(rig):
    owner = await binding(rig)
    first = await admit(owner, "immutable previous fixture")
    await store.voice_store.finish_admission(first, "completed")
    current = await admit(owner, "immutable previous fixture", "provisional current fixture")
    with pytest.raises(VoiceContextError, match="voice_turn_conflict") as exc:
        await admit(owner, "modified previous fixture", "amended current fixture")
    assert exc.value.status == 409 and not exc.value.metadata["previous_hash_matches"]
    await store.voice_store.check_current(current)
    valid = await admit(owner, "immutable previous fixture", "amended current fixture")
    assert valid.action == "run" and valid.response_message_id != current.response_message_id
    with pytest.raises(VoiceContextError, match="obsolete"):
        await store.voice_store.check_current(current)


@pytest.mark.parametrize("phase", ["utterance", "control"])
async def test_aba_hash_cannot_restore_old_claim_or_waiter(rig, phase):
    owner = await binding(rig)
    first_a = await admit(owner, "fixture A", phase=phase)
    waiter_a = await admit(owner, "fixture A") if phase == "utterance" else None
    middle_b = await admit(owner, "fixture B", phase=phase)
    final_a = await admit(owner, "fixture A", phase=phase)
    assert first_a.prefix_hash == final_a.prefix_hash
    assert len({a.response_message_id for a in (first_a, middle_b, final_a)}) == 3
    for old in (first_a, middle_b):
        with pytest.raises(VoiceContextError, match="obsolete"):
            await store.voice_store.check_current(old)
        if old.action == "run":
            assert not await store.voice_store.finish_admission(old, "completed")
            assert not await store.voice_store.finish_admission(old, "interrupted")
            with pytest.raises(VoiceContextError, match="obsolete"):
                async with store.voice_store.write_fence(old):
                    pytest.fail("El claim A inicial no revive con el mismo hash")
    if waiter_a:
        assert (await store.voice_store.wait_for_turn(waiter_a)).action == "obsolete"
    await store.voice_store.check_current(final_a)
    if phase == "control":
        final_a = await admit(owner, "fixture A")
        assert final_a.action == "run"
    duplicate = await admit(owner, "fixture A")
    assert (
        duplicate.action == "wait" and duplicate.response_message_id == final_a.response_message_id
    )
    assert await store.voice_store.finish_admission(final_a, "completed")
    replay = await admit(owner, "fixture A")
    assert replay.action == "replay" and replay.response_message_id == final_a.response_message_id
    with pytest.raises(VoiceContextError, match="obsolete"):
        await store.voice_store.check_current(first_a, completed_ok=True)


@pytest.mark.parametrize("field", ["user_id", "thread_id", "conversation_id"])
async def test_admission_and_wait_enforce_session_owner_and_conversation(rig, field):
    owner = await binding(rig)
    with pytest.raises(VoiceContextError, match="voice_context_binding"):
        await admit(replace(owner, **{field: "other"}), "fixture")
    runner = await admit(owner, "fixture")
    duplicate = await admit(owner, "fixture")
    with pytest.raises(VoiceContextError, match="voice_context_binding"):
        await store.voice_store.wait_for_turn(
            replace(duplicate, binding=replace(owner, **{field: "other"}))
        )
    await store.voice_store.check_current(runner)


async def test_wait_does_not_hold_transaction_revocation_and_timeout_never_regenerate(rig):
    owner = await binding(rig)
    runner = await admit(owner, "fixture")
    duplicate = await admit(owner, "fixture")
    with pytest.raises(TimeoutError):
        await store.voice_store.wait_for_turn(duplicate, timeout=0.03, interval=0.01)
    with pytest.raises(VoiceContextError, match="run_claim"):
        AdmittedVoiceGraph(rig.graph, store.voice_store, duplicate)
    waiting = asyncio.create_task(store.voice_store.wait_for_turn(duplicate))
    await asyncio.wait_for(store.voice_store.revoke(owner.session_id, owner.user_id), 1)
    with pytest.raises(VoiceContextError, match="inactive"):
        await waiting
    with pytest.raises(VoiceContextError, match="inactive"):
        await store.voice_store.check_current(runner)


async def test_interrupted_receipt_is_never_reclaimed_for_same_utterance(rig):
    owner = await binding(rig)
    first = await admit(owner, "fixture one")
    await store.voice_store.finish_admission(first, "interrupted")
    assert (await admit(owner, "fixture one")).action == "obsolete"
    assert (await admit(owner, "fixture one", "fixture two")).action == "run"


async def test_waiters_replay_real_checkpoint_with_exactly_one_model_invocation(rig):
    owner = await binding(rig)
    claims = await asyncio.gather(*[admit(owner, "fixture") for _ in range(6)])

    async def callback(claim):
        if claim.action == "wait":
            claim = await store.voice_store.wait_for_turn(claim)
        cfg = config(owner, claim)
        await prepare_admitted_turn(store.voice_store, claim, cfg, rig.saver)
        if claim.action == "run":
            result = await AdmittedVoiceGraph(rig.graph, store.voice_store, claim).ainvoke(
                {"input": "fixture", "agent": "base"}, config=cfg
            )
            assert await store.voice_store.finish_admission(claim, "completed")
            return result["messages"][-1].content
        await store.voice_store.check_current(claim, completed_ok=True)
        snapshot = await rig.graph.aget_state(cfg)
        return next(
            m.content for m in snapshot.values["messages"] if m.id == claim.response_message_id
        )

    results = await asyncio.gather(*[callback(claim) for claim in claims])
    assert results == ["fixture response"] * 6
    assert len(rig.model.prompts) == 1


async def test_month_rollover_revokes_prepared_claim_without_restoring_checkpoint(rig):
    owner = await binding(rig)
    claim = await admit(owner, "fixture")
    cfg = config(owner, claim)
    await prepare_admitted_turn(store.voice_store, claim, cfg, rig.saver)
    rig.now[0] = datetime(2026, 12, 1, 5, tzinfo=UTC)
    with pytest.raises(VoiceContextError, match="inactive"):
        await prepare_admitted_turn(store.voice_store, claim, cfg, rig.saver)
    assert rig.model.prompts == []
    assert (await rig.history.get(owner.user_id, owner.thread_id)).reset_reason == "monthly"


class SlowFirstModel(RecordingFakeModel):
    calls: int = 0
    entered: asyncio.Event = Field(default_factory=asyncio.Event)
    cancelled: asyncio.Event = Field(default_factory=asyncio.Event)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("amend", [False, True])
async def test_supersession_cancels_llm_new_run_waits_thread_and_saver_blocks_old_writes(
    rig, stream, amend
):
    owner = await binding(rig)
    model = SlowFirstModel(responses=[AIMessage(content="new fixture answer")])
    graph = build_graph(model=model, checkpointer=rig.saver, prepare_tools=[])
    first = await admit(owner, "fixture one")
    cfg1 = config(owner, first)
    await prepare_admitted_turn(store.voice_store, first, cfg1, rig.saver)
    old = AdmittedVoiceGraph(graph, store.voice_store, first)

    async def execute(wrapped, cfg, text):
        if not stream:
            return await wrapped.ainvoke({"input": text, "agent": "base"}, config=cfg)
        latest = None
        async for state in wrapped.astream(
            {"input": text, "agent": "base"}, config=cfg, stream_mode="values"
        ):
            latest = state
        return latest

    running = asyncio.create_task(execute(old, cfg1, "fixture one"))
    try:
        await asyncio.wait_for(model.entered.wait(), 3)
        # El nodo puede entrar antes de que el saver asíncrono haya terminado
        # el checkpoint con Human; un checkpoint inicial vacío no acredita esa
        # escritura. Esperar el mensaje hace determinista la aserción de retención.
        async with asyncio.timeout(3):
            while True:
                before = await rig.saver.aget_tuple(cfg1)
                if before and any(
                    m.type == "human" and m.content == "fixture one"
                    for m in before.checkpoint.get("channel_values", {}).get("messages", [])
                ):
                    break
                await asyncio.sleep(0.01)
        second_inputs = ("fixture one amended",) if amend else ("fixture one", "fixture two")
        second = await admit(owner, *second_inputs)
        if amend:
            assert second.ordinal == first.ordinal
            assert second.response_message_id != first.response_message_id
        cfg2 = config(owner, second)
        await asyncio.wait_for(prepare_admitted_turn(store.voice_store, second, cfg2, rig.saver), 3)
        assert model.cancelled.is_set()
        assert not await store.voice_store.finish_admission(first, "interrupted")
        await store.voice_store.check_current(second)
        if amend:
            before_new = await rig.saver.aget_tuple(cfg2)
            with pytest.raises(VoiceContextError, match="obsolete"):
                await old.checkpointer.aput(before.config, before.checkpoint, before.metadata, {})
            with pytest.raises(VoiceContextError, match="obsolete"):
                await old.checkpointer.aput_writes(before.config, [], "old-amendment-task")
            assert (await rig.saver.aget_tuple(cfg2)).config == before_new.config
        result = await execute(
            AdmittedVoiceGraph(graph, store.voice_store, second), cfg2, second_inputs[-1]
        )
        assert await store.voice_store.finish_admission(second, "completed")
        assert result["messages"][-1].content == "new fixture answer" and model.calls == 2
        assert [m.content for m in result["messages"] if m.type == "human"] == list(second_inputs)
        replay = await admit(owner, *second_inputs)
        assert (
            replay.action == "replay" and replay.response_message_id == second.response_message_id
        )
        assert graph.checkpointer is rig.saver  # No contaminar grafos compartidos.
        with pytest.raises(VoiceContextError, match="obsolete"):
            await old.checkpointer.aput(before.config, before.checkpoint, before.metadata, {})
        with pytest.raises(VoiceContextError, match="obsolete"):
            await old.checkpointer.aput_writes(before.config, [], "fixture-task")
        with pytest.raises(VoiceContextError, match="obsolete"):
            await running
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


async def test_write_fence_linearizes_persistence_before_admission_and_blocks_after(rig):
    owner = await binding(rig)
    first = await admit(owner, "fixture one")
    async with store.voice_store.write_fence(first):
        next_turn = asyncio.create_task(admit(owner, "fixture one", "fixture two"))
        await asyncio.sleep(0.08)
        assert not next_turn.done()
    assert (await asyncio.wait_for(next_turn, 1)).action == "run"
    with pytest.raises(VoiceContextError, match="obsolete"):
        async with store.voice_store.write_fence(first):
            pytest.fail("No escribir después del commit de supersesión")


async def test_clear_or_expiry_invalidates_admission_before_checkpoint_access(rig):
    owner = await binding(rig)
    first = await admit(owner, "fixture")
    cfg = config(owner, first)
    await prepare_admitted_turn(store.voice_store, first, cfg, rig.saver)
    response = await rig.client.delete("/api/v1/threads/thread/state?user_id=owner")
    assert response.status_code == 200
    with pytest.raises(VoiceContextError, match="inactive"):
        await AdmittedVoiceGraph(rig.graph, store.voice_store, first).ainvoke(
            {"input": "fixture"}, config=cfg
        )
    another = await binding(rig)
    async with rig.sessions() as session:
        row = await session.get(store.VitaVoiceSession, another.session_id)
        row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await session.commit()
    with pytest.raises(VoiceContextError, match="inactive"):
        await admit(another, "fixture")


def test_counter_migration_upgrade_and_non_destructive_rollback():
    url = isolated_test_url()
    root = Path(__file__).resolve().parent.parent
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    # Solo la migración de voz real previa; la base de producto no se recrea.
    command.stamp(cfg, "e7041890ceaa")
    command.upgrade(cfg, "18c4f6a91b20")
    engine = create_engine(url)
    try:
        with engine.connect() as sql:
            assert any(
                c["column_names"] == ["session_id", "provider_turn"]
                for c in inspect(sql).get_unique_constraints("vita_voice_turns", schema="ai")
            )
        command.upgrade(cfg, "94e1c2b75a06")
        command.downgrade(cfg, "18c4f6a91b20")
        command.upgrade(cfg, "94e1c2b75a06")
        with engine.begin() as sql:
            sql.execute(
                text(
                    "INSERT INTO ai.vita_voice_sessions "
                    "(id,user_id,thread_id,conversation_id,agent_id,language,expires_at) "
                    "VALUES ('fixture','owner','thread','conversation','agent','es',"
                    "now()+interval '1 minute')"
                )
            )
            for ordinal in (1, 2):
                sql.execute(
                    text(
                        "INSERT INTO ai.vita_voice_turns "
                        "(session_id,ordinal,provider_turn,prefix_hash,status,response_message_id) "
                        "VALUES ('fixture',:ordinal,1,'fixture','completed','fixture')"
                    ),
                    {"ordinal": ordinal},
                )
        with pytest.raises(Exception, match="voice_counter_duplicates"):
            command.downgrade(cfg, "18c4f6a91b20")
        with engine.connect() as sql:
            assert sql.scalar(text("SELECT count(*) FROM ai.vita_voice_turns")) == 2
            assert sql.scalar(text("SELECT version_num FROM ai.alembic_version")) == "94e1c2b75a06"
    finally:
        with engine.begin() as sql:
            sql.execute(text("DELETE FROM ai.vita_voice_sessions WHERE id='fixture'"))
        command.downgrade(cfg, "e7041890ceaa")
        engine.dispose()
