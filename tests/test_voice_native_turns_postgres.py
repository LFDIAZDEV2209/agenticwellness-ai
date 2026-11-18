"""Admisión/fencing reales sobre PostgreSQL aislado; ningún paciente/proveedor real."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic.config import Config
from anyio import CancelScope, sleep
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import Field
from sqlalchemy import create_engine, event, inspect, select, text

from alembic import command
from app.api.deps import get_graph
from app.api.routes.vita_voice import ProviderMessage, history_identity
from app.graph.graph import build_graph
from app.voice import store
from app.voice.store import VoiceBinding, VoiceContextError
from app.voice.turns import AdmittedVoiceGraph, _thread_slot, prepare_admitted_turn
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


async def test_thread_slot_anyio_cancel_releases_without_discarding_connection(rig):
    owner = await binding(rig)
    claim = await admit(owner, "fixture")
    invalidated = []
    event.listen(rig.engine.sync_engine, "invalidate", lambda *args: invalidated.append(args))
    cfg = config(owner, claim)
    with CancelScope() as scope:
        async with _thread_slot(store.voice_store, claim, cfg):
            scope.cancel()
            await sleep(0)
    assert not invalidated
    assert rig.engine.pool.checkedout() == 0
    async with _thread_slot(store.voice_store, claim, cfg, timeout=0.1):
        pass


async def test_colgar_durante_tool_y_cambiar_a_chat_conserva_hilo_sano(rig):
    owner = await binding(rig)
    claim = await admit(owner, "Lectura interrumpida")
    cfg = config(owner, claim)
    entered, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @tool
    async def fixture_read() -> str:
        """Lectura sintética que tarda en cerrar después de la cancelación."""
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()
        return "fixture result"

    model = RecordingFakeModel(
        responses=[
            AIMessage(
                content="", tool_calls=[{"name": "fixture_read", "args": {}, "id": "old-call"}]
            ),
            AIMessage(content="Respuesta de chat"),
        ]
    )
    graph = build_graph(model=model, checkpointer=rig.saver, tools=[fixture_read])
    rig.app.dependency_overrides[get_graph] = lambda: graph
    await prepare_admitted_turn(store.voice_store, claim, cfg, rig.saver)
    voice_run = asyncio.create_task(
        AdmittedVoiceGraph(graph, store.voice_store, claim).ainvoke(
            {"input": "Lectura interrumpida"},
            config=cfg,
        )
    )

    async def finish_cleanup():
        await closing.wait()
        await asyncio.sleep(0.12)
        release.set()

    cleanup = asyncio.create_task(finish_cleanup())
    try:
        async with asyncio.timeout(4):
            await entered.wait()
            while not any(
                getattr(m, "tool_calls", [])
                for m in (await graph.aget_state(cfg)).values["messages"]
            ):
                await asyncio.sleep(0.01)
            await store.voice_store.revoke(owner.session_id, owner.user_id)
            await closing.wait()
            response = await rig.client.post(
                "/api/v1/chat",
                json={
                    "message": "Continúo en chat",
                    "thread_id": owner.thread_id,
                    "user_id": owner.user_id,
                    "supports_vita": True,
                },
            )
        assert response.status_code == 200, response.text
        assert response.json()["answer"] == "Respuesta de chat"
        with pytest.raises(VoiceContextError, match="voice_session_inactive"):
            await voice_run
        messages = (await graph.aget_state(cfg)).values["messages"]
        assert not any(getattr(m, "tool_calls", []) or isinstance(m, ToolMessage) for m in messages)
        assert [m.content for m in messages if m.type == "human"] == [
            "Lectura interrumpida",
            "Continúo en chat",
        ]
        reconnected = await binding(rig)
        assert reconnected.session_id != owner.session_id
        fresh = await admit(reconnected, "Volví a voz")
        fresh_cfg = config(reconnected, fresh)
        fresh_model = RecordingFakeModel(responses=[AIMessage(content="Voz reanudada")])
        fresh_graph = build_graph(model=fresh_model, checkpointer=rig.saver, prepare_tools=[])
        await prepare_admitted_turn(store.voice_store, fresh, fresh_cfg, rig.saver)
        answer = await AdmittedVoiceGraph(fresh_graph, store.voice_store, fresh).ainvoke(
            {"input": "Volví a voz"}, config=fresh_cfg
        )
        assert answer["messages"][-1].content == "Voz reanudada"
        assert await store.voice_store.finish_admission(fresh, "completed")
        assert (await admit(reconnected, "Volví a voz")).action == "replay"
        assert len(fresh_model.prompts) == 1
    finally:
        release.set()
        for task in (cleanup, voice_run):
            if not task.done():
                task.cancel()
        await asyncio.gather(cleanup, voice_run, return_exceptions=True)


async def test_voice_stream_cancelled_close_releases_slot_and_preserves_checkpoint(rig):
    owner = await binding(rig)
    claim = await admit(owner, "fixture")
    cfg = config(owner, claim)
    await history_tests.seed(rig)
    await prepare_admitted_turn(store.voice_store, claim, cfg, rig.saver)
    before = await rig.saver.aget_tuple(cfg)
    closed = asyncio.Event()

    class PausedGraph:
        checkpointer = rig.saver

        async def astream(self, *args, **kwargs):
            try:
                yield "fixture token"
            finally:
                await sleep(0)
                closed.set()

    graph = AdmittedVoiceGraph(PausedGraph(), store.voice_store, claim)
    stream = graph.astream(config=cfg)
    try:
        assert await anext(stream) == "fixture token"
        with CancelScope() as scope:
            scope.cancel()
            await stream.aclose()
        assert closed.is_set()
        assert rig.engine.pool.checkedout() == 0
        async with _thread_slot(store.voice_store, claim, cfg, timeout=0.1):
            pass
        assert (await rig.saver.aget_tuple(cfg)).config == before.config
    finally:
        with CancelScope(shield=True):
            await stream.aclose()


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
    assert exc.value.metadata == {
        "previous_hash_matches": False,
        "counter_reused": True,
        "existing_completed": False,
        "latest_ordinal_matches": False,
        "prefix_hash_matches": False,
    }
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


@pytest.mark.parametrize("initial_state", ["running", "control", "completed", "interrupted"])
@pytest.mark.parametrize("new_phase", ["utterance", "control"])
async def test_latest_utterance_can_be_amended_with_revision_response_id(
    rig, initial_state, new_phase
):
    owner = await binding(rig)
    initial_phase = "control" if initial_state == "control" else "utterance"
    old = await admit(owner, "provisional fixture", phase=initial_phase)
    if initial_state in {"completed", "interrupted"}:
        assert await store.voice_store.finish_admission(old, initial_state)
    duplicate = await admit(owner, "provisional fixture", phase=initial_phase)
    if initial_state == "interrupted":
        assert duplicate.action == "obsolete"
        assert duplicate.response_message_id == old.response_message_id
    if initial_state == "completed":
        assert duplicate.action == "replay"
        assert duplicate.response_message_id == old.response_message_id
        assert duplicate.metadata == {
            "previous_hash_matches": True,
            "counter_reused": True,
            "existing_completed": True,
            "latest_ordinal_matches": True,
            "prefix_hash_matches": True,
        }
    new = await admit(owner, "final amended fixture", phase=new_phase)
    assert new.action == ("control" if new_phase == "control" else "run")
    assert new.ordinal == old.ordinal and new.response_message_id != old.response_message_id
    assert new.response_message_id.startswith(old.response_message_id + ":")
    assert len(new.response_message_id) <= 128
    assert new.previous_hash_matches and new.counter_reused
    assert new.metadata == {
        "previous_hash_matches": True,
        "counter_reused": True,
        "existing_completed": initial_state == "completed",
        "latest_ordinal_matches": True,
        "prefix_hash_matches": False,
    }
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


@pytest.mark.parametrize("state", ["running", "completed", "interrupted"])
async def test_amending_earlier_utterance_still_conflicts(rig, state):
    owner = await binding(rig)
    first = await admit(owner, "original fixture")
    if state != "running":
        assert await store.voice_store.finish_admission(first, state)
    current = await admit(owner, "original fixture", "next fixture")
    with pytest.raises(VoiceContextError, match="voice_turn_conflict") as exc:
        await admit(owner, "changed fixture")
    assert exc.value.status == 409 and exc.value.metadata["previous_hash_matches"]
    assert exc.value.metadata["existing_completed"] == (state == "completed")
    assert not exc.value.metadata["latest_ordinal_matches"]
    assert not exc.value.metadata["prefix_hash_matches"]
    async with rig.sessions() as session:
        row = await session.get(store.VitaVoiceTurn, (owner.session_id, 1))
        assert row.prefix_hash == first.prefix_hash
    await store.voice_store.check_current(current)


@pytest.mark.parametrize("state", ["running", "completed", "interrupted"])
@pytest.mark.parametrize("amend", [False, True])
async def test_amendment_requires_exact_nonempty_previous_prefix(rig, state, amend):
    owner = await binding(rig)
    first = await admit(owner, "immutable previous fixture")
    await store.voice_store.finish_admission(first, "completed")
    current = await admit(owner, "immutable previous fixture", "provisional current fixture")
    if state != "running":
        assert await store.voice_store.finish_admission(current, state)
    with pytest.raises(VoiceContextError, match="voice_turn_conflict") as exc:
        if amend:
            await admit(owner, "modified previous fixture", "amended current fixture")
        else:
            # Ni un duplicado de hash puede omitir la validación del prefijo previo.
            await store.voice_store.admit_turn(
                owner, current.ordinal, 1, current.prefix_hash, identity("wrong prior")[1]
            )
    assert exc.value.status == 409 and not exc.value.metadata["previous_hash_matches"]
    assert exc.value.metadata["existing_completed"] == (state == "completed")
    assert exc.value.metadata["latest_ordinal_matches"]
    assert exc.value.metadata["prefix_hash_matches"] == (not amend)
    if state == "interrupted":
        with pytest.raises(VoiceContextError, match="obsolete"):
            await store.voice_store.check_current(current, completed_ok=True)
    else:
        await store.voice_store.check_current(current, completed_ok=True)
    valid = await admit(owner, "immutable previous fixture", "amended current fixture")
    assert valid.action == "run" and valid.response_message_id != current.response_message_id
    with pytest.raises(VoiceContextError, match="obsolete"):
        await store.voice_store.check_current(current)


@pytest.mark.parametrize("phase", ["utterance", "control", "completed", "interrupted"])
async def test_aba_hash_cannot_restore_old_claim_or_waiter(rig, phase):
    owner = await binding(rig)
    admission_phase = "control" if phase == "control" else "utterance"
    first_a = await admit(owner, "fixture A", phase=admission_phase)
    waiter_a = await admit(owner, "fixture A") if phase != "control" else None
    if phase in {"completed", "interrupted"}:
        assert await store.voice_store.finish_admission(first_a, phase)
    middle_b = await admit(owner, "fixture B", phase=admission_phase)
    if phase in {"completed", "interrupted"}:
        assert await store.voice_store.finish_admission(middle_b, phase)
    final_a = await admit(owner, "fixture A", phase=admission_phase)
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
@pytest.mark.parametrize("state", ["running", "completed"])
async def test_admission_and_wait_enforce_session_owner_and_conversation(rig, field, state):
    owner = await binding(rig)
    runner = await admit(owner, "fixture")
    duplicate = await admit(owner, "fixture")
    if state == "completed":
        assert await store.voice_store.finish_admission(runner, "completed")
    with pytest.raises(VoiceContextError, match="voice_context_binding") as exc:
        await admit(replace(owner, **{field: "other"}), "amended fixture")
    assert exc.value.status == 403 and exc.value.metadata == {}
    with pytest.raises(VoiceContextError, match="voice_context_binding"):
        await store.voice_store.wait_for_turn(
            replace(duplicate, binding=replace(owner, **{field: "other"}))
        )
    await store.voice_store.check_current(runner, completed_ok=True)


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


@pytest.mark.parametrize("inactive", ["revoked", "expired"])
async def test_interrupted_revision_cannot_reactivate_inactive_session(rig, inactive):
    owner = await binding(rig)
    old = await admit(owner, "provisional fixture")
    assert await store.voice_store.finish_admission(old, "interrupted")
    if inactive == "revoked":
        await store.voice_store.revoke(owner.session_id, owner.user_id)
    else:
        async with rig.sessions() as session:
            row = await session.get(store.VitaVoiceSession, owner.session_id)
            row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
            await session.commit()
    with pytest.raises(VoiceContextError, match="voice_session_inactive") as exc:
        await admit(owner, "final amended fixture")
    assert exc.value.status == 403 and exc.value.metadata == {}
    async with rig.sessions() as session:
        row = await session.get(store.VitaVoiceTurn, (owner.session_id, old.ordinal))
        assert row.status == "interrupted" and row.prefix_hash == old.prefix_hash
        assert row.response_message_id == old.response_message_id


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


async def test_completed_revision_runs_once_waiters_replay_and_old_saver_cannot_write(rig):
    owner = await binding(rig)
    original = await admit(owner, "original fixture")
    old_waiter = await admit(owner, "original fixture")
    old_cfg = config(owner, original)
    await prepare_admitted_turn(store.voice_store, original, old_cfg, rig.saver)
    old_graph = AdmittedVoiceGraph(rig.graph, store.voice_store, original)
    await old_graph.ainvoke({"input": "original fixture", "agent": "base"}, config=old_cfg)
    assert await store.voice_store.finish_admission(original, "completed")
    old_replay = await admit(owner, "original fixture", counter=99)
    assert old_replay.action == "replay"
    assert old_replay.response_message_id == original.response_message_id
    assert len(rig.model.prompts) == 1

    replicas = [store.VoiceStore() for _ in range(6)]
    claims = await asyncio.gather(
        *[admit(owner, "amended fixture", repository=replica) for replica in replicas]
    )
    assert [claim.action for claim in claims].count("run") == 1
    assert [claim.action for claim in claims].count("wait") == 5
    revised = next(claim for claim in claims if claim.action == "run")
    assert revised.existing_completed and not revised.prefix_hash_matches
    assert revised.ordinal == original.ordinal
    assert revised.response_message_id != original.response_message_id
    assert (await store.voice_store.wait_for_turn(old_waiter)).action == "obsolete"
    with pytest.raises(VoiceContextError, match="obsolete"):
        await prepare_admitted_turn(store.voice_store, old_replay, old_cfg, rig.saver)

    before = await rig.saver.aget_tuple(old_cfg)
    with pytest.raises(VoiceContextError, match="obsolete"):
        await old_graph.checkpointer.aput(before.config, before.checkpoint, before.metadata, {})
    with pytest.raises(VoiceContextError, match="obsolete"):
        await old_graph.checkpointer.aput_writes(before.config, [], "obsolete-completed-task")
    assert (await rig.saver.aget_tuple(old_cfg)).config == before.config
    assert not await store.voice_store.finish_admission(original, "completed")
    assert not await store.voice_store.finish_admission(original, "interrupted")

    async def callback(claim):
        if claim.action == "wait":
            claim = await store.voice_store.wait_for_turn(claim)
        cfg = config(owner, claim)
        await prepare_admitted_turn(store.voice_store, claim, cfg, rig.saver)
        if claim.action == "run":
            await AdmittedVoiceGraph(rig.graph, store.voice_store, claim).ainvoke(
                {"input": "amended fixture", "agent": "base"}, config=cfg
            )
            assert await store.voice_store.finish_admission(claim, "completed")
        else:
            assert claim.action == "replay"
        await store.voice_store.check_current(claim, completed_ok=True)
        snapshot = await rig.graph.aget_state(cfg)
        return next(
            m.content for m in snapshot.values["messages"] if m.id == claim.response_message_id
        )

    assert await asyncio.gather(*[callback(claim) for claim in claims]) == ["fixture response"] * 6
    assert len(rig.model.prompts) == 2  # Una generación original y una para toda la revisión.
    replay = await admit(owner, "amended fixture", counter=100)
    assert replay.action == "replay" and replay.response_message_id == revised.response_message_id
    assert replay.metadata == {
        "previous_hash_matches": True,
        "counter_reused": False,
        "existing_completed": True,
        "latest_ordinal_matches": True,
        "prefix_hash_matches": True,
    }
    with pytest.raises(VoiceContextError, match="obsolete"):
        await store.voice_store.check_current(old_replay, completed_ok=True)
    async with rig.sessions() as session:
        row = await session.get(store.VitaVoiceTurn, (owner.session_id, original.ordinal))
        assert row.status == "completed" and row.prefix_hash == revised.prefix_hash
        assert row.response_message_id == revised.response_message_id and row.provider_turn == 1


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
async def test_cancelled_runner_final_revision_runs_once_and_fences_old_checkpoint(rig, stream):
    owner = await binding(rig)
    model = SlowFirstModel(responses=[AIMessage(content="final fixture answer")])
    graph = build_graph(model=model, checkpointer=rig.saver, prepare_tools=[])
    original = await admit(owner, "provisional fixture")
    old_waiter = await admit(owner, "provisional fixture")
    old_cfg = config(owner, original)
    await prepare_admitted_turn(store.voice_store, original, old_cfg, rig.saver)
    old_graph = AdmittedVoiceGraph(graph, store.voice_store, original)

    async def execute(wrapped, cfg, utterance):
        if not stream:
            return await wrapped.ainvoke({"input": utterance, "agent": "base"}, config=cfg)
        latest = None
        async for state in wrapped.astream(
            {"input": utterance, "agent": "base"}, config=cfg, stream_mode="values"
        ):
            latest = state
        return latest

    running = asyncio.create_task(execute(old_graph, old_cfg, "provisional fixture"))
    try:
        await asyncio.wait_for(model.entered.wait(), 3)
        async with asyncio.timeout(3):
            while True:
                before = await rig.saver.aget_tuple(old_cfg)
                if before and any(
                    m.type == "human" and m.content == "provisional fixture"
                    for m in before.checkpoint.get("channel_values", {}).get("messages", [])
                ):
                    break
                await asyncio.sleep(0.01)
        # Mismo orden que el callback cancelado: primero se cancela el runner,
        # luego se confirma interrupted, antes de recibir la transcripción final.
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert model.cancelled.is_set()
        assert await store.voice_store.finish_admission(original, "interrupted")
        assert (await admit(owner, "provisional fixture", counter=99)).action == "obsolete"
        assert (await store.voice_store.wait_for_turn(old_waiter)).action == "obsolete"

        claims = await asyncio.gather(
            *[
                admit(owner, "final amended fixture", repository=store.VoiceStore())
                for _ in range(6)
            ]
        )
        assert [c.action for c in claims].count("run") == 1
        assert [c.action for c in claims].count("wait") == 5
        revised = next(c for c in claims if c.action == "run")
        assert revised.ordinal == original.ordinal
        assert revised.response_message_id != original.response_message_id
        assert revised.metadata == {
            "previous_hash_matches": True,
            "counter_reused": True,
            "existing_completed": False,
            "latest_ordinal_matches": True,
            "prefix_hash_matches": False,
        }
        with pytest.raises(VoiceContextError, match="obsolete"):
            await store.voice_store.check_current(original, completed_ok=True)
        with pytest.raises(VoiceContextError, match="obsolete"):
            await old_graph.checkpointer.aput(before.config, before.checkpoint, before.metadata, {})
        with pytest.raises(VoiceContextError, match="obsolete"):
            await old_graph.checkpointer.aput_writes(before.config, [], "interrupted-old-task")
        assert (await rig.saver.aget_tuple(old_cfg)).config == before.config
        assert not await store.voice_store.finish_admission(original, "completed")
        assert not await store.voice_store.finish_admission(original, "interrupted")

        async def callback(claim):
            if claim.action == "wait":
                claim = await store.voice_store.wait_for_turn(claim)
            cfg = config(owner, claim)
            await prepare_admitted_turn(store.voice_store, claim, cfg, rig.saver)
            if claim.action == "run":
                result = await execute(
                    AdmittedVoiceGraph(graph, store.voice_store, claim),
                    cfg,
                    "final amended fixture",
                )
                assert [m.content for m in result["messages"] if m.type == "human"] == [
                    "final amended fixture"
                ]
                assert await store.voice_store.finish_admission(claim, "completed")
            else:
                assert claim.action == "replay"
            await store.voice_store.check_current(claim, completed_ok=True)
            snapshot = await graph.aget_state(cfg)
            return next(
                m.content for m in snapshot.values["messages"] if m.id == claim.response_message_id
            )

        assert await asyncio.gather(*[callback(c) for c in claims]) == ["final fixture answer"] * 6
        assert model.calls == 2  # Un intento cancelado y una sola generación final.
        replay = await admit(owner, "final amended fixture", counter=100)
        assert (
            replay.action == "replay" and replay.response_message_id == revised.response_message_id
        )
        assert graph.checkpointer is rig.saver
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


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
