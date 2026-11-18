"""Persistencia, inmutabilidad y rollback del event log en PostgreSQL aislado."""

import logging
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from alembic import command
from app.db.models import VitaTurn, VitaTurnEvent
from app.graph.graph import build_graph
from app.observability.turn_trace import TurnTraceCollector, attach_collector, persist_turn
from app.tools.registry import TOOLS_BY_NAME
from tests.fakes import FakeToolAwareModel
from tests.test_vita_history_postgres import isolated_test_url

PREVIOUS = "94e1c2b75a06"
REVISION = "5a7c3e9d1b42"


async def test_event_log_persiste_turno_y_revierte_migracion(caplog):
    caplog.set_level(logging.INFO, logger="app.observability.turn_trace")
    url = isolated_test_url()
    root = Path(__file__).resolve().parents[1]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    sync = create_engine(url)
    engine = create_async_engine(url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    with sync.connect() as connection:
        before = set(inspect(connection).get_table_names(schema="ai"))
        previous = MigrationContext.configure(
            connection, opts={"version_table_schema": "ai"}
        ).get_current_revision()

    # Aísla esta revisión de las de voz que se prueban con sus propias bases sintéticas.
    command.stamp(cfg, PREVIOUS)
    try:
        command.upgrade(cfg, REVISION)
        thread_id = "fixture-owner::fixture-thread"
        collector = TurnTraceCollector(thread_id=thread_id, channel="text", patient_id="fixture")
        model = FakeToolAwareModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "c1", "name": "calculate", "args": {"expression": "1+1"}},
                    ],
                    usage_metadata={"input_tokens": 30, "output_tokens": 10, "total_tokens": 40},
                ),
                AIMessage(
                    content="fixture-answer",
                    usage_metadata={
                        "input_tokens": 50,
                        "output_tokens": 5,
                        "total_tokens": 55,
                    },
                ),
            ]
        )
        graph = build_graph(
            model=model, checkpointer=MemorySaver(), tools=[TOOLS_BY_NAME["calculate"]]
        )
        state = await graph.ainvoke(
            {"input": "fixture-request"},
            config=attach_collector({"configurable": {"thread_id": thread_id}}, collector),
        )
        message_ref = state["messages"][-1].id
        trace = collector.finish(status="completed", message_ref=message_ref)
        turn_id = await persist_turn(trace, session_factory=sessions)
        assert turn_id is not None
        async with sessions() as session:
            turn = await session.get(VitaTurn, turn_id)
            rows = (
                await session.scalars(
                    select(VitaTurnEvent)
                    .where(VitaTurnEvent.turn_id == turn_id)
                    .order_by(VitaTurnEvent.seq)
                )
            ).all()
            assert (turn.thread_id, turn.patient_id, turn.channel) == (thread_id, "fixture", "text")
            assert turn.status == "completed" and turn.message_ref == message_ref
            assert (turn.tokens_in, turn.tokens_out) == (80, 15)
            assert turn.cost_usd > 0 and turn.finished_at >= turn.started_at
            assert [event.seq for event in rows] == list(range(1, len(rows) + 1))
            assert {("node", "guardrails"), ("node", "agent"), ("tool", "calculate")} <= {
                (event.kind, event.name) for event in rows
            }
            assert all(event.latency_ms >= 0 for event in rows)
        for table in ("vita_turns", "vita_turn_events"):
            with (
                pytest.raises(IntegrityError, match="vita_event_log_append_only"),
                sync.begin() as sql,
            ):
                sql.execute(text(f"UPDATE ai.{table} SET status = status"))

        assert "vita_turn " in caplog.text and '"tokens_in": 80' in caplog.text
        assert all(
            value not in caplog.text for value in ("fixture-request", "fixture-answer", thread_id)
        )
        caplog.clear()
        command.downgrade(cfg, PREVIOUS)
        with sync.connect() as connection:
            assert set(inspect(connection).get_table_names(schema="ai")) == before | {
                "alembic_version"
            }
            assert (
                connection.scalar(text("SELECT to_regprocedure('ai.vita_event_log_append_only()')"))
                is None
            )
    finally:
        command.downgrade(cfg, PREVIOUS)
        command.stamp(cfg, previous or "base")
        await engine.dispose()
        sync.dispose()
