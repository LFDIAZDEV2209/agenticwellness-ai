"""Event log append-only de Vita: ai.vita_turns + ai.vita_turn_events (tarea 2.4).

Solo metadata técnica por turno (canal, estado, tokens, costo, latencia, códigos
de error y referencias), nunca contenido clínico. Un trigger rechaza UPDATE en
ambas tablas: corregir = insertar. DELETE se permite para retención/derechos del
titular (la FK de eventos es ON DELETE CASCADE).

El downgrade elimina las tablas, sus datos, los triggers y la función.
"""

import sqlalchemy as sa

from alembic import op

revision = "5a7c3e9d1b42"
down_revision = "94e1c2b75a06"
branch_labels = None
depends_on = None

_CHANNELS = "('text', 'voice', 'whatsapp')"
_STATUSES = "('completed', 'failed', 'cancelled', 'rejected')"
_KINDS = "('node', 'tool', 'llm', 'policy')"


def upgrade() -> None:
    op.create_table(
        "vita_turns",
        sa.Column(
            "id",
            sa.Uuid(as_uuid=False),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("thread_id", sa.String(128), nullable=False),
        sa.Column("patient_id", sa.String(128)),
        sa.Column("channel", sa.String(16), nullable=False),
        sa.Column("agent_id", sa.String(128)),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("latency_ms", sa.Integer, nullable=False),
        sa.Column("tokens_in", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("tokens_out", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("cost_usd", sa.Numeric(12, 6)),
        sa.Column("error_code", sa.String(64)),
        sa.Column("execution_id", sa.String(64)),
        sa.Column("message_ref", sa.String(200)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(f"channel IN {_CHANNELS}", name="ck_vita_turns_channel"),
        sa.CheckConstraint(f"status IN {_STATUSES}", name="ck_vita_turns_status"),
        sa.CheckConstraint("finished_at >= started_at", name="ck_vita_turns_time_order"),
        schema="ai",
    )
    op.create_index(
        "ix_vita_turns_thread_started", "vita_turns", ["thread_id", "started_at"], schema="ai"
    )
    op.create_index(
        "ix_vita_turns_patient_started", "vita_turns", ["patient_id", "started_at"], schema="ai"
    )
    op.create_table(
        "vita_turn_events",
        sa.Column(
            "id",
            sa.Uuid(as_uuid=False),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "turn_id",
            sa.Uuid(as_uuid=False),
            sa.ForeignKey("ai.vita_turns.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("tokens_in", sa.Integer),
        sa.Column("tokens_out", sa.Integer),
        sa.Column("error_code", sa.String(64)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("turn_id", "seq", name="uq_vita_turn_events_turn_seq"),
        sa.CheckConstraint(f"kind IN {_KINDS}", name="ck_vita_turn_events_kind"),
        schema="ai",
    )
    # Append-only real: la BD rechaza cualquier UPDATE, también desde SQL manual.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION ai.vita_event_log_append_only() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          RAISE EXCEPTION 'vita_event_log_append_only: % no permitido en %', TG_OP, TG_TABLE_NAME
            USING ERRCODE = 'restrict_violation';
        END $$
        """
    )
    for table in ("vita_turns", "vita_turn_events"):
        op.execute(
            f"CREATE TRIGGER trg_{table}_append_only BEFORE UPDATE ON ai.{table} "
            "FOR EACH ROW EXECUTE FUNCTION ai.vita_event_log_append_only()"
        )


def downgrade() -> None:
    for table in ("vita_turn_events", "vita_turns"):
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_append_only ON ai.{table}")
    op.drop_table("vita_turn_events", schema="ai")
    op.drop_index("ix_vita_turns_patient_started", table_name="vita_turns", schema="ai")
    op.drop_index("ix_vita_turns_thread_started", table_name="vita_turns", schema="ai")
    op.drop_table("vita_turns", schema="ai")
    op.execute("DROP FUNCTION IF EXISTS ai.vita_event_log_append_only()")
