"""El contador del agente externo no identifica utterances.

Compatible con lectores/escritores anteriores: no cambia columnas ni la PK.
Downgrade no destructivo: rehúsa recrear UNIQUE si existen contadores repetidos.
Rollback funcional a voz legacy no necesita rollback de esta migración.
"""

from alembic import op

revision = "94e1c2b75a06"
down_revision = "18c4f6a91b20"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "uq_vita_voice_turns_session_id", "vita_voice_turns", type_="unique", schema="ai"
    )


def downgrade() -> None:
    # Impide TOCTOU entre la comprobación y CREATE UNIQUE. No elimina recibos.
    op.execute("LOCK TABLE ai.vita_voice_turns IN SHARE ROW EXCLUSIVE MODE")
    op.execute("""
        DO $$ BEGIN
          IF EXISTS (
            SELECT 1 FROM ai.vita_voice_turns
            GROUP BY session_id, provider_turn HAVING count(*) > 1
          ) THEN
            RAISE EXCEPTION 'voice_counter_duplicates: downgrade refused; retain schema and rollback application to legacy voice';
          END IF;
        END $$
    """)
    op.create_unique_constraint(
        "uq_vita_voice_turns_session_id",
        "vita_voice_turns",
        ["session_id", "provider_turn"],
        schema="ai",
    )
