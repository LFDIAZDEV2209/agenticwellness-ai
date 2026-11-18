"""Separa los chunks de paciente de las KB del programa.

El downgrade descarta únicamente chunks derivados privados e inactivos para
que la versión anterior nunca recupere datos que no sabe filtrar.
"""

import sqlalchemy as sa

from alembic import op

revision = "67d9e4b31c52"
down_revision = "5a7c3e9d1b42"
branch_labels = None
depends_on = None


def _hnsw(*, program_only: bool) -> None:
    predicate = " WHERE patient_id IS NULL" if program_only else ""
    op.execute(
        "CREATE INDEX ix_knowledge_chunks_embedding_hnsw "
        "ON ai.knowledge_chunks USING hnsw (embedding vector_cosine_ops)" + predicate
    )


def upgrade() -> None:
    op.add_column("knowledge_chunks", sa.Column("patient_id", sa.Uuid()), schema="ai")
    op.add_column(
        "knowledge_chunks",
        sa.Column("retrieval_status", sa.String(16), nullable=False, server_default="active"),
        schema="ai",
    )
    op.alter_column("knowledge_chunks", "knowledge_base_id", nullable=True, schema="ai")
    op.create_check_constraint(
        "ck_knowledge_chunks_scope",
        "knowledge_chunks",
        "(knowledge_base_id IS NULL) <> (patient_id IS NULL)",
        schema="ai",
    )
    op.create_check_constraint(
        "ck_knowledge_chunks_retrieval_status",
        "knowledge_chunks",
        "retrieval_status IN ('active', 'archived', 'deleted', 'disabled')",
        schema="ai",
    )
    op.create_index(
        "ix_knowledge_chunks_patient_status",
        "knowledge_chunks",
        ["patient_id", "retrieval_status"],
        schema="ai",
        postgresql_where=sa.text("patient_id IS NOT NULL"),
    )
    op.drop_index("ix_knowledge_chunks_embedding_hnsw", table_name="knowledge_chunks", schema="ai")
    _hnsw(program_only=True)


def downgrade() -> None:
    op.execute(
        "DELETE FROM ai.knowledge_chunks "
        "WHERE patient_id IS NOT NULL OR retrieval_status <> 'active'"
    )
    op.drop_index("ix_knowledge_chunks_embedding_hnsw", table_name="knowledge_chunks", schema="ai")
    op.drop_index("ix_knowledge_chunks_patient_status", table_name="knowledge_chunks", schema="ai")
    op.drop_constraint("ck_knowledge_chunks_scope", "knowledge_chunks", schema="ai", type_="check")
    op.drop_constraint(
        "ck_knowledge_chunks_retrieval_status", "knowledge_chunks", schema="ai", type_="check"
    )
    op.alter_column("knowledge_chunks", "knowledge_base_id", nullable=False, schema="ai")
    op.drop_column("knowledge_chunks", "retrieval_status", schema="ai")
    op.drop_column("knowledge_chunks", "patient_id", schema="ai")
    _hnsw(program_only=False)
