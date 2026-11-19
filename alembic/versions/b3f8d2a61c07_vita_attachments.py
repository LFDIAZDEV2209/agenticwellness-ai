"""Adjuntos efímeros del chat Vita (Ola B, 6.3).

Tabla aislada del expediente y del RAG: solo binario validado + metadatos
mínimos con `expires_at`. La purga periódica elimina filas expiradas.
"""

import sqlalchemy as sa

from alembic import op

revision = "b3f8d2a61c07"
down_revision = "67d9e4b31c52"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "vita_attachments",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("thread_id", sa.String(128)),
        sa.Column("kind", sa.String(8), nullable=False),
        sa.Column("content_type", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("page_count", sa.Integer()),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("kind IN ('image', 'pdf')", name="ck_vita_attachments_kind"),
        sa.CheckConstraint("size_bytes > 0", name="ck_vita_attachments_size"),
        sa.CheckConstraint("expires_at > created_at", name="ck_vita_attachments_ttl"),
        schema="ai",
    )
    op.create_index(
        "ix_vita_attachments_patient",
        "vita_attachments",
        ["patient_id", "expires_at"],
        schema="ai",
    )
    op.create_index(
        "ix_vita_attachments_expires_at", "vita_attachments", ["expires_at"], schema="ai"
    )


def downgrade() -> None:
    # Los adjuntos son efímeros por contrato: retroceder los descarta.
    op.drop_table("vita_attachments", schema="ai")
