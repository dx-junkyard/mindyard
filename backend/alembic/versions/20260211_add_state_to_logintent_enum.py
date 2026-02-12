"""add STATE value to logintent enum

Revision ID: 20260211_state
Revises: 20260208_thread
Create Date: 2026-02-11

"""
from alembic import op


revision = "20260211_state"
down_revision = "20260208_thread"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # PostgreSQL の ENUM に新しい値を追加
    op.execute("ALTER TYPE logintent ADD VALUE IF NOT EXISTS 'STATE'")


def downgrade() -> None:
    # PostgreSQL の ENUM から値を削除するのは困難なため、ダウングレードは非対応
    pass
