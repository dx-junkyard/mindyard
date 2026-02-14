"""add profile_data to users

Revision ID: 20260213_profile
Revises: 20260211_state
Create Date: 2026-02-13
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "20260213_profile"
down_revision = "20260211_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "profile_data",
            JSONB,
            nullable=True,
            comment="UserProfiler による蓄積型プロファイル",
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "profile_data")
