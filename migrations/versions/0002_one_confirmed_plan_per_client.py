"""At most one CONFIRMED payment plan per client.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-02

Fails if a client already has two CONFIRMED plans — resolve those first
(set the older ones to another status) before upgrading.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_CONFIRMED = sa.text("status = 'CONFIRMED'")


def upgrade() -> None:
    op.create_index(
        "uq_payment_plans_one_confirmed_per_client",
        "payment_plans",
        ["client_id"],
        unique=True,
        sqlite_where=_CONFIRMED,
        postgresql_where=_CONFIRMED,
    )


def downgrade() -> None:
    op.drop_index("uq_payment_plans_one_confirmed_per_client", table_name="payment_plans")
