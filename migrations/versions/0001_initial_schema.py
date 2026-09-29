"""Initial schema (replaces the old schema.sql).

Revision ID: 0001
Revises:
Create Date: 2026-09-29
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_now = sa.text("CURRENT_TIMESTAMP")


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("nom", sa.String(255), nullable=False),
        sa.Column("prenom", sa.String(255), nullable=False),
        sa.Column("email_or_phone", sa.String(255), nullable=False),
        sa.Column("telephone", sa.String(50)),
    )

    op.create_table(
        "contrats",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("statut", sa.String(50), nullable=False, server_default="ACTIVE"),
        sa.Column("type_contrat", sa.String(50), nullable=False, server_default="POSTPAYE"),
        sa.Column("numero_contrat", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=_now),
    )
    op.create_index("ix_contrats_user_id", "contrats", ["user_id"])

    op.create_table(
        "profils_esim",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("contrat_id", sa.Integer, sa.ForeignKey("contrats.id"), nullable=False),
        sa.Column("statut", sa.String(50), nullable=False, server_default="ACTIF"),
    )
    op.create_index("ix_profils_esim_contrat_id", "profils_esim", ["contrat_id"])

    op.create_table(
        "contrats_sim",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("statut", sa.String(50), nullable=False, server_default="ACTIVE"),
        sa.Column("type_contrat", sa.String(50), nullable=False, server_default="POSTPAYE"),
        sa.Column("numero_contrat", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=_now),
    )
    op.create_index("ix_contrats_sim_user_id", "contrats_sim", ["user_id"])

    op.create_table(
        "souscriptions_sim",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("contrat_sim_id", sa.Integer, sa.ForeignKey("contrats_sim.id"), nullable=False),
        sa.Column("statut", sa.String(50), nullable=False, server_default="ACTIF"),
    )
    op.create_index("ix_souscriptions_sim_contrat_sim_id", "souscriptions_sim", ["contrat_sim_id"])

    op.create_table(
        "factures",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("contrat_id", sa.Integer, sa.ForeignKey("contrats.id")),
        sa.Column("contrat_sim_id", sa.Integer, sa.ForeignKey("contrats_sim.id")),
        sa.Column("montant", sa.Numeric(12, 2), nullable=False),
        sa.Column("date_echeance", sa.Date, nullable=False),
        sa.Column("date_paiement", sa.Date),
        sa.Column("statut", sa.String(50), nullable=False, server_default="EN_ATTENTE"),
    )
    op.create_index("ix_factures_contrat_id", "factures", ["contrat_id"])
    op.create_index("ix_factures_contrat_sim_id", "factures", ["contrat_sim_id"])

    op.create_table(
        "action_recouvrement",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("contrat_id", sa.Integer, nullable=False),
        sa.Column("type_action", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=_now),
    )
    op.create_index("ix_action_recouvrement_contrat_id", "action_recouvrement", ["contrat_id"])

    op.create_table(
        "client_interactions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("client_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("session_id", sa.String(100)),
        sa.Column("summary", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=_now),
    )
    op.create_index("ix_client_interactions_client_id", "client_interactions", ["client_id"])

    op.create_table(
        "admin_notifications",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("type", sa.String(50), nullable=False, server_default="PAYMENT_PLAN"),
        sa.Column("client_id", sa.Integer),
        sa.Column("reference_id", sa.Integer),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("read", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=_now),
    )

    op.create_table(
        "tickets",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("client_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("numero_ticket", sa.String(100)),
        sa.Column("complaint_type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="OPEN"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=_now),
    )
    op.create_index("ix_tickets_client_id", "tickets", ["client_id"])

    op.create_table(
        "payment_plans",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("client_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("session_id", sa.String(100), nullable=False),
        sa.Column("plan_data", sa.Text, nullable=False),
        sa.Column("summary", sa.Text),
        sa.Column("status", sa.String(50), nullable=False, server_default="CONFIRMED"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=_now),
        sa.Column("updated_at", sa.DateTime),
    )
    op.create_index("ix_payment_plans_client_id", "payment_plans", ["client_id"])


def downgrade() -> None:
    for table in (
        "payment_plans", "tickets", "admin_notifications", "client_interactions",
        "action_recouvrement", "factures", "souscriptions_sim", "contrats_sim",
        "profils_esim", "contrats", "users",
    ):
        op.drop_table(table)
